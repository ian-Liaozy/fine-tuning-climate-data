import os
import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe, SplitPoint
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset

def setup_distributed():
    if dist.is_initialized():
        return dist.get_rank()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank

def patch_rotary_emb(model):
    # Iterate over each transformer layer in the model.
    for layer in model.model.layers:
        # Check if the self-attention module has a rotary_emb attribute that is not None.
        if getattr(layer.self_attn, 'rotary_emb', None) is None:
            print("Patching rotary_emb for a layer with a dummy function.")
            def dummy_rotary(position_ids):
                head_dim = model.config.hidden_size // model.config.num_attention_heads
                batch_size, seq_len = position_ids.shape
                cos = torch.ones(batch_size, seq_len, head_dim, device=position_ids.device)
                sin = torch.zeros(batch_size, seq_len, head_dim, device=position_ids.device)
                return (cos, sin)
            layer.self_attn.rotary_emb = dummy_rotary
        else:
            # If rotary_emb exists (and is callable), wrap it to handle None outputs.
            orig_rotary = layer.self_attn.rotary_emb
            def patched_rotary(position_ids, orig_rotary=orig_rotary):
                result = orig_rotary(position_ids)
                if result is None:
                    head_dim = model.config.hidden_size // model.config.num_attention_heads
                    batch_size, seq_len = position_ids.shape
                    cos = torch.ones(batch_size, seq_len, head_dim, device=position_ids.device)
                    sin = torch.zeros(batch_size, seq_len, head_dim, device=position_ids.device)
                    return (cos, sin)
                return result
            layer.self_attn.rotary_emb = patched_rotary


def get_model(model_name, parallel_mode="none", devices=None):
    rank = setup_distributed()
    world_size = dist.get_world_size()

    # Use your own tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    if parallel_mode == "data":
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, use_cache=False)
        model = DDP(model, device_ids=[rank])
        return model, tokenizer

    elif parallel_mode == "tensor":
        class AllGatherFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input_tensor, world_size):
                ctx.world_size = world_size
                ctx.input_shape = input_tensor.shape
                
                output = [torch.zeros_like(input_tensor) for _ in range(world_size)]
                dist.all_gather(output, input_tensor)
                
                return torch.cat(output, dim=-1)
            
            @staticmethod
            def backward(ctx, grad_output):
                world_size = ctx.world_size
                input_shape = ctx.input_shape
                
                dim_size = grad_output.size(-1) // world_size
                
                rank = dist.get_rank()
                grad_slice = grad_output.narrow(-1, rank * dim_size, dim_size)
                
                return grad_slice, None

        def all_gather_with_grad(tensor, world_size=None):
            if world_size is None:
                world_size = dist.get_world_size()
                
            return AllGatherFunction.apply(tensor, world_size)
        class TPLinear(nn.Module):
            def __init__(self, in_features, out_features, bias=True):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                
                self.world_size = dist.get_world_size()
                self.rank = dist.get_rank()
                
                self.out_features_per_gpu = out_features // self.world_size
                
                self.linear = nn.Linear(in_features, self.out_features_per_gpu, bias=bias)
            
            def forward(self, x):
                local_output = self.linear(x)
                
                return all_gather_with_grad(local_output)
        class TPLlamaModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = nn.Embedding(32000, 768)
                
                self.layers = nn.ModuleList([
                    TPLinear(768, 768) for _ in range(6)
                ])
                
                self.lm_head = TPLinear(768, 32000)
            
            def forward(self, input_ids, labels=None):
                x = self.embeddings(input_ids)
                for layer in self.layers:
                    x = nn.functional.gelu(layer(x))
                logits = self.lm_head(x)
                
                loss = None
                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, 32000), labels.view(-1))
                
                return type('Output', (), {'loss': loss, 'logits': logits})()
        # model = AutoModelForCausalLM.from_pretrained(model_name)
        # model = model.cuda(rank)
        # tp_mesh = DeviceMesh("cuda", list(range(world_size)))
        # for name, module in model.named_modules():
        #     if hasattr(module, "self_attn") and hasattr(module, "mlp"):
        #         try:
        #             ColwiseParallel(module.self_attn.q_proj, tp_mesh)
        #             ColwiseParallel(module.self_attn.k_proj, tp_mesh)
        #             ColwiseParallel(module.self_attn.v_proj, tp_mesh)
        #             RowwiseParallel(module.self_attn.out_proj, tp_mesh)
        #             ColwiseParallel(module.mlp.fc1, tp_mesh)
        #             RowwiseParallel(module.mlp.fc2, tp_mesh)
        #             print(f"[TP] Applied tensor parallel to {name}")
        #         except Exception as e:
        #             print(f"[TP] Skipped {name} due to: {e}")
        # return model, tokenizer
        device = torch.device("cuda", rank)
        model = TPLlamaModel().to(device)

        return model, tokenizer
    elif parallel_mode == "pipeline":
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # Patch the rotary embedding to avoid None returns.
        patch_rotary_emb(model)
        layers = model.model.layers
        half = len(layers) // 2

        class Stage0(nn.Module):
            def __init__(self, embed_tokens, layers):
                super().__init__()
                self.embed_tokens = embed_tokens
                self.layers = nn.ModuleList(layers)

            def forward(self, input_ids, position_ids=None):
                if isinstance(input_ids, tuple):
                    input_ids = input_ids[0]
                print("Stage0 input_ids shape:", input_ids.shape)
                print("Stage0 position_ids:", position_ids)
                batch_size, seq_length = input_ids.shape
                if position_ids is None:
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
                    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                hidden_states = self.embed_tokens(input_ids)
                for layer in self.layers:
                    hidden_states = layer(hidden_states, position_ids=position_ids)[0]
                return hidden_states

        class Stage1(nn.Module):
            def __init__(self, layers, norm, lm_head):
                super().__init__()
                self.layers = nn.ModuleList(layers)
                self.norm = norm
                self.lm_head = lm_head

            def forward(self, hidden_states, position_ids=None):
                if position_ids is None:
                    batch_size, seq_length, _ = hidden_states.shape
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device)
                    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                for layer in self.layers:
                    hidden_states = layer(hidden_states, position_ids=position_ids)[0]
                hidden_states = self.norm(hidden_states)
                return self.lm_head(hidden_states)

        if rank == 0:
            stage_module = Stage0(model.model.embed_tokens, layers[:half])
        else:
            stage_module = Stage1(layers[half:], model.model.norm, model.lm_head)

        stage_module = stage_module.to(rank)
        stage = PipelineStage(
            submodule=stage_module,
            stage_index=rank,
            num_stages=2,
            device=torch.device(f"cuda:{rank}")
        )
        return stage, tokenizer

    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
        model = model.cuda(rank)
        return model, tokenizer

def tokenize_function(tokenizer, examples):
    tokenized_inputs = tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=42
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel-mode", type=str, default="none",
                        choices=["none", "data", "tensor", "pipeline"])
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dataset_path = "/scratch/zl3057/processed_txt"
    model_name = "/scratch/zl3057/llama-3b-hf"

    dataset = load_dataset("text", data_files={
        "train": f"{dataset_path}/train/*.txt",
        "test": f"{dataset_path}/test/*.txt"
    })

    model, tokenizer = get_model(model_name, parallel_mode=args.parallel_mode)

    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(tokenizer, examples),
        batched=True,
        load_from_cache_file=True,
        num_proc=1
    )
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]
    small_eval_dataset = test_dataset.select(range(500))

    # Set dataset format to return PyTorch tensors.
    train_dataset.set_format("torch", columns=["input_ids", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "labels"])

    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        fp16=False,
        bf16=False,
        optim="adamw_bnb_8bit",
        save_total_limit=2,
        dataloader_num_workers=4,
        output_dir="./checkpoints/",
        warmup_steps=5,
        max_steps=25,
        eval_strategy="steps",
        eval_steps=25,
        save_steps=25,
        logging_steps=25,
        learning_rate=5e-5,
        report_to="none",
        remove_unused_columns=False,
        save_safetensors=False,
        ddp_find_unused_parameters=False,
    )

    if args.parallel_mode == "pipeline":
        from torch.utils.data import DataLoader
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")
        schedule = ScheduleGPipe(model, n_microbatches=4)

        # Create a DataLoader from the train_dataset.
        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0).expand(input_ids.shape[0], -1)
            if rank == 0:
                outputs = schedule.step(input_ids, position_ids=position_ids)
                print("Pipeline output shape:", outputs[0].shape)
            else:
                _ = schedule.step()
            break  # Process one batch for demonstration.
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=small_eval_dataset,
        )
        trainer.train()
        trainer.save_model("./checkpoints/final_dist_model")
        tokenizer.save_pretrained("./checkpoints/final_dist_model")
        print("Training complete and model saved.")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
