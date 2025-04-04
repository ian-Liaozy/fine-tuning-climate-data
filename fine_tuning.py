import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    BitsAndBytesConfig
)
from datasets import load_dataset

def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank

def split_llama(model, stage_idx, num_stages):
    """Manually partition model for 2-stage pipeline."""
    assert num_stages == 2
    layers = model.model.layers
    n = len(layers)
    half = n // 2

    if stage_idx == 0:
        model.model.layers = layers[:half]
        model.model.norm = None
        model.lm_head = None
    else:
        model.model.embed_tokens = None
        model.model.layers = layers[half:]
    return model

def get_model(model_name, parallel_mode="none", devices=None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    rank = setup_distributed()
    model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False)

    if parallel_mode == "pipeline":
        model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False)
        layers = model.model.layers
        half = len(layers) // 2

        # === Manual partition ===
        if rank == 0:
            model.model.layers = layers[:half]
            model.model.norm = None
            model.lm_head = None
        else:
            model.model.embed_tokens = None
            model.model.layers = layers[half:]

        model = model.to(rank)  # ensure all parts on CUDA

        # === PipelineStage with runtime shape inference ===
        stage = PipelineStage(
            submodule=model,
            stage_index=rank,
            num_stages=2,
            device=torch.device(f"cuda:{rank}"),
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return stage, tokenizer

    elif parallel_mode == "tensor":
        model = model.cuda(rank)
        mesh = DeviceMesh("cuda", list(range(dist.get_world_size())))
        for name, module in model.named_modules():
            if hasattr(module, "self_attn") and hasattr(module, "mlp"):
                try:
                    ColwiseParallel(module.self_attn.q_proj, mesh)
                    ColwiseParallel(module.self_attn.k_proj, mesh)
                    ColwiseParallel(module.self_attn.v_proj, mesh)
                    RowwiseParallel(module.self_attn.out_proj, mesh)
                    ColwiseParallel(module.mlp.fc1, mesh)
                    RowwiseParallel(module.mlp.fc2, mesh)
                    print(f"[TP] Applied tensor parallel to {name}")
                except Exception as e:
                    print(f"[TP] Skipped {name} due to: {e}")
        return model, AutoTokenizer.from_pretrained(model_name)

    elif parallel_mode == "data":
        model = AutoModelForCausalLM.from_pretrained(model_name,
            quantization_config=bnb_config, use_cache=False)
        model = DDP(model.cuda(rank), device_ids=[rank])
        return model, AutoTokenizer.from_pretrained(model_name)

    else:
        model = model.cuda()
        return model, AutoTokenizer.from_pretrained(model_name)

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

    model, tokenizer = get_model(model_name, args.parallel_mode)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        result = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=42)
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = dataset.map(tokenize_fn, batched=True, num_proc=1)
    train_ds = tokenized["train"]
    eval_ds = tokenized["test"].select(range(500))

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
        max_steps=500,
        evaluation_strategy="steps",
        eval_steps=25,
        save_steps=25,
        logging_steps=25,
        learning_rate=5e-5,
        report_to="none",
        remove_unused_columns=False,
        save_safetensors=False,
        ddp_find_unused_parameters=False,
    )

    if parallel_mode == "pipeline":
        model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False)

        # === Manual split ===
        layers = model.model.layers
        half = len(layers) // 2

        if rank == 0:
            model.model.layers = layers[:half]
            model.model.norm = None
            model.lm_head = None
        else:
            model.model.embed_tokens = None
            model.model.layers = layers[half:]

        model = model.to(rank)  # ensure device match!

        # === PipelineStage with runtime shape inference ===
        stage = PipelineStage(
            submodule=model,
            stage_index=rank,
            num_stages=2,
            device=torch.device(f"cuda:{rank}"),
            # ✅ no input_args => use runtime shape inference
        )

        return stage, AutoTokenizer.from_pretrained(model_name)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()
    trainer.save_model("./checkpoints/final_dist_model")
    tokenizer.save_pretrained("./checkpoints/final_dist_model")
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()
