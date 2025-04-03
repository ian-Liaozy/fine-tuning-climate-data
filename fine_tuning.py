import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe

# ---- Setup ----
def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank

# ---- Model Loading ----
def get_model(model_name, parallel_mode="none", devices=None):
    rank = setup_distributed()

    # ---- Common config for LoRA, quantization, etc. ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    if parallel_mode == "pipeline":
        # ----- Load Full Model -----
        model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False)
        model = model.eval()  # needed for tracing

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Example microbatch input for tracing
        dummy_input = torch.randint(0, tokenizer.vocab_size, (4, 42)).cuda(rank)

        # ---- Pipeline Split Spec ----
        pipe = pipeline(
            module=model,
            mb_args=(dummy_input,),
            split_spec={f"model.layers.{len(model.model.layers)//2}": SplitPoint.BEGINNING},
        )

        stage_mod = pipe.get_stage_module(rank)
        model = pipe.build_stage(stage_index=rank, device=f"cuda:{rank}", group=None)

        return model, tokenizer

    elif parallel_mode == "tensor":
        model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False).cuda(rank)

        tp_mesh = DeviceMesh("cuda", list(range(dist.get_world_size())))

        # Traverse model and apply TP to specific layers
        for name, module in model.named_modules():
            if hasattr(module, "self_attn") and hasattr(module, "mlp"):
                try:
                    ColwiseParallel(module.self_attn.q_proj, tp_mesh)
                    ColwiseParallel(module.self_attn.k_proj, tp_mesh)
                    ColwiseParallel(module.self_attn.v_proj, tp_mesh)
                    RowwiseParallel(module.self_attn.out_proj, tp_mesh)
                    ColwiseParallel(module.mlp.fc1, tp_mesh)
                    RowwiseParallel(module.mlp.fc2, tp_mesh)
                    print(f"[TP] Applied to {name}")
                except Exception as e:
                    print(f"[TP] Skipped {name}: {e}")

        return model, AutoTokenizer.from_pretrained(model_name)

    elif parallel_mode == "data":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            use_cache=False,
        ).cuda(rank)
        model = DDP(model, device_ids=[rank])
        return model, AutoTokenizer.from_pretrained(model_name)

    else:  # Single GPU (none)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, use_cache=False
        ).cuda()
        return model, AutoTokenizer.from_pretrained(model_name)

# ---- Main ----
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel-mode", type=str, default="none",
                        choices=["none", "data", "tensor", "pipeline"])
    args = parser.parse_args()

    dataset_path = "/scratch/zl3057/processed_txt"
    model_name = "/scratch/zl3057/llama-3b-hf"

    # ---- Load Dataset ----
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dataset = load_dataset("text", data_files={
        "train": f"{dataset_path}/train/*.txt",
        "test": f"{dataset_path}/test/*.txt"
    })

    # ---- Get model + tokenizer ----
    model, tokenizer = get_model(model_name, parallel_mode=args.parallel_mode)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=42)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_datasets = dataset.map(tokenize_function, batched=True, load_from_cache_file=True, num_proc=1)
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]
    small_eval_dataset = test_dataset.select(range(500))

    # ---- Training args ----
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        fp16=False,
        bf16=False,
        optim="adamw_bnb_8bit",
        output_dir="./checkpoints/",
        warmup_steps=5,
        max_steps=500,
        evaluation_strategy="steps",
        eval_steps=25,
        save_steps=25,
        logging_steps=25,
        save_total_limit=2,
        dataloader_num_workers=4,
        learning_rate=5e-5,
        remove_unused_columns=False,
        report_to="none",
    )

    # ---- Trainer ----
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

if __name__ == "__main__":
    main()
