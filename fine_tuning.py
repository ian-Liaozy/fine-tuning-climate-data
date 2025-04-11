import os
import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe, SplitPoint
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import deepspeed
import math

def setup_distributed(local_rank=None):
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Correct: use local_rank for setting CUDA device
    if local_rank is not None and local_rank >= 0:
        torch.cuda.set_device(local_rank)
    else:
        local_rank = rank
        torch.cuda.set_device(rank)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def get_model(model_name, parallel_mode="none", local_rank=None):
    rank, world_size, local_rank = setup_distributed(local_rank)

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
        # model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, use_cache=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=None,  # Explicitly set to None
            use_cache=False,
            torch_dtype=torch.float16
        ).to(f"cuda:{local_rank}")

        # model = model.cuda(local_rank)
        model = prepare_model_for_kbit_training(model)

        # Add LoRA adapters
        lora_config = LoraConfig(
            r=8,  
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model = DDP(model, device_ids=[rank])
        return model, tokenizer

    elif parallel_mode == "tensor":
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.cuda(rank)

        mesh = init_device_mesh("cuda", [world_size])


        parallelize_plan = {
            "model.layers.self_attn.q_proj": ColwiseParallel(),
            "model.layers.self_attn.k_proj": ColwiseParallel(),
            "model.layers.self_attn.v_proj": ColwiseParallel(),
            "model.layers.self_attn.o_proj": RowwiseParallel(),
            "model.layers.mlp.gate_proj": ColwiseParallel(),
            "model.layers.mlp.up_proj": ColwiseParallel(),
            "model.layers.mlp.down_proj": RowwiseParallel(),
        }
    
        model = parallelize_module(model, mesh, parallelize_plan)
        model = model.to(f"cuda:{local_rank}")

        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"], 
            lora_dropout=0.1,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        return model, tokenizer

    elif parallel_mode == "pipeline":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=None,
            use_cache=False,
            torch_dtype=torch.float16,
        ).to(f"cuda:{local_rank}")

        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        return model, tokenizer
    elif parallel_mode == "mixed":
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.cuda(rank)

        mesh = init_device_mesh("cuda", [world_size])


        parallelize_plan = {
            "model.layers.self_attn.q_proj": ColwiseParallel(),
            "model.layers.self_attn.k_proj": ColwiseParallel(),
            "model.layers.self_attn.v_proj": ColwiseParallel(),
            "model.layers.self_attn.o_proj": RowwiseParallel(),
            "model.layers.mlp.gate_proj": ColwiseParallel(),
            "model.layers.mlp.up_proj": ColwiseParallel(),
            "model.layers.mlp.down_proj": RowwiseParallel(),
        }
    
        model = parallelize_module(model, mesh, parallelize_plan)
        model = model.to(f"cuda:{local_rank}")

        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"], 
            lora_dropout=0.1,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model = DDP(model, device_ids=[rank])
        return model, tokenizer
    else:
        # deepspeed
        # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=None,  # Explicitly set to None
            use_cache=False,
            torch_dtype=torch.float16
        ).to(f"cuda:{local_rank}")

        # model = model.cuda(local_rank)
        model = prepare_model_for_kbit_training(model)

        # Add LoRA adapters
        lora_config = LoraConfig(
            r=8,  
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        return model, tokenizer

def tokenize_function(tokenizer, examples):
    tokenized_inputs = tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=64
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do-eval-only", action="store_true", help="Only run evaluation and report perplexity")
    parser.add_argument("--parallel-mode", type=str, default="none",
                        choices=["none", "data", "tensor", "pipeline", "mixed"],)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by DeepSpeed")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dataset_path = "/scratch/zl3057/processed_txt"
    model_name = "/scratch/zl3057/llama-3b-hf"

    dataset = load_dataset("text", data_files={
        "train": f"{dataset_path}/train/*.txt",
        "test": f"{dataset_path}/test/*.txt"
    })

    model, tokenizer = get_model(model_name, parallel_mode=args.parallel_mode, local_rank=args.local_rank)
    if args.parallel_mode == "pipeline":
        ds_config_name = "ds_config_pipeline.json"
    elif args.parallel_mode == "tensor":
        ds_config_name = "ds_config_tensor.json"
    elif args.parallel_mode == "data":
        ds_config_name = "ds_config_data.json"
    elif args.parallel_mode == "none":
        ds_config_name = "ds_config_base.json"
    elif args.parallel_mode == "mixed":
        ds_config_name = "ds_config.json"
    else:
        raise ValueError(f"Unknown parallel mode: {args.parallel_mode}")
    
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        fp16=True,
        bf16=False,
        optim="adamw_bnb_8bit",
        save_total_limit=2,
        dataloader_num_workers=4,
        output_dir="./checkpoints/",
        warmup_steps=5,
        max_steps=500,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        logging_steps=50,
        learning_rate=5e-5,
        report_to="none",
        remove_unused_columns=False,
        save_safetensors=False,
        ddp_find_unused_parameters=False,
        deepspeed=ds_config_name if args.do_eval_only == None else None,
        label_names=["labels"]
    )
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(tokenizer, examples),
        batched=True,
        load_from_cache_file=True,
        num_proc=1
    )
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]
    small_eval_dataset = test_dataset.select(range(500))

    if args.do_eval_only:
        MODEL_PATH = "./checkpoints/final_dist_model_" + args.parallel_mode
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map=None, load_in_4bit=True)
        eval_args = TrainingArguments(
            output_dir="./eval_results_dist_" + args.parallel_mode,
            per_device_eval_batch_size=4,
            dataloader_num_workers=4,
            do_eval=True,
            report_to="none",
            fp16=True, 
            bf16=False,
        )
        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=test_dataset,
        )
        metrics = trainer.evaluate()
        eval_loss = metrics["eval_loss"]
        perplexity = math.exp(eval_loss)
        print(f"\n==== Evaluation Results ====")
        print(f"Eval loss: {eval_loss:.4f}")
        print(f"Perplexity: {perplexity:.2f}")
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    

    # Set dataset format to return PyTorch tensors.
    train_dataset.set_format("torch", columns=["input_ids", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "labels"])

    
    # rank = dist.get_rank()
    # device = torch.device(f"cuda:{rank}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=small_eval_dataset,
    )
    trainer.train()

    trainer.save_model("./checkpoints/final_dist_model_" + args.parallel_mode)
    tokenizer.save_pretrained("./checkpoints/final_dist_model_" + args.parallel_mode)
    print("Training complete and model saved.")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
