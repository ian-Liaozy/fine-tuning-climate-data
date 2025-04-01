import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.tensor.parallel import parallelize_module
# from torch.distributed.pipeline.sync import Pipe
from fairscale.nn.pipe import Pipe
from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
import argparse
from transformers import BitsAndBytesConfig
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel



def setup_distributed():
    if dist.is_initialized():
        return dist.get_rank()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank

def get_model(model_name, parallel_mode="none", devices=None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


    if parallel_mode == "data":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            use_cache=False,
        )
        rank = setup_distributed()
        model = DDP(model, device_ids=[rank])

    elif parallel_mode == "tensor":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_cache=False,
        )
        rank = setup_distributed()
        model = model.cuda(rank)
        # model = parallelize_module(model, parallel_mode="column", devices=devices)
        
        global tp_mesh
        tp_mesh = DeviceMesh("cuda", list(range(dist.get_world_size())))

        # Traverse and apply
        for name, module in model.named_modules():
            if hasattr(module, "self_attn") and hasattr(module, "mlp"):
                try:
                    ColwiseParallel(module.self_attn.q_proj, tp_mesh)
                    ColwiseParallel(module.self_attn.k_proj, tp_mesh)
                    ColwiseParallel(module.self_attn.v_proj, tp_mesh)
                    RowwiseParallel(module.self_attn.out_proj, tp_mesh)
                    ColwiseParallel(module.mlp.fc1, tp_mesh)  # First linear layer in MLP
                    RowwiseParallel(module.mlp.fc2, tp_mesh)  # Second linear layer in MLP
                    print(f"[TP] Applied tensor parallel to {name}")
                except Exception as e:
                    print(f"[TP] Skipped {name} due to: {e}")



    elif parallel_mode == "pipeline":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            use_cache=False,
        )
        rank = setup_distributed()
        model = model.cuda(rank)
        model = Pipe(model, balance=[3, 3], devices=devices)

    else:
        model = model.cuda()

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel-mode", type=str, default="none",
                        choices=["none", "data", "tensor", "pipeline"])
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dataset_path = "/scratch/zl3057/processed_txt"

    dataset = load_dataset("text", data_files={
        "train": f"{dataset_path}/train/*.txt",
        "test": f"{dataset_path}/test/*.txt"
    })

    model_name = "/scratch/zl3057/llama-3b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        tokenized_inputs = tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=42
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True, load_from_cache_file=True, num_proc=1)
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]
    small_eval_dataset = test_dataset.select(range(500))



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


    model = get_model(model_name, parallel_mode=args.parallel_mode, devices=[0, 1])


    print("Model ready")

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
