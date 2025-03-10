from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
# from bitsandbytes import quantize
from datasets import load_dataset
import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dataset_path = "/scratch/zl3057/processed_txt"

dataset = load_dataset("text", data_files={
    "train": f"{dataset_path}/train/*.txt",
    "test": f"{dataset_path}/test/*.txt"
})


model_name = "/scratch/zl3057/llama-3b-hf"  # Pretrained model path
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    tokenized_inputs = tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=42
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs


tokenized_datasets = dataset.map(tokenize_function, batched=True, load_from_cache_file=False, num_proc=1)
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]
# training_args = TrainingArguments(
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     gradient_accumulation_steps=4, 
#     fp16=True,  # Mixed precision training
#     bf16=False,
#     optim="adamw_bnb_8bit",
#     save_total_limit=2,
#     dataloader_num_workers=4,
#     output_dir="./checkpoints/",
#     warmup_steps=5,
#     max_steps=25,
#     eval_strategy="steps",
#     eval_steps=25,
#     save_steps=25,
#     logging_steps=5,
#     learning_rate=1e-4,
#     report_to="none",
# )

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # Reduced from 8 to 4 to use less memory
    warmup_steps=5,
    max_steps=500,  # training steps reduced
    learning_rate=1e-4,  # learning rate reduced
    fp16=True,  # Mixed precision training for memory optimization
    logging_steps=5,
    save_steps=25,
    evaluation_strategy="steps",
    eval_steps=25,
    save_total_limit=2,
    gradient_checkpointing=True,  # Gradient checkpointing for memory optimization purpose
    report_to="none",
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    optim="paged_adamw_8bit"  # Use 8-bit optimizer for memory optimization purpose
)

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype="float16",
# )

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_compute_on_cpu=False,
)


# model = AutoModelForCausalLM.from_pretrained(
#     model_name, 
#     quantization_config=bnb_config, 
#     device_map="auto",
#     use_cache=False)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False # Disable cache to save memory
)



# model = prepare_model_for_kbit_training(model)

# lora_config = LoraConfig(
#     r=8,  
#     lora_alpha=16,
#     lora_dropout=0.1,
#     bias="none",
#     task_type=TaskType.CAUSAL_LM,
# )

prepare_model_for_kbit_training(model)
print("Apply LoRA for memory-efficient fine-tuning")
lora_config = LoraConfig(
    r=8,  # LoRA rank is set to 8
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target all attention modules
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Model prepared with LoRA")

model.gradient_checkpointing_enable()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()


trainer.save_model("./checkpoints/final_model")  
tokenizer.save_pretrained("./checkpoints/final_model")

model.save_pretrained("./checkpoints/lora_model")

print("Training complete and model saved.")
