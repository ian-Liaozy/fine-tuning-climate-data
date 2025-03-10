from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
# from bitsandbytes import quantize
from datasets import load_dataset

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
        examples["text"], truncation=True, padding="max_length", max_length=512
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs


tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4, 
    fp16=True,  # Mixed precision training
    bf16=False,
    optim="adamw_bnb_8bit",
    save_total_limit=2,
    dataloader_num_workers=4,
    output_dir="./checkpoints/",
    warmup_steps=5,
    max_steps=500,
    eval_strategy="steps",
    eval_steps=25,
    save_steps=50,
    logging_steps=5,
    learning_rate=1e-4,
    report_to="none",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=bnb_config, 
    device_map="auto",
    attn_implementation="flash_attention_2",
    use_cache=False)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,  
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

print("Model prepared with LoRA")

model.gradient_checkpointing_enable()
training_args.gradient_checkpointing = True

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
