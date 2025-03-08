from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
# from bitsandbytes import quantize
from datasets import load_dataset

dataset_path = "/scratch/zl3057/processed_txt"

dataset = load_dataset("text", data_files={
    "train": f"{dataset_path}/train/*.txt",
    "test": f"{dataset_path}/test/*.txt"
})




model_name = "/scratch/zl3057/llama-3b-hf"  # Pretrained model path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

lora_config = LoraConfig(
    r=8,  # Rank of low-rank matrices
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

print("Model prepared with LoRA")

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Effectively increases batch size
    fp16=True,  # Mixed precision training
    save_steps=500,
    evaluation_strategy="steps",
    output_dir="./checkpoints/",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)

model.gradient_checkpointing_enable()
training_args.gradient_checkpointing = True

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()