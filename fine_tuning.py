from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from bitsandbytes import quantize


model_name = "/scratch/BDML25SP/LLaMA-3B"  # Pretrained model path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

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