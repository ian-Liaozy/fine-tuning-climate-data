from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import PeftModel, PeftConfig
from datasets import load_dataset
import torch
import math
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Load Tokenizer ===
MODEL_NAME = "meta-llama/Llama-2-13b-hf"  # Base model
ADAPTER_PATH = "./checkpoints/final_model"  # Contains only LoRA weights

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# === Load base model and merge LoRA ===
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model = model.merge_and_unload()  # 💥 merge LoRA into full model

# === Prepare dataset ===
DATASET_PATH = "/scratch/zl3057/processed_txt"
dataset = load_dataset("text", data_files={"test": f"{DATASET_PATH}/test/*.txt"})
test_dataset = dataset["test"]

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=32,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = tokenized_test_dataset.remove_columns(["text"])

# === Training arguments ===
eval_args = TrainingArguments(
    output_dir="./eval_results",
    per_device_eval_batch_size=1,
    do_eval=True,
    report_to="none",
)

# === Trainer wrapper to prevent `.to()` ===
from transformers import Trainer

class SafeTrainer(Trainer):
    def _move_model_to_device(self, model, device):
        return model

trainer = SafeTrainer(
    model=model,
    args=eval_args,
    eval_dataset=tokenized_test_dataset,
)

# === Evaluate ===
with torch.no_grad():
    metrics = trainer.evaluate()

metrics["perplexity"] = math.exp(metrics["eval_loss"])
print("Evaluation Results:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
