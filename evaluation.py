from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from transformers import BitsAndBytesConfig
import torch
import math
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "meta-llama/Llama-2-13b-hf" 
ADAPTER_PATH = "./checkpoints/final_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.float16,
#     device_map=None,
#     trust_remote_code=True,
# )


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

DATASET_PATH = "/scratch/zl3057/processed_txt"
dataset = load_dataset("text", data_files={"test": f"{DATASET_PATH}/test/*.txt"})
test_dataset = dataset["test"]

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=64,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = tokenized_test_dataset.remove_columns(["text"])

eval_args = TrainingArguments(
    output_dir="./eval_results",
    per_device_eval_batch_size=16,
    do_eval=True,
    report_to="none",
    fp16=True,
    eval_accumulation_steps=1,
    
)

from transformers import Trainer

class SafeTrainer(Trainer):
    def _move_model_to_device(self, model, device):
        return model

trainer = SafeTrainer(
    model=model,
    args=eval_args,
    eval_dataset=tokenized_test_dataset,
)

with torch.no_grad():
    metrics = trainer.evaluate()

metrics["perplexity"] = math.exp(metrics["eval_loss"])
print("Evaluation Results:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
