# from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
# from peft import PeftModel, PeftConfig
# from datasets import load_dataset
# import torch
# import math
# import os

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# MODEL_NAME = "meta-llama/Llama-2-13b-hf" 
# ADAPTER_PATH = "./checkpoints/final_model"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     trust_remote_code=True,
# )

# model = PeftModel.from_pretrained(model, ADAPTER_PATH)

# model = model.merge_and_unload() 


# DATASET_PATH = "/scratch/zl3057/processed_txt"
# dataset = load_dataset("text", data_files={"test": f"{DATASET_PATH}/test/*.txt"})
# test_dataset = dataset["test"]

# def tokenize_function(examples):
#     tokenized = tokenizer(
#         examples["text"],
#         truncation=True,
#         padding="max_length",
#         max_length=16,
#     )
#     tokenized["labels"] = tokenized["input_ids"].copy()
#     return tokenized

# tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
# tokenized_test_dataset = tokenized_test_dataset.remove_columns(["text"])

# eval_args = TrainingArguments(
#     output_dir="./eval_results",
#     per_device_eval_batch_size=1,
#     do_eval=True,
#     report_to="none",
    
# )

# from transformers import Trainer

# class SafeTrainer(Trainer):
#     def _move_model_to_device(self, model, device):
#         return model

# trainer = SafeTrainer(
#     model=model,
#     args=eval_args,
#     eval_dataset=tokenized_test_dataset,
# )

# with torch.no_grad():
#     metrics = trainer.evaluate()

# metrics["perplexity"] = math.exp(metrics["eval_loss"])
# print("Evaluation Results:")
# for k, v in metrics.items():
#     print(f"{k}: {v:.4f}")


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from transformers import BitsAndBytesConfig
import torch
import math
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "meta-llama/Llama-2-13b-hf"
ADAPTER_PATH = "./checkpoints/final_model"
DATASET_PATH = "/scratch/zl3057/processed_txt"

# Quantized model config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load base model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load and merge LoRA
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model = model.merge_and_unload()
model.eval()

# Load dataset
dataset = load_dataset("text", data_files={"test": f"{DATASET_PATH}/test/*.txt"})
test_dataset = dataset["test"].select(range(100))  # use 100 samples for quick eval

def tokenize(example):
    input_ids = tokenizer(example["text"], truncation=True, padding="max_length", max_length=16, return_tensors="pt").input_ids
    return {"input_ids": input_ids, "labels": input_ids.clone()}

test_dataset = test_dataset.map(tokenize)
total_loss = 0.0
count = 0

for row in test_dataset:
    input_ids = row["input_ids"].to(model.device)
    labels = row["labels"].to(model.device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        count += 1

avg_loss = total_loss / count
perplexity = math.exp(avg_loss)

print(f"Evaluation Results:\nLoss: {avg_loss:.4f}\nPerplexity: {perplexity:.4f}")
