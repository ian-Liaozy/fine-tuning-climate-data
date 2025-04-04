import os
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = "./checkpoints/final_dist_model"
DATASET_PATH = "/scratch/zl3057/processed_txt"
BATCH_SIZE = 16
MAX_LENGTH = 42

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    load_in_4bit=True
)

# Load dataset
dataset = load_dataset("text", data_files={"test": f"{DATASET_PATH}/test/*.txt"})
test_dataset = dataset["test"]

# Tokenize
def tokenize_function(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Setup Accelerator
accelerator = Accelerator()
dataloader = DataLoader(tokenized_test_dataset, batch_size=BATCH_SIZE)
model, dataloader = accelerator.prepare(model, dataloader)

# Evaluation
model.eval()
losses = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Evaluating"):
        outputs = model(**batch)
        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(BATCH_SIZE)))

# Aggregate and compute perplexity
all_losses = torch.cat(losses)
eval_loss = all_losses.mean().item()
perplexity = math.exp(eval_loss)

# Output
print(f"Eval loss: {eval_loss:.4f}")
print(f"Perplexity: {perplexity:.4f}")

with open("eval_results.txt", "w") as f:
    f.write(f"Eval loss: {eval_loss:.4f}\n")
    f.write(f"Perplexity: {perplexity:.4f}\n")
