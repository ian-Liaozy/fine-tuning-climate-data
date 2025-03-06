import torch
import math
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

model.eval()

test_text = "Climate change is affecting global temperatures."
inputs = tokenizer(test_text, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss

perplexity = math.exp(loss.item())
print(f"Perplexity: {perplexity}")
