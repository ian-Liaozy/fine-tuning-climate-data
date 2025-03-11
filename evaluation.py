from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import os
import math

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = "./checkpoints/final_model"  # Load the latest trained model
DATASET_PATH = "/scratch/zl3057/processed_txt"

dataset = load_dataset("text", data_files={"test": f"{DATASET_PATH}/test/*.txt"})
test_dataset = dataset["test"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cuda")

def tokenize_function(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=42,
        return_tensors="pt",
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    return tokenized_inputs

tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, num_proc=1)

eval_args = TrainingArguments(
    output_dir="./eval_results",
    per_device_eval_batch_size=16,  
    dataloader_num_workers=8,
    do_eval=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=tokenized_test_dataset,
)

metrics = trainer.evaluate()

perplexity = math.exp(metrics["eval_loss"])
metrics["perplexity"] = perplexity

print("Evaluation Results:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

with open("eval_results.txt", "w") as f:
    for key, value in metrics.items():
        f.write(f"{key}: {value:.4f}\n")
