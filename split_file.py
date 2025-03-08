import os
import json
import random

DATASET_DIR = "/scratch/zl3057/climate_text_dataset/"
SPLIT_FILE = "split_files.json"


random.seed(42)

files = os.listdir(DATASET_DIR)
random.shuffle(files)

split_idx = int(0.9 * len(files))
train_files = files[:split_idx]
test_files = files[split_idx:]

# Save the split to a JSON file
split_data = {"train": train_files, "test": test_files}
with open(SPLIT_FILE, "w") as f:
    json.dump(split_data, f, indent=4)

print(f"Precomputed split saved to {SPLIT_FILE}")
print(f"Training: {len(train_files)}, Testing: {len(test_files)}")