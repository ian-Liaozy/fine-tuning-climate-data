import os
from PyPDF2 import PdfReader
import random

input_dir = "/scratch/zl3057/climate_text_dataset/"  
train_dir = "/scratch/zl3057/processed_txt/train/"
test_dir = "/scratch/zl3057/processed_txt/test/"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

file_list = os.listdir(input_dir)
random.shuffle(file_list)
# split file list

train_list = file_list[:int(len(file_list) * 0.9)]
test_list = file_list[int(len(file_list) * 0.9):]

for file in train_list:
    if file.endswith(".pdf"):
        with open(input_dir + file, "rb") as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            with open(train_dir + file.replace(".pdf", ".txt"), "w") as f:
                f.write(text)
            print(f"Processed {file}")

for file in test_list:
    if file.endswith(".pdf"):
        with open(input_dir + file, "rb") as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            with open(test_dir + file.replace(".pdf", ".txt"), "w") as f:
                f.write(text)
            print(f"Processed {file}")


