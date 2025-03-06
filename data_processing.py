import os
from PyPDF2 import PdfReader
import random

input_dir = "/scratch/zl3057/climate_text_dataset/"  
output_dir = "/scratch/zl3057/processed_txt/"

os.makedirs(output_dir, exist_ok=True)

file_list = os.listdir(input_dir)
random.shuffle(file_list)
# split file list
txt_list = os.listdir(output_dir)

train_list = file_list[:int(len(file_list) * 0.9)]
test_list = file_list[int(len(file_list) * 0.9):]

for file in train_list:
    if file.endswith(".pdf"):
        with open(input_dir + file, "rb") as f:
            if file.replace(".pdf", ".txt") in txt_list:
                continue
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            with open(output_dir + file.replace(".pdf", ".txt"), "w") as f:
                f.write(text)
            print(f"Processed {file}")

for file in test_list:
    if file.endswith(".pdf"):
        with open(input_dir + file, "rb") as f:
            if file.replace(".pdf", ".txt") in txt_list:
                continue
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            with open(output_dir + file.replace(".pdf", ".txt"), "w") as f:
                f.write(text)
            print(f"Processed {file}")


