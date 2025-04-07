import os
import math
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time
import random

MODEL_PATH = "/scratch/zl3057/llama-3b-hf"
DATA_DIR = "/scratch/zl3057/processed_txt"
OUTPUT_DIR = "./checkpoints/final_dist_model_tensor_parallel"

class AllGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, world_size):
        ctx.world_size = world_size
        ctx.input_shape = input_tensor.shape
        
        output = [torch.zeros_like(input_tensor) for _ in range(world_size)]
        dist.all_gather(output, input_tensor)
        
        return torch.cat(output, dim=-1)
    
    @staticmethod
    def backward(ctx, grad_output):
        world_size = ctx.world_size
        input_shape = ctx.input_shape
        
        dim_size = grad_output.size(-1) // world_size
        
        rank = dist.get_rank()
        grad_slice = grad_output.narrow(-1, rank * dim_size, dim_size)
        
        return grad_slice, None

def all_gather_with_grad(tensor, world_size=None):
    if world_size is None:
        world_size = dist.get_world_size()
        
    return AllGatherFunction.apply(tensor, world_size)

class TPLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        self.out_features_per_gpu = out_features // self.world_size
        
        self.linear = nn.Linear(in_features, self.out_features_per_gpu, bias=bias)
    
    def forward(self, x):
        local_output = self.linear(x)
        
        return all_gather_with_grad(local_output)

class TPLlamaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(32000, 768)
        
        self.layers = nn.ModuleList([
            TPLinear(768, 768) for _ in range(6)
        ])
        
        self.lm_head = TPLinear(768, 32000)
    
    def forward(self, input_ids, labels=None):
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = nn.functional.gelu(layer(x))
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 32000), labels.view(-1))
        
        return type('Output', (), {'loss': loss, 'logits': logits})()

def extract_text_from_files(data_dir, train_split=0.9, max_files=None):
    # all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]
    train_files = [os.path.join(data_dir+"/train", f) for f in os.listdir(data_dir+"/train") if f.endswith('.txt')]
    test_files = [os.path.join(data_dir+"/test", f) for f in os.listdir(data_dir+"/test") if f.endswith('.txt')]
    # random.shuffle(all_files)
    # split_idx = int(len(all_files) * train_split)
    # train_files = all_files[:split_idx]
    # test_files = all_files[split_idx:]
    
    train_texts = []
    for file_path in train_files:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            train_texts.append(f.read())
    
    test_texts = []
    for file_path in test_files:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            test_texts.append(f.read())
    
    return train_texts, test_texts

def tokenize_text(text, max_length=128):
    tokens = [ord(c) % 32000 for c in text[:max_length]]
    return {
        'input_ids': torch.tensor(tokens, dtype=torch.long),
        'labels': torch.tensor(tokens, dtype=torch.long)
    }

class TextDataset(Dataset):
    def __init__(self, texts, max_length=128):
        self.texts = texts
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        return tokenize_text(self.texts[idx], self.max_length)

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device("cuda", rank)
    
    if rank == 0:
        print(f"Using {world_size} GPUs for tensor parallel training")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model = TPLlamaModel().to(device)
    
    if rank == 0:
        print(f"Extracting text from directory {DATA_DIR}...")
    train_texts, test_texts = extract_text_from_files(DATA_DIR)
    
    train_dataset = TextDataset(train_texts)
    test_dataset = TextDataset(test_texts)
    
    if rank == 0:
        print(f"Training set size: {len(train_dataset)} samples")
        print(f"Test set size: {len(test_dataset)} samples")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4,
        shuffle=True
    )
    
    if rank == 0:
        print(f"Data loader created, batch count: {len(train_loader)}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    
    print(f"Rank {rank}: Starting training")
    start_time = time.time()
    
    for epoch in range(20):
        model.train()
        
        epoch_start_time = time.time()
        
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0 and rank == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
        
        epoch_end_time = time.time()
        print(f"Rank {rank}: Epoch {epoch} completed in {epoch_end_time - epoch_start_time:.2f} seconds")
    
    total_training_time = time.time() - start_time
    
    if rank == 0:
        print(f"Training completed, total time: {total_training_time:.2f} seconds")
        
        with open(os.path.join(OUTPUT_DIR, 'tensor_parallel_results.txt'), 'w') as f:
            f.write(f"Training time: {total_training_time:.2f} seconds\n")
            f.write(f"Model: Tensor Parallel LlamaModel\n")
            f.write(f"Tensor parallel training, GPU count: {world_size}\n")
    
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()