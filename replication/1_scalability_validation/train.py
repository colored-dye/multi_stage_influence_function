import os

from transformers import (
    GPTNeoXForCausalLM, AutoConfig, AutoTokenizer,
)

device = "cuda:3"
pretrained_model_path = "/data/home/Model/Pythia/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

config = AutoConfig.from_pretrained(pretrained_model_path)
config.num_hidden_layers = 2
config.hidden_size = 256
config.num_attention_heads = 2
config.intermediate_size = 256*4
config.vocab_size = tokenizer.vocab_size
config.save_pretrained(".")

model = GPTNeoXForCausalLM(config).to(device)
print(sum([p.numel() for p in model.parameters()]))
def count_mlp_params(model):
    n = 0
    for layer in model.gpt_neox.layers:
        n += sum([p.numel() for p in layer.mlp.dense_4h_to_h.parameters()])
    return n
print(count_mlp_params(model))

max_length = 512
import torch
tokenized_ds = torch.load("tokenized_ds.pt")
from torch.utils.data import DataLoader, TensorDataset

def collate_fn(batch):
    return torch.stack([x[0] for x in batch])

def get_dataloader(tokenized_split, batch_size, shuffle):
    ds = TensorDataset(tokenized_split['input_ids'])
    loader = DataLoader(ds, batch_size, shuffle, collate_fn=collate_fn)
    return loader

train_loader = get_dataloader(tokenized_ds['train'], 8, True)
next(iter(train_loader)).shape
lr = 1e-3
epochs = 10
batch_size = 64
weight_decay = 0.01

from tqdm import tqdm

import torch.nn.functional as F


def loss_fn(logits, target):
    logits = logits.reshape(-1, logits.size(-1))
    target = target.reshape(-1)
    loss = F.cross_entropy(logits, target)
    return loss


def test_epoch(model, data_loader, device):
    model.eval()
    
    total_loss = 0
    acc = 0
    with torch.no_grad():
        for input_ids in data_loader:
            input_ids = input_ids.to(device)
            output = model(input_ids).logits
            output = output[:, :-1]
            loss = loss_fn(output, input_ids[:, 1:])
            total_loss += loss.item()

            preds = torch.argmax(output.reshape(-1, output.size(-1)), dim=-1)
            n_correct = torch.count_nonzero(preds==input_ids[:, 1:].reshape(-1), dim=-1).item()
            acc += n_correct
    avg_loss = total_loss / len(data_loader)
    return avg_loss, acc/len(data_loader.dataset)/max_length


def train_epoch(model, train_loader, scheduler, optimizer, device):
    total_loss = 0
    for input_ids in tqdm(train_loader):
        input_ids = input_ids.to(device)
        output = model(input_ids).logits
        loss = loss_fn(output[:, :-1], input_ids[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    scheduler.step()
    return avg_loss


def train_loop(model, device, train_loader, test_loader, scheduler, optimizer, num_epochs):
    train_loss = None
    test_loss = None
    test_acc = None
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, scheduler, optimizer, device)
        test_loss, test_acc = test_epoch(model, test_loader, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


# train_loader = get_dataloader({'input_ids': tokenized_ds['train']['input_ids'][:1024]}, batch_size, True)
train_loader = get_dataloader(tokenized_ds['train'], batch_size, True)
valid_loader = get_dataloader(tokenized_ds['valid'], batch_size, False)

warmup_steps = 4000

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, epochs*len(train_loader))
train_loop(model, device, train_loader, valid_loader, scheduler, optimizer, epochs)

model.cpu()
torch.save(model.state_dict(), "model.ckpt")
