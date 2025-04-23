import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.models import resnet18, ResNet50_Weights
import sys
import torch.profiler
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torchinfo import summary
from ptflops import get_model_complexity_info
import time
import random
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, classification_report

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')


train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  
    transforms.RandomHorizontalFlip(),     
    ToTensor(),                           
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  
])

test_transforms = transforms.Compose([
    ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

train_dataset = CIFAR10(root="./data_cifar", train=True, download=True, transform=train_transforms)
test_dataset = CIFAR10(root="./data_cifar", train=False, download=True, transform=test_transforms)
print(f"train_dataset length: {len(train_dataset)}")
print(f"test_dataset length: {len(test_dataset)}")

print('-------Setting hyperparameters----------')
batch_size = 256
epochs = 200
lr = 1e-3
weight_decay = 0.05
checkpoint_frequency = 100 

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

print('-------Creating dataloaders----------')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

print('-------Creating ResNet-18 model----------')
model = resnet18(weights=None)  
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10)

model = model.to(device)

macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=False)
print(f"Computational complexity: {macs}")
print(f"Number of parameters: {params}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

print('-------Initializing optimizer----------')
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
total_updates = len(train_dataloader) * epochs
warmup_updates = int(total_updates * 0.1)
lrs = torch.concat(
    [
        torch.linspace(0, optimizer.defaults["lr"], warmup_updates),
        torch.linspace(optimizer.defaults["lr"], 0, total_updates - warmup_updates),
    ],
)

def save_checkpoint(epoch, model, optimizer, loss, train_accuracy, test_accuracy, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': loss.item() if loss is not None else 0,
        'train_accuracy': train_accuracy if train_accuracy is not None else 0,
        'test_accuracy': test_accuracy if test_accuracy is not None else 0
    }, filename)
    print(f"Checkpoint saved at {filename}")

print('-------Training model----------')
update = 0
pbar = tqdm(total=total_updates)
pbar.update(0)
pbar.set_description("train_loss: ????? train_accuracy: ????% test_accuracy: ????%")
test_accuracy = 0.0
train_losses = []
train_accuracies = []
test_accuracies = []
loss = None
train_accuracy = None

for e in range(epochs):
    model.train()
    for x, y in train_dataloader:
        x = x.to(device)
        y = y.to(device)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lrs[update]

        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        train_accuracy = ((y_hat.argmax(dim=1) == y).sum() / y.numel()).item()
        update += 1
        pbar.update()
        pbar.set_description(
            f"train_loss: {loss.item():.4f} "
            f"train_accuracy: {train_accuracy * 100:4.1f}% "
            f"test_accuracy: {test_accuracy * 100:4.1f}% "
            f"epoch: {e+1}/{epochs}"
        )
        train_losses.append(loss.item())
        train_accuracies.append(train_accuracy)
    
    num_correct = 0
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    test_accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)
    test_accuracies.append(test_accuracy)
    
    pbar.set_description(
        f"train_loss: {loss.item():.4f} "
        f"train_accuracy: {train_accuracy * 100:4.1f}% "
        f"test_accuracy: {test_accuracy * 100:4.1f}% "
        f"epoch: {e+1}/{epochs}"
    )
    
    if (e + 1) % checkpoint_frequency == 0 or (e + 1) == epochs:
        checkpoint_path = os.path.join(checkpoint_dir, f"resnet50_cifar10_epoch_{e+1}.pt")
        save_checkpoint(
            epoch=e+1,
            model=model,
            optimizer=optimizer,
            loss=loss,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            filename=checkpoint_path
        )

num_correct = 0
model.eval()
eval_start = time.time()
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

eval_total = time.time() - eval_start
throughput = len(test_dataset) / eval_total
print(f"Inference time: {eval_total:.2f}s — {throughput:.1f} images/sec")

test_accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)

pbar.set_description(
    f"train_loss: {loss.item():.4f} "
    f"train_accuracy: {train_accuracy * 100:4.1f}% "
    f"test_accuracy: {test_accuracy * 100:4.1f}%"
)
pbar.close()

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
print(f"\nPrecision: {precision:.4f}  Recall: {recall:.4f}  F1‑score: {f1:.4f}")
print("\nClassification report:\n", classification_report(y_true, y_pred, digits=4))

final_checkpoint_path = os.path.join(checkpoint_dir, f"resnet50_cifar10_final.pt")
save_checkpoint(
    epoch=epochs,
    model=model,
    optimizer=optimizer,
    loss=loss,
    train_accuracy=train_accuracy,
    test_accuracy=test_accuracy,
    filename=final_checkpoint_path
)