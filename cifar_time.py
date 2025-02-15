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
import sys
import torch.profiler
# from fvcore.nn import FlopCountAnalysis, parameter_count_table
import time
from ptflops import get_model_complexity_info

#Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Define data augmentations for training
train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Randomly crop the image to 32x32 with padding
    transforms.RandomHorizontalFlip(),     # Randomly flip the image horizontally
    ToTensor(),                            # Convert the image to a tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Normalize with CIFAR-10 means and stds
])

# Define transforms for testing (no augmentation)
test_transforms = transforms.Compose([
    ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# Create datasets with augmentations
train_dataset = CIFAR10(root="./data_cifar", train=True, download=True, transform=train_transforms)
test_dataset = CIFAR10(root="./data_cifar", train=False, download=False, transform=test_transforms)
print(f"train_dataset length: {len(train_dataset)}")
print(f"test_dataset length: {len(test_dataset)}")

print('-------Setting hyperparameters----------')
# hyperparameters
batch_size = 256
epochs = 200
lr = 1e-3
weight_decay = 0.05


# setup dataloaders
print('-------Creating dataloaders----------')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


# create model
print('-------Creating model----------')
# from vision_lstm.vision_minLSTM import VisionMinLSTMConcat
from vision_lstm.vision_minlstm import VisionMinLSTM
from vision_lstm import VisionLSTM2

model = VisionMinLSTM(
    dim=192,
    input_shape=(3, 32, 32),
    depth=12,
    output_shape=(10,),
    pooling="bilateral_flatten",
    patch_size=4,
    drop_path_rate=0.0,
).to(device)

# model = VisionLSTM2(
#     dim=192,  # latent dimension (192 for ViL-T)
#     depth=12,  # how many ViL blocks (1 block consists 2 subblocks of a forward and backward block)
#     patch_size=4,  # patch_size (results in 64 patches for 32x32 images)
#     input_shape=(3, 32, 32),  # RGB images with resolution 32x32
#     output_shape=(10,),  # classifier with 10 classes
#     drop_path_rate=0.0,  # stochastic depth parameter (disabled for ViL-T)
# ).to(device)

optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
total_updates = len(train_dataloader) * epochs
warmup_updates = int(total_updates * 0.1)
lrs = torch.concat(
    [
        # linear warmup
        torch.linspace(0, optim.defaults["lr"], warmup_updates),
        # linear decay
        torch.linspace(optim.defaults["lr"], 0, total_updates - warmup_updates),
    ],
)

total_time = 0

model.eval()
for x, y in test_dataloader:
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        start = time.time()
        y_hat = model(x)
        end = time.time()
        total_time += end - start

print(total_time)