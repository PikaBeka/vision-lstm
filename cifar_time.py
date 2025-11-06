from vision_lstm import VisionLSTM2
from vision_lstm.vision_minlstm import VisionMinLSTM
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

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Define data augmentations for training
train_transforms = transforms.Compose([
    # Randomly crop the image to 32x32 with padding
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),     # Randomly flip the image horizontally
    ToTensor(),                            # Convert the image to a tensor
    # Normalize with CIFAR-10 means and stds
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# Define transforms for testing (no augmentation)
test_transforms = transforms.Compose([
    ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# Create datasets with augmentations
train_dataset = CIFAR10(root="./data_cifar", train=True,
                        download=True, transform=train_transforms)
test_dataset = CIFAR10(root="./data_cifar", train=False,
                       download=False, transform=test_transforms)
print(f"train_dataset length: {len(train_dataset)}")
print(f"test_dataset length: {len(test_dataset)}")

print('-------Setting hyperparameters----------')
# hyperparameters
batch_size = 256
epochs = 200
lr = 1.0e-5
weight_decay = 0.05


# setup dataloaders
print('-------Creating dataloaders----------')
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


# create model
print('-------Creating model----------')
# from vision_lstm.vision_minLSTM import VisionMinLSTMConcat

model = VisionMinLSTM(
    dim=192,
    input_shape=(3, 32, 32),
    depth=12,
    output_shape=(10,),
    pooling="bilateral_flatten",
    patch_size=4,
    drop_path_rate=0.0,
).to(device)

print(model)

print("MODEL LOADED")

# model = VisionLSTM2(
#     dim=192,  # latent dimension (192 for ViL-T)
#     depth=12,  # how many ViL blocks (1 block consists 2 subblocks of a forward and backward block)
#     patch_size=4,  # patch_size (results in 64 patches for 32x32 images)
#     input_shape=(3, 32, 32),  # RGB images with resolution 32x32
#     output_shape=(10,),  # classifier with 10 classes
#     drop_path_rate=0.0,  # stochastic depth parameter (disabled for ViL-T)
# ).to(device)

total_time = 0
batch_size = 16
test_input = torch.randn(batch_size, 3, 224, 224).to(device)

model.eval()
for i in range(10000):
    with torch.no_grad():
        start = time.time()
        y_hat = model(test_input)
        end = time.time()
        total_time += end - start

print(total_time)
