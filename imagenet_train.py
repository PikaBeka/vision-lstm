import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F
from torchvision.datasets import ImageNet
from torchvision.transforms import ToTensor
import sys
import torch.profiler
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torchinfo import summary
from ptflops import get_model_complexity_info
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.cuda.amp as amp

# Path to ImageNet
data_path = "./data_cifar"

#Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

print('-------Setting hyperparameters----------')
# hyperparameters
patch_size = 16
latent_dim = 192 # for T model
depth = 24
pooling = "bilateral concat"
stochastic_depth = 0 # for T model
batch_size = 2048 # for pretraining, change to 1024 for finetuning
epochs = 800 # for T, chnage to 20 for finetune

base_lr = 1e-3
end_lr = 1e-6
weight_decay = 0.05
beta1_momentum = 0.9
beta2_momentum = 0.999
lr_scaling_division = 1024
warmup_epochs = 5 # for T model
precision = "mixed_bfloat16"
grad_clip = 1.0

train_resolution = 192 # pretrain, 224 for finetune
interpolation = 'bicubic'
random_scale = (0.08, 1.0)
horizontal_flip_prob = 0.5
gaussian_blur_sigma = (0.1, 2.0)
color_jitter = (0.3, 0.3, 0.3, 0.0)
mixup_alpha = 0.8
cutmix_alpha = 1.0

# Define data augmentations for training
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(train_resolution, scale=random_scale, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
    transforms.ColorJitter(*color_jitter),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define transforms for testing (no augmentation)
val_transforms = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(192),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets with augmentations
train_dataset = ImageNet(root=data_path, split='train', transform=train_transforms)
val_dataset = ImageNet(root=data_path, split='val', transform=val_transforms)
print(f"train_dataset length: {len(train_dataset)}")
print(f"test_dataset length: {len(val_dataset)}")

# setup dataloaders
print('-------Creating dataloaders----------')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# create model
print('-------Creating model----------')
# from vision_lstm.vision_minLSTM import VisionMinLSTMConcat
from vision_lstm import VisionLSTM2

model = VisionLSTM2(
    dim=192,  # latent dimension (192 for ViL-T)
    depth=12,  # how many ViL blocks (1 block consists 2 subblocks of a forward and backward block)
    patch_size=patch_size,  # patch_size (results in 64 patches for 32x32 images)
    input_shape=(3, 192, 192),  # RGB images with resolution 32x32
    output_shape=(1000,),  # classifier with 10 classes
    drop_path_rate=stochastic_depth,  # stochastic depth parameter (disabled for ViL-T)
).to(device)

macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=False)
print(macs, params)

print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay, betas=(beta1_momentum, beta2_momentum))
lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=end_lr)
# Mixed Precision
scaler = amp.GradScaler(device)

total_updates = len(train_loader) * epochs
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
    # train for an epoch
    model.train()
    for x, y in train_loader:
        # prepare forward pass
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        with amp.autocast(device, enabled=True):
            y_hat = model(x)
            loss = F.cross_entropy(y_hat, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # status update
        train_accuracy = ((y_hat.argmax(dim=1) == y).sum() / y.numel()).item()
        update += 1
        pbar.update()
        pbar.set_description(
            f"train_loss: {loss.item():.4f} "
            f"train_accuracy: {train_accuracy * 100:4.1f}% "
            f"test_accuracy: {test_accuracy * 100:4.1f}%"
            f"epoch: {e}"
        )
        train_losses.append(loss.item())
        train_accuracies.append(train_accuracy)
    lr_scheduler.step()

# evaluate
num_correct = 0
model.eval()
for x, y in val_loader:
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        y_hat = model(x)
        num_correct += (y_hat.argmax(dim=1) == y).sum().item()

test_accuracy = num_correct / len(val_dataset)
test_accuracies.append(test_accuracy)
pbar.set_description(
    f"train_loss: {loss.item():.4f} "
    f"train_accuracy: {train_accuracy * 100:4.1f}% "
    f"test_accuracy: {test_accuracy * 100:4.1f}%"
)
pbar.close()