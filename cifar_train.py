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
from vision_lstm.vision_minLSTM import VisionMinLSTMConcat
from vision_lstm.vision_minLSTM import VisionMinLSTM
from vision_lstm import VisionLSTM2



model = VisionMinLSTMConcat(
    dim=192,  # latent dimension (192 for ViL-T)
    depth=12,  # how many ViL blocks (1 block consists 2 subblocks of a forward and backward block)
    patch_size=4,  # patch_size (results in 64 patches for 32x32 images)
    input_shape=(3, 32, 32),  # RGB images with resolution 32x32
    output_shape=(10,),  # classifier with 10 classes
    #drop_path_rate=0.0,  # stochastic depth parameter (disabled for ViL-T)
).to(device)

# model = VisionMinLSTM(
#     dim=192,  # latent dimension (192 for ViL-T)
#     depth=12,  # how many ViL blocks (1 block consists 2 subblocks of a forward and backward block)
#     patch_size=4,  # patch_size (results in 64 patches for 32x32 images)
#     input_shape=(3, 32, 32),  # RGB images with resolution 32x32
#     output_shape=(10,),  # classifier with 10 classes
#     #drop_path_rate=0.0,  # stochastic depth parameter (disabled for ViL-T)
# ).to(device)

# model = VisionLSTM2(
#     dim=192,  # latent dimension (192 for ViL-T)
#     depth=12,  # how many ViL blocks (1 block consists 2 subblocks of a forward and backward block)
#     patch_size=4,  # patch_size (results in 64 patches for 32x32 images)
#     input_shape=(3, 32, 32),  # RGB images with resolution 32x32
#     output_shape=(10,),  # classifier with 10 classes
#     drop_path_rate=0.0,  # stochastic depth parameter (disabled for ViL-T)
# ).to(device)

print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print(model)
#model = torch.compile(model) #This makes training faster

# Stop execution for debugging
# sys.exit("Debug: Stopping after printing the model.")

print('-------Initializing optimizer----------')
# initialize optimizer and learning rate schedule (linear warmup for first 10% -> linear decay)
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


print('-------Training model----------')
# train model
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
for _ in range(epochs):
    # train for an epoch
    model.train()
    for x, y in train_dataloader:
        # prepare forward pass
        x = x.to(device)
        y = y.to(device)

        # schedule learning rate
        for param_group in optim.param_groups:
            param_group["lr"] = lrs[update]

        # forward pass (this tutorial doesnt use mixed precision because T4 cards dont support bfloat16)
        # we recommend bfloat16 mixed precision training
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)

        # backward pass
        loss.backward()

        # update step
        optim.step()
        optim.zero_grad()

        # status update
        train_accuracy = ((y_hat.argmax(dim=1) == y).sum() / y.numel()).item()
        update += 1
        pbar.update()
        pbar.set_description(
            f"train_loss: {loss.item():.4f} "
            f"train_accuracy: {train_accuracy * 100:4.1f}% "
            f"test_accuracy: {test_accuracy * 100:4.1f}%"
        )
        train_losses.append(loss.item())
        train_accuracies.append(train_accuracy)

    # evaluate
    num_correct = 0
    model.eval()
    for x, y in test_dataloader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_hat = model(x)
            num_correct += (y_hat.argmax(dim=1) == y).sum().item()
    test_accuracy = num_correct / len(test_dataset)
    test_accuracies.append(test_accuracy)
    pbar.set_description(
        f"train_loss: {loss.item():.4f} "
        f"train_accuracy: {train_accuracy * 100:4.1f}% "
        f"test_accuracy: {test_accuracy * 100:4.1f}%"
    )
pbar.close()