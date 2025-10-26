from vision_lstm import VisionLSTM2
from vision_lstm.vision_minlstm import VisionMinLSTM
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
import sys
import torch.profiler
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torchinfo import summary
from ptflops import get_model_complexity_info
import random
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report
import time
import matplotlib.pyplot as plt
import os
from datetime import datetime
import wandb


SEED = 52
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MODEL_NAME = "VisionLSTM2_not_ours"

# Initialize wandb
wandb.init(
    project="vision-lstm2-cifar100",
    name=f"{MODEL_NAME}_dim192_depth12",
    config={
        "model": MODEL_NAME,
        "dataset": "CIFAR-100",
        "epochs": 200,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "weight_decay": 0.05,
        "seed": SEED,
        "dim": 192,
        "depth": 12,
        "patch_size": 4,
    }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3))
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def rand_bbox(size, lam):
    """
    Generate random bbox coordinates for CutMix.
    size: tuple of (batch_size, channels, height, width)
    lam: lambda value (mixing ratio)
    """
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    """
    Applies the CutMix augmentation to a batch of images (x) and labels (y).
    Returns: 
      - Mixed images,
      - Original labels (target_a),
      - Permuted labels (target_b),
      - Adjusted lambda value.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    rand_index = torch.randperm(batch_size).to(x.device)
    target_a = y
    target_b = y[rand_index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[rand_index, :, bby1:bby2, bbx1:bbx2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / float(x.size(-1) * x.size(-2)))
    return x, target_a, target_b, lam


train_dataset = CIFAR100(root="./data_cifar", train=True,
                         download=True, transform=train_transforms)
test_dataset = CIFAR100(root="./data_cifar", train=False,
                        download=True, transform=test_transforms)
print(f"train_dataset length: {len(train_dataset)}")
print(f"test_dataset length: {len(test_dataset)}")


print('-------Setting hyperparameters----------')
batch_size = 256
epochs = 300
lr = 5e-4
weight_decay = 0.05
cutmix_prob = 0.5

print('-------Creating dataloaders----------')
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


print('-------Creating model----------')
# from vision_lstm.vision_minLSTM import VisionMinLSTMConcat

model = VisionMinLSTM(
    dim=192,
    input_shape=(3, 32, 32),
    depth=12,
    output_shape=(100,),
    pooling="bilateral_flatten",
    patch_size=4,
    drop_path_rate=0.1,
).to(device)

# model = VisionLSTM2(
#     dim=192,
#     depth=12,
#     patch_size=4,
#     input_shape=(3, 32, 32),
#     output_shape=(100,),
#     drop_path_rate=0.1,
# ).to(device)

wandb.watch(model, log_freq=100)

# summary(model, input_size=(1, 3, 32, 32), depth=6)
macs, params = get_model_complexity_info(
    model, (3, 32, 32), as_strings=True, print_per_layer_stat=False)
print(macs, params)

# Log model complexity metrics to wandb
wandb.log({
    "model_parameters_M": sum(p.numel() for p in model.parameters()) / 1e6,
    "model_macs": macs,
})

print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
# model = torch.compile(model)  # This makes training faster

# Stop execution for debugging
# sys.exit("Debug: Stopping after printing the model.")

print('-------Initializing optimizer----------')
# initialize optimizer and learning rate schedule (linear warmup for first 10% -> linear decay)
optim = torch.optim.AdamW(model.parameters(
), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
total_updates = len(train_dataloader) * epochs
warmup_updates = int(total_updates * 0.1)
lrs = torch.concat(
    [
        # linear warmup
        torch.linspace(0, optim.defaults["lr"], warmup_updates),
        # linear decay
        torch.linspace(optim.defaults["lr"], 0,
                       total_updates - warmup_updates),
    ],
)

plot_dir = f"training_plots/{MODEL_NAME}"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

print('-------Training model----------')
# train model
update = 0
pbar = tqdm(total=total_updates)
pbar.update(0)
pbar.set_description(
    "train_loss: ????? train_accuracy: ????% test_accuracy: ????%")
test_accuracy = 0.0
train_losses = []
train_accuracies = []
test_accuracies = []
loss = None
train_accuracy = None
train_start = time.time()

# Add epoch-wise metrics tracking
epoch_train_losses = []
epoch_train_accs = []
epoch_test_accs = []

for e in range(epochs):
    # Reset epoch metrics
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0

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

        if np.random.rand() < cutmix_prob:
            x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)
            outputs = model(x)
            # Compute the loss as a weighted combination of the two targets
            loss = lam * F.cross_entropy(outputs, y_a, label_smoothing=0.1) + (
                1 - lam) * F.cross_entropy(outputs, y_b, label_smoothing=0.1)
        else:
            outputs = model(x)
            loss = F.cross_entropy(outputs, y)

        # backward pass
        loss.backward()

        # update step
        optim.step()
        optim.zero_grad()

        # status update
        train_accuracy = (
            (outputs.argmax(dim=1) == y).sum() / y.numel()).item()

        # Accumulate epoch metrics
        epoch_loss += loss.item() * y.size(0)
        epoch_correct += (outputs.argmax(dim=1) == y).sum().item()
        epoch_total += y.size(0)

        # Log iteration metrics to wandb
        wandb.log({
            "iteration": update,
            "iter_loss": loss.item(),
            "iter_accuracy": train_accuracy,
            "learning_rate": param_group["lr"]
        })

        update += 1
        pbar.update()
        pbar.set_description(
            f"train_loss: {loss.item():.4f} "
            f"train_accuracy: {train_accuracy * 100:4.1f}% "
            f"test_accuracy: {test_accuracy * 100:4.1f}% "
            f"epoch: {e}"
        )
        train_losses.append(loss.item())
        train_accuracies.append(train_accuracy)

    # Compute epoch-level metrics
    epoch_train_loss = epoch_loss / epoch_total
    epoch_train_acc = epoch_correct / epoch_total
    epoch_train_losses.append(epoch_train_loss)
    epoch_train_accs.append(epoch_train_acc)

    # Evaluate the model after each epoch
    if (e + 1) % 5 == 0 or e == 0 or e == epochs - 1:  # Evaluate every 5 epochs to save time
        # evaluate + precision/recall/F1
        y_true, y_pred = [], []
        model.eval()
        epoch_test_correct = 0
        epoch_test_total = 0

        with torch.no_grad():
            for x, y in test_dataloader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)

                epoch_test_correct += (preds == y).sum().item()
                epoch_test_total += y.size(0)

                y_true.extend(y.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        epoch_test_acc = epoch_test_correct / epoch_test_total
        epoch_test_accs.append(epoch_test_acc)
        test_accuracy = epoch_test_acc

        # Log epoch-level metrics to wandb
        wandb.log({
            "epoch": e,
            "epoch_train_loss": epoch_train_loss,
            "epoch_train_acc": epoch_train_acc,
            "epoch_test_acc": epoch_test_acc,
        })

        pbar.set_description(
            f"train_loss: {loss.item():.4f} "
            f"train_accuracy: {train_accuracy * 100:4.1f}% "
            f"test_accuracy: {test_accuracy * 100:4.1f}% "
            f"epoch: {e}"
        )

        if (e + 1) % 10 == 0 or e == epochs - 1:
            plt.figure(figsize=(15, 10))

            plt.subplot(2, 2, 1)
            plt.plot(range(len(epoch_train_losses)), epoch_train_losses)
            plt.title(f'{MODEL_NAME} - Training Loss per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)

            plt.subplot(2, 2, 2)
            epochs_completed = range(len(epoch_train_accs))
            plt.plot(epochs_completed, [
                     acc * 100 for acc in epoch_train_accs], label='Train Accuracy')
            plt.plot(range(len(epoch_test_accs)), [
                     acc * 100 for acc in epoch_test_accs], label='Test Accuracy')
            plt.title(f'{MODEL_NAME} - Accuracy per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True)
            plt.ylim(0, 100)

            plt.subplot(2, 2, 3)
            plt.plot(range(len(lrs[:update])), lrs[:update])
            plt.title('Learning Rate Schedule')
            plt.xlabel('Iteration')
            plt.ylabel('Learning Rate')
            plt.grid(True)

            plt.suptitle(
                f'{MODEL_NAME} Training on CIFAR-100 - Epoch {e+1}/{epochs}', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.97])

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(
                f"{plot_dir}/training_curves_epoch{e+1}_{timestamp}.png", dpi=300)

            wandb.log({"training_curve": wandb.Image(plt)})

            plt.close()

train_total = time.time() - train_start
print(
    f"Total training time: {train_total:.2f}s ({train_total/epochs:.2f}s/epoch)")

wandb.log({
    "total_training_time": train_total,
    "time_per_epoch": train_total/epochs
})

print("Performing final evaluation...")
eval_start = time.time()

y_true, y_pred = [], []
model.eval()

with torch.no_grad():
    for x, y in test_dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

# end of inference time calculation
eval_total = time.time() - eval_start
throughput = len(test_dataset) / eval_total
print(f"Inference time: {eval_total:.2f}s — {throughput:.1f} images/sec")

test_accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)
test_accuracies.append(test_accuracy)

# Log test performance metrics to wandb
wandb.log({
    "final_test_accuracy": test_accuracy,
    "inference_time": eval_total,
    "throughput": throughput
})

pbar.set_description(
    f"train_loss: {loss.item():.4f} "
    f"train_accuracy: {train_accuracy * 100:4.1f}% "
    f"test_accuracy: {test_accuracy * 100:4.1f}%"
)
pbar.close()

# Compute metrics
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro")
print(
    f"\nPrecision: {precision:.4f}  Recall: {recall:.4f}  F1‑score: {f1:.4f}")
print("\nClassification report:\n",
      classification_report(y_true, y_pred, digits=4))

# Log final metrics to wandb
wandb.log({
    "final_precision": precision,
    "final_recall": recall,
    "final_f1": f1,
})

# Create and save final training curve
plt.figure(figsize=(15, 10))

# Plot training loss
plt.subplot(2, 2, 1)
plt.plot(range(len(epoch_train_losses)), epoch_train_losses)
plt.title(f'{MODEL_NAME} - Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Plot training/test accuracy
plt.subplot(2, 2, 2)
plt.plot(range(len(epoch_train_accs)), [
         acc * 100 for acc in epoch_train_accs], label='Train Accuracy')
plt.plot(range(len(epoch_test_accs)), [
         acc * 100 for acc in epoch_test_accs], label='Test Accuracy')
plt.title(f'{MODEL_NAME} - Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.ylim(0, 100)

# Plot learning rate
plt.subplot(2, 2, 3)
plt.plot(range(len(lrs)), lrs)
plt.title('Learning Rate Schedule')
plt.xlabel('Iteration')
plt.ylabel('Learning Rate')
plt.grid(True)


# Add overall title and adjust layout
plt.suptitle(f'{MODEL_NAME} Final Training Results on CIFAR-100', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save final figure
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
final_plot_path = f"{plot_dir}/final_training_curves_{timestamp}.png"
plt.savefig(final_plot_path, dpi=300)
print(f"Final training curves saved to {final_plot_path}")

# Log final figure to wandb
wandb.log({"final_training_curve": wandb.Image(plt)})

# Save model
model_save_path = f"{plot_dir}/model_{MODEL_NAME}_{timestamp}.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optim.state_dict(),
    'epoch': epochs,
    'test_accuracy': test_accuracy,
}, model_save_path)
print(f"Model saved to {model_save_path}")

wandb.save(model_save_path)

wandb.finish()

print(f"Training of {MODEL_NAME} completed!")
