import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Your model class (adjust accordingly)
from vision_lstm.vision_minlstm import VisionMinLSTM  # replace with your actual model import


def main():
    # Load checkpoint
    checkpoint_path = './minlstm_conv.pt'  # replace with your actual checkpoint path
    model = VisionMinLSTM(
        dim=192,
        input_shape=(3, 32, 32),
        depth=12,
        output_shape=(10,),
        pooling="bilateral_flatten",
        patch_size=4,
        drop_path_rate=0.0,
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Normalize using CIFAR-10 mean/std
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))
    ])

    # Load test set
    testset = torchvision.datasets.CIFAR10(root='./data_cifar', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True, num_workers=2)

    # Class labels
    classes = testset.classes

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # Run through model
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # Denormalize function
    def denormalize(img):
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        return img

    # Plot
    plt.figure(figsize=(12, 5))  # Wider height for 2 rows
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(denormalize(images[i].cpu()))
        plt.title(f"T: {classes[labels[i]]}\nP: {classes[predicted[i]]}", fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # This is where you would typically call the function or run the script
    from multiprocessing import freeze_support
    freeze_support()
    main()