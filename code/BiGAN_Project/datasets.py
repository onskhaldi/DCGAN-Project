import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------
# Individuelle Transform je nach Datensatz
# -------------------------------------------
def get_transforms(dataset_name):
    if dataset_name == 'mnist':
        return transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif dataset_name == 'cifar10':
        return transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif dataset_name == 'celeba':
        return transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# -------------------------------------------
# Datensatzladefunktionen
# -------------------------------------------
def load_mnist(root='data', train=True, batch_size=128, num_workers=2):
    transform = get_transforms('mnist')
    dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

def load_cifar10(root='data', train=True, batch_size=128, num_workers=2):
    transform = get_transforms('cifar10')
    dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

def load_celeba(root='data', train=True, batch_size=128, num_workers=2):
    transform = get_transforms('celeba')
    split = 'train' if train else 'valid'
    dataset = datasets.CelebA(root=root, split=split, download=True, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  
    )
# -------------------------------------------
# Unified Interface – Rückgabe auch image_size
# -------------------------------------------
def create_dataset(dataset_name, root='data', batch_size=128, num_workers=2):
    name = dataset_name.lower()
    if name == 'mnist':
        nc = 1
        image_size = 32
        train_loader = load_mnist(root, True, batch_size, num_workers)
        test_loader = load_mnist(root, False, batch_size, num_workers)
    elif name == 'cifar10':
        nc = 3
        image_size = 32
        train_loader = load_cifar10(root, True, batch_size, num_workers)
        test_loader = load_cifar10(root, False, batch_size, num_workers)
    elif name == 'celeba':
        nc = 3
        image_size = 64
        train_loader = load_celeba(root, True, batch_size, num_workers)
        test_loader = load_celeba(root, False, batch_size, num_workers)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_loader, test_loader, nc, image_size

# -------------------------------------------
# Visualisierung
# -------------------------------------------
def visualize_dataset(dataloader, n_images=25):
    images, _ = next(iter(dataloader))
    images = images[:n_images]
    images = images * 0.5 + 0.5  # zurückskalieren auf [0, 1]
    grid = utils.make_grid(images, nrow=5, padding=2)
    np_img = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(np_img)
    plt.axis('off')
    plt.title("5x5 Beispielbilder")
    plt.show()

    