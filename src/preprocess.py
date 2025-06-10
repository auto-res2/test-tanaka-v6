"""
Data preprocessing module for HFR-DT experiments.
Handles CIFAR-10 dataset loading and preprocessing for the experiments.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_dataloader(batch_size=32, train=True, download=True):
    """
    Get CIFAR-10 dataloader with appropriate transforms for the experiments.
    
    Args:
        batch_size (int): Batch size for the dataloader
        train (bool): Whether to load training or test set
        download (bool): Whether to download the dataset if not present
    
    Returns:
        DataLoader: Configured CIFAR-10 dataloader
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=train, download=download, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2)
    
    return dataloader

def preprocess_data():
    """
    Main preprocessing function that prepares the data for experiments.
    """
    print("Preprocessing CIFAR-10 data...")
    
    train_loader = get_cifar10_dataloader(batch_size=32, train=True, download=True)
    test_loader = get_cifar10_dataloader(batch_size=32, train=False, download=False)
    
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    return train_loader, test_loader

if __name__ == "__main__":
    preprocess_data()
