"""
Data Loading Utilities for MNIST Dataset
Implemented in PyTorch
"""

import numpy as np
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

class MNISTDataLoader:
    """Universal data loader for MNIST dataset in PyTorch"""
    
    def __init__(self):
        """Initialize data loader"""
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        
    def load_data(self, validation_split=0.1):
        """
        Load and preprocess MNIST dataset for PyTorch
        
        Args:
            validation_split: Fraction of training data to use for validation
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load MNIST dataset
        train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
        
        # Split training data into train and validation
        if validation_split > 0:
            val_size = int(len(train_dataset) * validation_split)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(
                train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
        else:
            val_dataset = None
        
        print(f"Training samples: {len(train_dataset)}")
        if val_dataset is not None:
            print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_data_loaders(self, train_data, val_data, test_data, batch_size=128):
        """
        Create data loaders for PyTorch
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            test_data: Test dataset
            batch_size: Batch size for data loaders
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        ) if val_data is not None else None
        
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader


def get_sample_images(dataset, num_samples=10):
    """
    Get sample images from dataset
    
    Args:
        dataset: PyTorch dataset
        num_samples: Number of samples to retrieve
        
    Returns:
        Tuple of (sample_images, sample_labels)
    """
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    sample_images = []
    sample_labels = []
    for idx in indices:
        img, label = dataset[idx]
        sample_images.append(img)
        sample_labels.append(label)
    return torch.stack(sample_images), torch.tensor(sample_labels)


def analyze_dataset_statistics(train_dataset, test_dataset):
    """
    Analyze and print dataset statistics
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
    """
    print("=" * 60)
    print("MNIST Dataset Statistics (PyTorch)")
    print("=" * 60)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Get labels from PyTorch dataset
    # Note: This might be slow for large datasets if not handles carefully
    print("\nCalculating class distribution...")
    
    def get_labels(dataset):
        if isinstance(dataset, torch.utils.data.dataset.Subset):
            return [dataset.dataset.targets[i].item() for i in dataset.indices]
        return dataset.targets.tolist()

    train_labels = get_labels(train_dataset)
    test_labels = get_labels(test_dataset)
    
    train_class_counts = np.array(train_labels)
    test_class_counts = np.array(test_labels)
    
    print("\nClass distribution (Training):")
    for i in range(10):
        count = np.sum(train_class_counts == i)
        print(f"  Digit {i}: {count} ({100 * count / len(train_class_counts):.2f}%)")
    
    print("\nClass distribution (Test):")
    for i in range(10):
        count = np.sum(test_class_counts == i)
        print(f"  Digit {i}: {count} ({100 * count / len(test_class_counts):.2f}%)")
    
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing PyTorch Data Loading")
    print("=" * 60)
    pt_loader = MNISTDataLoader()
    train_data, val_data, test_data = pt_loader.load_data()
    analyze_dataset_statistics(train_data, test_data)
    
    train_loader, val_loader, test_loader = pt_loader.create_data_loaders(
        train_data, val_data, test_data, batch_size=128
    )
    print(f"\nTrain batches: {len(train_loader)}")
    if val_loader:
        print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
