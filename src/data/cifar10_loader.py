"""
CIFAR-10 Dataset Loader
Preprocesses CIFAR-10 to 224x224 for ImageNet pretrained models.
Downloads once, reuses for all experiments.
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
import json


class CIFAR10Processor:
    """
    CIFAR-10 dataset processor for knowledge distillation.
    Resizes images to 224x224 for compatibility with ImageNet pretrained models.
    """
    
    def __init__(self, root_dir='./datasets', seed=42):
        """
        Args:
            root_dir (str): Root directory for datasets
            seed (int): Random seed for reproducibility
        """
        self.root_dir = Path(root_dir)
        self.cifar10_dir = self.root_dir / 'cifar10'
        self.seed = seed
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # ImageNet normalization (standard for pretrained models)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Training transforms (with augmentation)
        self.train_transforms = transforms.Compose([
            transforms.Resize(256),  # Resize to 256 first
            transforms.RandomCrop(224),  # Random crop to 224x224
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            self.normalize
        ])
        
        # Validation/Test transforms (no augmentation)
        self.test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),  # Center crop to 224x224
            transforms.ToTensor(),
            self.normalize
        ])
        
        self.num_classes = 10
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    def download_and_prepare(self):
        """
        Download CIFAR-10 dataset if not present.
        Creates directory structure and saves metadata.
        """
        print("=" * 60)
        print("CIFAR-10 Dataset Preparation")
        print("=" * 60)
        
        # Create directories
        self.cifar10_dir.mkdir(parents=True, exist_ok=True)
        
        # Download training data
        print("\n[1/3] Downloading CIFAR-10 training set...")
        train_dataset = torchvision.datasets.CIFAR10(
            root=str(self.cifar10_dir),
            train=True,
            download=True,
            transform=None  # We'll apply transforms during loading
        )
        
        # Download test data
        print("\n[2/3] Downloading CIFAR-10 test set...")
        test_dataset = torchvision.datasets.CIFAR10(
            root=str(self.cifar10_dir),
            train=False,
            download=True,
            transform=None
        )
        
        # Save metadata
        print("\n[3/3] Saving metadata...")
        metadata = {
            'dataset': 'CIFAR-10',
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset),
            'image_size': '224x224 (resized from 32x32)',
            'normalization': 'ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])',
            'seed': self.seed,
            'prepared': True
        }
        
        metadata_path = self.cifar10_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"\n✓ CIFAR-10 preparation complete!")
        print(f"  Location: {self.cifar10_dir}")
        print(f"  Training samples: {metadata['train_samples']}")
        print(f"  Test samples: {metadata['test_samples']}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Metadata saved: {metadata_path}")
        print("=" * 60)
        
        return metadata
    
    def get_dataloaders(self, batch_size=32, num_workers=4, val_split=0.1):
        """
        Get train, validation, and test dataloaders.
        
        Args:
            batch_size (int): Batch size for dataloaders
            num_workers (int): Number of workers for data loading
            val_split (float): Fraction of training data to use for validation
        
        Returns:
            dict: Dictionary containing train_loader, val_loader, test_loader
        """
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root=str(self.cifar10_dir),
            train=True,
            download=False,
            transform=self.train_transforms
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=str(self.cifar10_dir),
            train=False,
            download=False,
            transform=self.test_transforms
        )
        
        # Split train into train and validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        
        split_idx = int(np.floor(val_split * num_train))
        train_indices, val_indices = indices[split_idx:], indices[:split_idx]
        
        # Create subsets
        train_subset = Subset(train_dataset, train_indices)
        
        # Validation uses test transforms (no augmentation)
        val_dataset = torchvision.datasets.CIFAR10(
            root=str(self.cifar10_dir),
            train=True,
            download=False,
            transform=self.test_transforms  # No augmentation for validation
        )
        val_subset = Subset(val_dataset, val_indices)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        print(f"\nDataloaders created:")
        print(f"  Train: {len(train_subset)} samples ({len(train_loader)} batches)")
        print(f"  Val:   {len(val_subset)} samples ({len(val_loader)} batches)")
        print(f"  Test:  {len(test_dataset)} samples ({len(test_loader)} batches)")
        print(f"  Batch size: {batch_size}")
        print(f"  Num workers: {num_workers}")
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }
    
    def verify_dataset(self):
        """
        Verify dataset integrity and proper preprocessing.
        
        Returns:
            bool: True if dataset is valid, False otherwise
        """
        try:
            # Check if metadata exists
            metadata_path = self.cifar10_dir / 'metadata.json'
            if not metadata_path.exists():
                print("❌ Metadata file not found. Run download_and_prepare() first.")
                return False
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if not metadata.get('prepared', False):
                print("❌ Dataset not properly prepared.")
                return False
            
            # Load a sample batch
            train_dataset = torchvision.datasets.CIFAR10(
                root=str(self.cifar10_dir),
                train=True,
                download=False,
                transform=self.test_transforms
            )
            
            # Check first sample
            img, label = train_dataset[0]
            
            # Verify shape
            if img.shape != torch.Size([3, 224, 224]):
                print(f"❌ Invalid image shape: {img.shape}. Expected: [3, 224, 224]")
                return False
            
            # Verify normalization (approximately)
            if not (-3 < img.min() < 3 and -3 < img.max() < 3):
                print(f"❌ Invalid normalization range: [{img.min():.2f}, {img.max():.2f}]")
                return False
            
            print("\n✓ Dataset verification passed!")
            print(f"  Image shape: {img.shape}")
            print(f"  Value range: [{img.min():.3f}, {img.max():.3f}]")
            print(f"  Num classes: {metadata['num_classes']}")
            print(f"  Train samples: {metadata['train_samples']}")
            print(f"  Test samples: {metadata['test_samples']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Verification failed: {str(e)}")
            return False


if __name__ == "__main__":
    # Test the processor
    processor = CIFAR10Processor(root_dir='./datasets')
    processor.download_and_prepare()
    processor.verify_dataset()
