"""
MedMNIST Dataset Loader
Supports multiple MedMNIST datasets with 224x224 preprocessing.
Compatible with ImageNet pretrained models.
"""

import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import json
from PIL import Image

try:
    import medmnist
    from medmnist import INFO
except ImportError:
    print("MedMNIST not installed. Install with: pip install medmnist")
    medmnist = None
    INFO = None


class MedMNISTProcessor:
    """
    MedMNIST dataset processor for knowledge distillation.
    Resizes images to 224x224 for compatibility with ImageNet pretrained models.
    """
    
    AVAILABLE_DATASETS = [
        'pathmnist',    # Pathology (100k images, 9 classes)
        'chestmnist',   # Chest X-Ray (112k images, 14 classes)
        'dermamnist',   # Dermatoscopy (10k images, 7 classes)
        'octmnist',     # OCT (109k images, 4 classes)
        'pneumoniamnist', # Pneumonia (5.9k images, 2 classes)
        'retinamnist',  # Retinal OCT (1.6k images, 5 classes)
        'breastmnist',  # Breast Ultrasound (780 images, 2 classes)
        'bloodmnist',   # Blood Cell (17k images, 8 classes)
        'tissuemnist',  # Tissue (236k images, 8 classes)
        'organamnist',  # Organ (59k images, 11 classes)
        'organcmnist',  # Organ (23k images, 11 classes)
        'organsmnist',  # Organ (26k images, 11 classes)
    ]
    
    def __init__(self, dataset_name='pathmnist', root_dir='./datasets', seed=42):
        """
        Args:
            dataset_name (str): Name of MedMNIST dataset (e.g., 'pathmnist')
            root_dir (str): Root directory for datasets
            seed (int): Random seed for reproducibility
        """
        if medmnist is None:
            raise ImportError("MedMNIST not installed. Install with: pip install medmnist")
        
        if dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not available. Choose from: {self.AVAILABLE_DATASETS}")
        
        self.dataset_name = dataset_name
        self.root_dir = Path(root_dir)
        self.medmnist_dir = self.root_dir / 'medmnist' / dataset_name
        self.seed = seed
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Get dataset info
        self.info = INFO[dataset_name]
        self.num_classes = len(self.info['label'])
        self.task = self.info['task']  # 'multi-label' or 'multi-class'
        
        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Training transforms
        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),  # Convert grayscale to RGB
            self.normalize
        ])
        
        # Test transforms
        self.test_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),  # Convert grayscale to RGB
            self.normalize
        ])
    
    def download_and_prepare(self):
        """
        Download MedMNIST dataset if not present.
        Creates directory structure and saves metadata.
        """
        print("=" * 60)
        print(f"MedMNIST Dataset Preparation: {self.dataset_name.upper()}")
        print("=" * 60)
        
        # Create directories
        self.medmnist_dir.mkdir(parents=True, exist_ok=True)
        
        # Get dataset class
        DataClass = getattr(medmnist, self.info['python_class'])
        
        # Download splits
        print("\n[1/4] Downloading training set...")
        train_dataset = DataClass(
            split='train',
            root=str(self.medmnist_dir),
            download=True,
            transform=None
        )
        
        print("\n[2/4] Downloading validation set...")
        val_dataset = DataClass(
            split='val',
            root=str(self.medmnist_dir),
            download=True,
            transform=None
        )
        
        print("\n[3/4] Downloading test set...")
        test_dataset = DataClass(
            split='test',
            root=str(self.medmnist_dir),
            download=True,
            transform=None
        )
        
        # Save metadata
        print("\n[4/4] Saving metadata...")
        metadata = {
            'dataset': self.dataset_name,
            'description': self.info['description'],
            'task': self.task,
            'num_classes': self.num_classes,
            'class_names': self.info['label'],
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'image_size': '224x224 (resized from 28x28 or 64x64)',
            'normalization': 'ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])',
            'seed': self.seed,
            'prepared': True
        }
        
        metadata_path = self.medmnist_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"\n✓ {self.dataset_name.upper()} preparation complete!")
        print(f"  Location: {self.medmnist_dir}")
        print(f"  Task: {metadata['task']}")
        print(f"  Training samples: {metadata['train_samples']}")
        print(f"  Validation samples: {metadata['val_samples']}")
        print(f"  Test samples: {metadata['test_samples']}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Metadata saved: {metadata_path}")
        print("=" * 60)
        
        return metadata
    
    def get_dataloaders(self, batch_size=32, num_workers=4):
        """
        Get train, validation, and test dataloaders.
        
        Args:
            batch_size (int): Batch size for dataloaders
            num_workers (int): Number of workers for data loading
        
        Returns:
            dict: Dictionary containing train_loader, val_loader, test_loader
        """
        # Get dataset class
        DataClass = getattr(medmnist, self.info['python_class'])
        
        # Custom dataset wrapper for transforms
        class TransformedDataset(Dataset):
            def __init__(self, base_dataset, transform):
                self.base_dataset = base_dataset
                self.transform = transform
            
            def __len__(self):
                return len(self.base_dataset)
            
            def __getitem__(self, idx):
                img, label = self.base_dataset[idx]
                if self.transform:
                    img = self.transform(img)
                # Flatten label if multi-class (single label)
                if label.ndim > 0 and len(label) == 1:
                    label = label[0]
                return img, label
        
        # Load datasets
        train_base = DataClass(split='train', root=str(self.medmnist_dir), download=False)
        val_base = DataClass(split='val', root=str(self.medmnist_dir), download=False)
        test_base = DataClass(split='test', root=str(self.medmnist_dir), download=False)
        
        # Wrap with transforms
        train_dataset = TransformedDataset(train_base, self.train_transforms)
        val_dataset = TransformedDataset(val_base, self.test_transforms)
        test_dataset = TransformedDataset(test_base, self.test_transforms)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
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
        
        print(f"\nDataloaders created for {self.dataset_name}:")
        print(f"  Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
        print(f"  Val:   {len(val_dataset)} samples ({len(val_loader)} batches)")
        print(f"  Test:  {len(test_dataset)} samples ({len(test_loader)} batches)")
        print(f"  Batch size: {batch_size}")
        print(f"  Num workers: {num_workers}")
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'num_classes': self.num_classes,
            'class_names': self.info['label'],
            'task': self.task
        }
    
    def verify_dataset(self):
        """
        Verify dataset integrity and proper preprocessing.
        
        Returns:
            bool: True if dataset is valid, False otherwise
        """
        try:
            # Check metadata
            metadata_path = self.medmnist_dir / 'metadata.json'
            if not metadata_path.exists():
                print("❌ Metadata file not found. Run download_and_prepare() first.")
                return False
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if not metadata.get('prepared', False):
                print("❌ Dataset not properly prepared.")
                return False
            
            # Load sample
            DataClass = getattr(medmnist, self.info['python_class'])
            test_dataset = DataClass(split='test', root=str(self.medmnist_dir), download=False)
            
            # Transform sample
            img, label = test_dataset[0]
            img_transformed = self.test_transforms(img)
            
            # Verify shape
            if img_transformed.shape != torch.Size([3, 224, 224]):
                print(f"❌ Invalid image shape: {img_transformed.shape}. Expected: [3, 224, 224]")
                return False
            
            # Verify normalization
            if not (-3 < img_transformed.min() < 3 and -3 < img_transformed.max() < 3):
                print(f"❌ Invalid normalization: [{img_transformed.min():.2f}, {img_transformed.max():.2f}]")
                return False
            
            print(f"\n✓ {self.dataset_name.upper()} verification passed!")
            print(f"  Image shape: {img_transformed.shape}")
            print(f"  Value range: [{img_transformed.min():.3f}, {img_transformed.max():.3f}]")
            print(f"  Num classes: {metadata['num_classes']}")
            print(f"  Task: {metadata['task']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Verification failed: {str(e)}")
            return False


if __name__ == "__main__":
    # Test with PathMNIST
    processor = MedMNISTProcessor(dataset_name='pathmnist', root_dir='./datasets')
    processor.download_and_prepare()
    processor.verify_dataset()
