# Datasets Directory

This directory contains all preprocessed datasets for knowledge distillation experiments.

## Overview

All datasets are:
- **Preprocessed once** before training
- **Read-only** during training (never modified)
- **Resized to 224×224** for ImageNet pretrained model compatibility
- **Normalized with ImageNet statistics** (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Datasets

### CIFAR-10
- **Location**: `datasets/cifar10/`
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Samples**: 50,000 train + 10,000 test
- **Original size**: 32×32 (resized to 224×224)
- **Task**: Multi-class classification

### MedMNIST
- **Location**: `datasets/medmnist/{dataset_name}/`
- **Variants**: PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, etc.
- **Original size**: 28×28 or 64×64 (resized to 224×224)
- **Task**: Multi-class or multi-label classification
- **Details**: https://medmnist.com/

## Preparation

### First Time Setup

Run the dataset preparation script **once** before training:

```bash
# Prepare all datasets (CIFAR-10 + PathMNIST)
python dataset_prepare.py --all

# Prepare only CIFAR-10
python dataset_prepare.py --cifar10

# Prepare specific MedMNIST dataset
python dataset_prepare.py --medmnist pathmnist
python dataset_prepare.py --medmnist chestmnist

# Verify all prepared datasets
python dataset_prepare.py --verify
```

### Directory Structure

```
datasets/
├── README.md              # This file
├── cifar10/               # CIFAR-10 dataset
│   ├── cifar-10-batches-py/
│   └── metadata.json      # Dataset info
└── medmnist/              # MedMNIST datasets
    ├── pathmnist/
    │   ├── pathmnist.npz
    │   └── metadata.json
    ├── chestmnist/
    │   ├── chestmnist.npz
    │   └── metadata.json
    └── ...
```

## Usage in Training

### CIFAR-10

```python
from src.data.cifar10_loader import CIFAR10Processor

# Initialize processor
processor = CIFAR10Processor(root_dir='./datasets')

# Get dataloaders (read-only)
dataloaders = processor.get_dataloaders(
    batch_size=32,
    num_workers=4,
    val_split=0.1  # 10% of training for validation
)

train_loader = dataloaders['train_loader']
val_loader = dataloaders['val_loader']
test_loader = dataloaders['test_loader']
```

### MedMNIST

```python
from src.data.medmnist_loader import MedMNISTProcessor

# Initialize processor
processor = MedMNISTProcessor(
    dataset_name='pathmnist',
    root_dir='./datasets'
)

# Get dataloaders (read-only)
dataloaders = processor.get_dataloaders(
    batch_size=32,
    num_workers=4
)

train_loader = dataloaders['train_loader']
val_loader = dataloaders['val_loader']  # MedMNIST has separate val split
test_loader = dataloaders['test_loader']
```

## Data Transformations

### Training Set
- Resize to 256×256
- Random crop to 224×224
- Random horizontal flip (p=0.5)
- Color jitter (brightness, contrast, saturation)
- Normalize with ImageNet statistics

### Validation/Test Set
- Resize to 256×256
- Center crop to 224×224
- Normalize with ImageNet statistics
- **No augmentation**

## Important Notes

1. **One-time preparation**: Run `dataset_prepare.py` only once
2. **Read-only**: Training scripts never modify the datasets
3. **Reproducibility**: Fixed random seeds (seed=42) for deterministic splits
4. **ImageNet compatibility**: All images are 224×224 RGB with ImageNet normalization
5. **Grayscale handling**: MedMNIST grayscale images are automatically converted to RGB (3 channels)

## Validation

Verify dataset integrity:

```bash
python dataset_prepare.py --verify
```

This checks:
- ✓ Metadata files exist
- ✓ Image dimensions (224×224×3)
- ✓ Normalization range
- ✓ Number of samples
- ✓ Class counts

## Storage Requirements

- **CIFAR-10**: ~170 MB
- **PathMNIST**: ~200 MB
- **ChestMNIST**: ~500 MB
- **Other MedMNIST**: Varies (50 MB - 1 GB)

## Troubleshooting

### Dataset not found
```bash
# Re-download dataset
python dataset_prepare.py --cifar10
```

### Verification failed
```bash
# Delete corrupted data and re-download
rm -rf datasets/cifar10/
python dataset_prepare.py --cifar10 --verify
```

### Out of memory during loading
```python
# Reduce batch size and num_workers
dataloaders = processor.get_dataloaders(
    batch_size=16,  # Smaller batch
    num_workers=2   # Fewer workers
)
```

## References

- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- MedMNIST: https://medmnist.com/
- ImageNet normalization: https://pytorch.org/vision/stable/models.html
