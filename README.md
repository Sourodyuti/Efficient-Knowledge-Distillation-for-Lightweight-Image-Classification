# Efficient Knowledge Distillation for Lightweight Image Classification

Research-grade knowledge distillation framework using Teacher → Assistant → Student training pipeline for CNN and Vision Transformer architectures.

## 🎯 Overview

This project implements a comprehensive knowledge distillation pipeline with:
- **Two independent experiments**: CNN (ResNet) and Vision Transformer (ViT)
- **Three-stage distillation**: Teacher → Assistant → Student
- **Multiple datasets**: CIFAR-10 and MedMNIST
- **Pretraining variants**: ImageNet pretrained vs. From scratch
- **Hardware optimization**: Optimized for RTX 4050 (6GB VRAM)

## 🛠️ Architecture

### CNN Pipeline (Independent)
```
ResNet-50 (Teacher)
    ↓ Distill
ResNet-34 (Assistant)
    ↓ Distill
ResNet-18 (Student)
```

### ViT Pipeline (Independent)
```
ViT-Base (Teacher)
    ↓ Distill
ViT-Small (Assistant)
    ↓ Distill
ViT-Tiny (Student)
```

## 📊 Datasets

### CIFAR-10
- 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- 50,000 training + 10,000 test images
- Resized to 224×224 for ImageNet compatibility

### MedMNIST
- Multiple medical imaging datasets (PathMNIST, ChestMNIST, etc.)
- Grayscale/RGB images resized to 224×224
- Multi-class or multi-label classification
- Details: [medmnist.com](https://medmnist.com/)

## ⚡ Features

- **Memory Optimized**: AMP, gradient checkpointing for 6GB VRAM
- **Reproducible**: Fixed seeds, deterministic splits
- **Comprehensive Logging**: CSV logs with accuracy, precision, recall, F1, loss, GPU memory
- **Live Visualization**: CNN feature maps, ViT attention maps, gradient flow
- **Automated Cleanup**: Memory cleanup scripts in every experiment folder
- **Git Integration**: Auto-commit and push functionality

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Sourodyuti/Efficient-Knowledge-Distillation-for-Lightweight-Image-Classification.git
cd Efficient-Knowledge-Distillation-for-Lightweight-Image-Classification

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Datasets

**Run this ONCE before training:**

```bash
# Prepare all datasets (CIFAR-10 + PathMNIST)
python dataset_prepare.py --all

# Or prepare specific datasets
python dataset_prepare.py --cifar10
python dataset_prepare.py --medmnist pathmnist

# Verify datasets
python dataset_prepare.py --verify
```

### 3. Training (Coming Soon)

```bash
# CNN pipeline
python scripts/train_cnn_pipeline.py --dataset cifar10 --pretrained

# ViT pipeline
python scripts/train_vit_pipeline.py --dataset cifar10 --pretrained
```

## 📁 Project Structure

```
├── datasets/                    # Dataset storage (prepared once)
│   ├── cifar10/
│   └── medmnist/
├── src/
│   ├── data/                   # Dataset loaders
│   │   ├── cifar10_loader.py
│   │   └── medmnist_loader.py
│   ├── models/                 # Model definitions (CNN & ViT)
│   ├── distillation/           # KD logic
│   ├── training/               # Training pipelines
│   ├── visualization/          # Live visualization
│   └── utils/                  # Utilities
├── configs/                    # YAML configurations
├── experiments/                # Experiment outputs (separate CNN/ViT)
├── scripts/                    # Execution scripts
├── dataset_prepare.py          # Dataset preparation script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 📊 Logging

### Per-Epoch CSV Logs
Each experiment logs the following metrics per epoch:
- Accuracy (train, val, test)
- Precision, Recall, F1-score
- Cross-Entropy Loss
- Knowledge Distillation Loss
- Learning Rate
- Epoch Time
- GPU Memory Usage

### TensorBoard
Real-time visualization of:
- Loss curves
- Accuracy trends
- Learning rate schedule
- CNN feature maps
- ViT attention maps

## 🔧 Hardware Requirements

### Minimum
- GPU: 6GB VRAM (RTX 4050 or equivalent)
- RAM: 16GB
- Storage: 10GB free space

### Recommended
- GPU: 8GB+ VRAM
- RAM: 32GB
- Storage: 20GB free space
- Ubuntu 24.04 LTS

## 🧪 Memory Optimization

### For ViT-Base (Teacher)
- **Automatic Mixed Precision (AMP)**: FP16 training
- **Gradient Checkpointing**: Reduce activation memory
- **Small Batch Sizes**: 8-16 per GPU
- **Memory-Efficient Attention**: Custom attention implementation

### Cleanup Scripts
Every experiment folder contains `clean.sh`:
```bash
# Clear CUDA cache, Python GC, system memory
bash experiments/cnn/imagenet_pretrained/cifar10/teacher/clean.sh
```

## 📝 Training Configuration

### Hyperparameters
- **Epochs**: ~300 per model
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4 (with cosine annealing)
- **Batch Size**: 16-32 (CNN), 8-16 (ViT)
- **KD Temperature**: 4.0
- **KD Alpha**: 0.7

### Knowledge Distillation Loss
```
L_total = α * L_KD + (1 - α) * L_CE

where:
  L_KD  = KL divergence between teacher and student logits
  L_CE  = Cross-entropy loss on true labels
  α     = Balance parameter (0.7)
```

## 📊 Experiment Organization

Experiments are fully separated by:
1. **Architecture**: CNN vs. ViT
2. **Pretraining**: ImageNet pretrained vs. From scratch
3. **Dataset**: CIFAR-10 vs. MedMNIST
4. **Stage**: Teacher, Assistant, Student

```
experiments/
├── cnn/
│   ├── imagenet_pretrained/
│   │   ├── cifar10/
│   │   │   ├── teacher/    (ResNet-50)
│   │   │   ├── assistant/  (ResNet-34)
│   │   │   └── student/    (ResNet-18)
│   │   └── medmnist/
│   └── from_scratch/
└── vit/
    ├── imagenet_pretrained/
    │   ├── cifar10/
    │   │   ├── teacher/    (ViT-Base)
    │   │   ├── assistant/  (ViT-Small)
    │   │   └── student/    (ViT-Tiny)
    │   └── medmnist/
    └── from_scratch/
```

## 🔬 Verification

### Dataset Integrity
```bash
python dataset_prepare.py --verify
```

Checks:
- ✓ Image shape (224×224×3)
- ✓ Normalization range
- ✓ Number of samples
- ✓ Metadata completeness

## 🐛 Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size
batch_size = 8  # For ViT
batch_size = 16  # For CNN

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Run cleanup
bash experiments/.../clean.sh
```

### Dataset Download Failed
```bash
# Re-download specific dataset
rm -rf datasets/cifar10/
python dataset_prepare.py --cifar10
```

### CUDA Out of Memory During ViT Training
```bash
# Use smaller ViT variant or enable more optimizations
# See src/models/vit/memory_utils.py
```

## 📚 References

### Papers
- Knowledge Distillation: [Hinton et al., 2015](https://arxiv.org/abs/1503.02531)
- ResNet: [He et al., 2016](https://arxiv.org/abs/1512.03385)
- Vision Transformer: [Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929)

### Datasets
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- MedMNIST: https://medmnist.com/

## 📝 License

MIT License - See LICENSE file for details

## ✍️ Author

Sourodyuti Biswas Sanyal
- GitHub: [@Sourodyuti](https://github.com/Sourodyuti)

## 🚀 Roadmap

- [x] Dataset preprocessing pipeline
- [ ] CNN training pipeline
- [ ] ViT training pipeline
- [ ] Live visualization
- [ ] Automated evaluation
- [ ] Results analysis notebooks
- [ ] Pre-trained model weights

## 🔗 Citation

If you use this code in your research, please cite:

```bibtex
@misc{biswas2025kd,
  author = {Biswas Sanyal, Sourodyuti},
  title = {Efficient Knowledge Distillation for Lightweight Image Classification},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Sourodyuti/Efficient-Knowledge-Distillation-for-Lightweight-Image-Classification}
}
```

---

**Status**: 🚧 Active Development | **Phase**: Dataset Pipeline Complete
