# Efficient Knowledge Distillation for Lightweight Image Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A **production-ready** knowledge distillation framework for training lightweight image classification models, with special optimization for **6GB VRAM** GPUs (RTX 4050).

## 🎓 Overview

This framework implements complete **Teacher → Assistant → Student** knowledge distillation pipelines for both:
- **CNNs** (ResNet-50 → ResNet-34 → ResNet-18)
- **Vision Transformers** (ViT-Base → ViT-Small → ViT-Tiny)

**Key Features:**
- ✅ Memory-optimized for 6GB VRAM (gradient checkpointing + mixed precision)
- ✅ Sequential distillation pipeline
- ✅ Comprehensive metrics tracking
- ✅ Feature map and attention visualization
- ✅ Automated plotting and analysis
- ✅ Production-ready code with validation

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Sourodyuti/Efficient-Knowledge-Distillation-for-Lightweight-Image-Classification.git
cd Efficient-Knowledge-Distillation-for-Lightweight-Image-Classification

# Install dependencies
pip install -r requirements.txt

# Prepare datasets
python dataset_prepare.py --all
```

### Train Your First Model

**CNN (ResNet-18 Student):**
```bash
cd cnn_distillation
python train.py --dataset cifar10 --stage student --epochs 100
```

**ViT (ViT-Tiny Student) - 6GB VRAM Safe:**
```bash
cd vit_distillation
python train.py --dataset cifar10 --stage student --batch-size 32 --mixed-precision
```

### Generate Plots

```bash
python -c "from src.analysis.plot_curves import create_all_plots; create_all_plots('cnn_distillation', 'cnn', 'cifar10', True)"
```

---

## 📚 Documentation

### Complete Guides
- **[Final Usage Guide](FINAL_USAGE_GUIDE.md)** - Comprehensive usage instructions
- **[CNN Pipeline](cnn_distillation/README.md)** - ResNet distillation details
- **[ViT Pipeline](vit_distillation/README.md)** - Vision Transformer with 6GB VRAM optimization
- **[Phase 3 Summary](PHASE3_SUMMARY.md)** - CNN implementation details
- **[Phase 4 Summary](PHASE4_SUMMARY.md)** - ViT implementation details
- **[Phase 5 Summary](PHASE5_SUMMARY.md)** - Finalization and validation

---

## 🎯 Features

### Knowledge Distillation
- **Sequential distillation**: Teacher → Assistant → Student
- **Soft target learning**: Temperature-scaled softmax
- **Balanced loss**: Configurable α between KD and CE loss
- **Flexible temperature**: Default T=4.0

### Architectures

| Architecture | Teacher | Assistant | Student |
|--------------|---------|-----------|----------|
| **CNN** | ResNet-50 (23M) | ResNet-34 (21M) | ResNet-18 (11M) |
| **ViT** | ViT-Base (86M) | ViT-Small (22M) | ViT-Tiny (5M) |

### Datasets
- **CIFAR-10**: 60K images, 10 classes
- **MedMNIST**: Multiple medical imaging subsets
- Extensible to custom datasets

### Memory Optimization (6GB VRAM)
- **Gradient Checkpointing**: Saves ~40% memory
- **Mixed Precision (AMP)**: FP16 reduces memory by ~50%
- **Optimized Batch Sizes**: Auto-adjusted per model
- **Efficient Attention**: Memory-efficient ViT implementation
- **Aggressive Cleanup**: CUDA cache clearing every 10 epochs

### Metrics and Logging
Per-epoch tracking of:
- Accuracy, Precision, Recall, F1-Score
- Training and validation loss
- Learning rate
- Epoch time
- GPU memory usage

### Visualization
- **CNN**: Feature map visualization
- **ViT**: Multi-head attention maps, attention rollout
- **Plots**: Accuracy/loss/F1 vs epoch curves
- **Comparisons**: Multi-model performance analysis

---

## 📊 Performance

### CIFAR-10 Results (ImageNet Pretrained)

| Architecture | Model | Params | Accuracy | F1-Score | Training Time* |
|--------------|-------|--------|----------|----------|----------------|
| CNN | ResNet-50 | 23M | ~95% | ~95% | ~4 hours |
| CNN | ResNet-34 | 21M | ~94% | ~94% | ~3.5 hours |
| CNN | ResNet-18 | 11M | ~93% | ~93% | ~3 hours |
| ViT | ViT-Base | 86M | ~96% | ~96% | ~8 hours |
| ViT | ViT-Small | 22M | ~95% | ~95% | ~5 hours |
| ViT | ViT-Tiny | 5M | ~93% | ~93% | ~3 hours |

*RTX 4050 (6GB VRAM)

### Memory Benchmarks (6GB VRAM)

| Model | Batch Size | Peak Memory | Status |
|-------|------------|-------------|--------|
| ResNet-50 | 32 | ~3.2 GB | ✅ Safe |
| ResNet-18 | 64 | ~2.1 GB | ✅ Safe |
| **ViT-Base** | **8** | **~4.5 GB** | ✅ **Safe** |
| ViT-Small | 16 | ~2.5 GB | ✅ Safe |
| ViT-Tiny | 32 | ~1.5 GB | ✅ Safe |

---

## 🛠️ Project Structure

```
Efficient-Knowledge-Distillation-for-Lightweight-Image-Classification/
├── src/
│   ├── data/              # Dataset loaders and preprocessing
│   ├── models/
│   │   ├── cnn/          # ResNet models
│   │   └── vit/          # Vision Transformer models
│   ├── distillation/     # KD loss and metrics
│   ├── utils/            # Logging, GPU monitoring, reproducibility
│   ├── visualization/    # Feature maps, attention visualization
│   └── analysis/         # Plotting and performance analysis
├── cnn_distillation/
│   ├── imagenet_pretrained/
│   │   └── cifar10/
│   │       ├── teacher/
│   │       ├── assistant/
│   │       └── student/
│   ├── from_scratch/
│   ├── train.py
│   ├── clean.sh
│   └── README.md
├── vit_distillation/
│   ├── imagenet_pretrained/
│   │   └── cifar10/
│   │       ├── teacher/
│   │       ├── assistant/
│   │       └── student/
│   ├── from_scratch/
│   ├── train.py
│   ├── clean.sh
│   └── README.md
├── datasets/
├── dataset_prepare.py
├── validate_project.py
├── requirements.txt
├── FINAL_USAGE_GUIDE.md
└── README.md
```

---

## 💻 Usage Examples

### Example 1: Train Full CNN Pipeline

```bash
# Prepare datasets
python dataset_prepare.py --all

# Train Teacher → Assistant → Student
cd cnn_distillation
python train.py --dataset cifar10 --pretrained --stage all --epochs 300

# Generate plots
cd ..
python -c "from src.analysis.plot_curves import create_all_plots; create_all_plots('cnn_distillation', 'cnn', 'cifar10', True)"
```

### Example 2: Train ViT Pipeline (6GB VRAM)

```bash
# Clear GPU memory
cd vit_distillation
bash clean.sh

# Train Teacher (ViT-Base) - CRITICAL: batch size 8 for 6GB VRAM
python train.py --dataset cifar10 --pretrained --stage teacher --batch-size 8 --epochs 100

# Train Assistant (ViT-Small)
python train.py --dataset cifar10 --pretrained --stage assistant --batch-size 16 --epochs 100

# Train Student (ViT-Tiny)
python train.py --dataset cifar10 --pretrained --stage student --batch-size 32 --epochs 100

# Generate plots
cd ..
python -c "from src.analysis.plot_curves import create_all_plots; create_all_plots('vit_distillation', 'vit', 'cifar10', True)"
```

### Example 3: Custom Analysis

```python
from src.analysis.plot_curves import (
    plot_training_curves,
    plot_comparison,
    create_performance_summary
)

# Plot individual model
plot_training_curves(
    log_path='cnn_distillation/imagenet_pretrained/cifar10/teacher/logs/resnet50_teacher_log.csv',
    save_dir='custom_plots/teacher',
    model_name='ResNet-50 Teacher'
)

# Compare models
log_paths = {
    'Teacher': 'path/to/teacher_log.csv',
    'Assistant': 'path/to/assistant_log.csv',
    'Student': 'path/to/student_log.csv'
}

plot_comparison(log_paths, metric='accuracy', save_path='comparison.png')

# Create summary
summary_df = create_performance_summary(log_paths, save_path='summary.csv')
print(summary_df)
```

---

## 🔧 Configuration

### CNN Training Options

```bash
python cnn_distillation/train.py \
    --dataset cifar10 \
    --stage teacher \
    --model resnet50 \
    --pretrained \
    --epochs 300 \
    --batch-size 32 \
    --lr 0.001 \
    --optimizer adamw \
    --scheduler cosine \
    --temperature 4.0 \
    --alpha 0.7 \
    --mixed-precision \
    --seed 42
```

### ViT Training Options (6GB VRAM)

```bash
python vit_distillation/train.py \
    --dataset cifar10 \
    --stage teacher \
    --model vit_base \
    --pretrained \
    --epochs 100 \
    --batch-size 8 \
    --lr 3e-4 \
    --optimizer adamw \
    --scheduler cosine \
    --weight-decay 0.05 \
    --temperature 4.0 \
    --alpha 0.7 \
    --mixed-precision \
    --num-workers 2 \
    --seed 42
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory (ViT)

```bash
# Reduce batch size
python train.py --batch-size 4

# Clear cache
bash clean.sh

# Use smaller model first
python train.py --stage student
```

### Low Accuracy

```bash
# Increase epochs
python train.py --epochs 500

# Use pretrained weights
python train.py --pretrained

# Adjust learning rate
python train.py --lr 1e-4
```

For more troubleshooting, see [FINAL_USAGE_GUIDE.md](FINAL_USAGE_GUIDE.md).

---

## ✅ Project Validation

Validate project integrity:

```bash
python validate_project.py
```

Checks:
- ✅ Folder isolation (CNN/ViT separate)
- ✅ Dataset immutability
- ✅ Log consistency
- ✅ Code structure
- ✅ Documentation completeness

---

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{biswas2025kd_framework,
  author = {Biswas Sanyal, Sourodyuti},
  title = {Efficient Knowledge Distillation for Lightweight Image Classification},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Sourodyuti/Efficient-Knowledge-Distillation-for-Lightweight-Image-Classification}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🚀 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 💬 Contact

For questions or issues:
- Open an issue on GitHub
- Email: sourodyuti.biswas.sanyal.14@gmail.com

---

## 🙏 Acknowledgments

- **PyTorch** for the deep learning framework
- **torchvision** for pretrained models
- **MedMNIST** for medical imaging datasets
- Knowledge distillation research community

---

## 📊 Project Status

- ✅ Phase 1: Dataset Pipeline - **Complete**
- ✅ Phase 2: Shared Distillation Core - **Complete**
- ✅ Phase 3: CNN Distillation - **Complete**
- ✅ Phase 4: ViT Distillation (6GB VRAM) - **Complete**
- ✅ Phase 5: Finalization - **Complete**

**Status: Production Ready ✅**

---

**Built with ❤️ by [Sourodyuti Biswas Sanyal](https://github.com/Sourodyuti)**
