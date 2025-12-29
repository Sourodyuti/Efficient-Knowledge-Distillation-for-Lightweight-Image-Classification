# Complete Usage Guide - Knowledge Distillation Framework

## Project Overview

This framework implements **complete knowledge distillation pipelines** for both CNNs and Vision Transformers, with special optimization for **6GB VRAM** (RTX 4050).

**Repository:** [Efficient-Knowledge-Distillation-for-Lightweight-Image-Classification](https://github.com/Sourodyuti/Efficient-Knowledge-Distillation-for-Lightweight-Image-Classification)

---

## 🚀 Quick Start (5 Minutes)

### 1. Clone Repository
```bash
git clone https://github.com/Sourodyuti/Efficient-Knowledge-Distillation-for-Lightweight-Image-Classification.git
cd Efficient-Knowledge-Distillation-for-Lightweight-Image-Classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Datasets
```bash
python dataset_prepare.py --all
```

### 4. Train Your First Model

**CNN (ResNet-18):**
```bash
cd cnn_distillation
python train.py --dataset cifar10 --stage student --epochs 100
```

**ViT (ViT-Tiny) - 6GB VRAM Safe:**
```bash
cd vit_distillation
python train.py --dataset cifar10 --stage student --batch-size 32 --mixed-precision
```

### 5. Generate Plots
```bash
python -c "from src.analysis.plot_curves import create_all_plots; create_all_plots('cnn_distillation', 'cnn', 'cifar10', True)"
```

---

## 📚 Complete Workflows

### CNN Distillation Pipeline

**Full Sequential Training:**
```bash
cd cnn_distillation

# Train all stages: Teacher (ResNet-50) → Assistant (ResNet-34) → Student (ResNet-18)
python train.py \
    --dataset cifar10 \
    --pretrained \
    --stage all \
    --epochs 300 \
    --mixed-precision

# Generate plots
cd ..
python -c "from src.analysis.plot_curves import create_all_plots; create_all_plots('cnn_distillation', 'cnn', 'cifar10', True)"
```

**Individual Stage Training:**
```bash
cd cnn_distillation

# Stage 1: Teacher
python train.py --dataset cifar10 --pretrained --stage teacher --epochs 300

# Stage 2: Assistant (requires trained teacher)
python train.py --dataset cifar10 --pretrained --stage assistant --epochs 300

# Stage 3: Student (requires trained assistant)
python train.py --dataset cifar10 --pretrained --stage student --epochs 300
```

---

### ViT Distillation Pipeline (6GB VRAM Optimized)

**Full Sequential Training:**
```bash
cd vit_distillation

# Train all stages with automatic batch size adjustment
python train.py \
    --dataset cifar10 \
    --pretrained \
    --stage all \
    --epochs 100 \
    --mixed-precision

# Generate plots
cd ..
python -c "from src.analysis.plot_curves import create_all_plots; create_all_plots('vit_distillation', 'vit', 'cifar10', True)"
```

**Individual Stage Training (6GB VRAM Safe):**
```bash
cd vit_distillation

# Stage 1: Teacher (ViT-Base) - CRITICAL: Batch size 8 for 6GB VRAM
python train.py \
    --dataset cifar10 \
    --pretrained \
    --stage teacher \
    --batch-size 8 \
    --epochs 100 \
    --mixed-precision

# Stage 2: Assistant (ViT-Small)
python train.py \
    --dataset cifar10 \
    --pretrained \
    --stage assistant \
    --batch-size 16 \
    --epochs 100 \
    --mixed-precision

# Stage 3: Student (ViT-Tiny)
python train.py \
    --dataset cifar10 \
    --pretrained \
    --stage student \
    --batch-size 32 \
    --epochs 100 \
    --mixed-precision
```

---

## 📊 Plotting and Analysis

### Generate All Plots

**For CNN Experiments:**
```python
from src.analysis.plot_curves import create_all_plots

create_all_plots(
    experiment_dir='cnn_distillation',
    architecture='cnn',
    dataset='cifar10',
    pretrained=True
)
```

**For ViT Experiments:**
```python
from src.analysis.plot_curves import create_all_plots

create_all_plots(
    experiment_dir='vit_distillation',
    architecture='vit',
    dataset='cifar10',
    pretrained=True
)
```

### Custom Plots

**Individual Training Curves:**
```python
from src.analysis.plot_curves import plot_training_curves

plot_training_curves(
    log_path='cnn_distillation/imagenet_pretrained/cifar10/teacher/logs/resnet50_teacher_log.csv',
    save_dir='results/plots/teacher',
    model_name='ResNet-50 Teacher'
)
```

**Model Comparison:**
```python
from src.analysis.plot_curves import plot_comparison

log_paths = {
    'Teacher (ResNet-50)': 'cnn_distillation/imagenet_pretrained/cifar10/teacher/logs/resnet50_teacher_log.csv',
    'Assistant (ResNet-34)': 'cnn_distillation/imagenet_pretrained/cifar10/assistant/logs/resnet34_assistant_log.csv',
    'Student (ResNet-18)': 'cnn_distillation/imagenet_pretrained/cifar10/student/logs/resnet18_student_log.csv'
}

plot_comparison(
    log_paths=log_paths,
    metric='accuracy',
    save_path='results/comparison_accuracy.png'
)
```

---

## 🛠️ Configuration Options

### CNN Training Arguments

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

### ViT Training Arguments (6GB VRAM)

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

## 💾 Memory Management

### Before Training
```bash
# Clear all caches
bash cnn_distillation/clean.sh
# or
bash vit_distillation/clean.sh

# Check GPU memory
nvidia-smi
```

### During Training
```bash
# Monitor in real-time
watch -n 1 nvidia-smi
```

### If Out of Memory (ViT)
```bash
# Reduce batch size
python train.py --batch-size 4  # For ViT-Base

# Clear cache manually
python -c "import torch; torch.cuda.empty_cache()"

# Use smaller model
python train.py --stage student  # Train ViT-Tiny first
```

---

## 📋 Project Validation

### Run Validation Checks
```bash
python validate_project.py
```

This validates:
- ✅ Folder isolation (CNN/ViT experiments separate)
- ✅ Dataset immutability (datasets not modified)
- ✅ Log consistency (all required columns present)
- ✅ Code structure
- ✅ Documentation

---

## 📁 Project Structure

```
Efficient-Knowledge-Distillation-for-Lightweight-Image-Classification/
├── src/
│   ├── data/              # Dataset loaders
│   ├── models/
│   │   ├── cnn/          # ResNet models
│   │   └── vit/          # ViT models
│   ├── distillation/     # KD loss and metrics
│   ├── utils/            # Logging, reproducibility, GPU monitoring
│   ├── visualization/    # Feature maps, attention visualization
│   └── analysis/         # Plotting scripts
├── cnn_distillation/
│   ├── imagenet_pretrained/
│   ├── from_scratch/
│   ├── train.py
│   ├── clean.sh
│   └── README.md
├── vit_distillation/
│   ├── imagenet_pretrained/
│   ├── from_scratch/
│   ├── train.py
│   ├── clean.sh
│   └── README.md
├── datasets/             # Auto-downloaded datasets
├── dataset_prepare.py   # Dataset preparation script
├── validate_project.py  # Project validation
└── FINAL_USAGE_GUIDE.md # This file
```

---

## 🚑 Troubleshooting

### Common Issues

**1. CUDA Out of Memory (ViT)**
```bash
# Solution 1: Reduce batch size
python train.py --batch-size 4

# Solution 2: Clear cache
bash clean.sh

# Solution 3: Use smaller model first
python train.py --stage student
```

**2. Teacher Checkpoint Not Found**
```bash
# Train teacher first
python train.py --stage teacher --epochs 100

# Then train assistant
python train.py --stage assistant --epochs 100
```

**3. Low Accuracy**
```bash
# Increase epochs
python train.py --epochs 500

# Use pretrained weights
python train.py --pretrained

# Adjust learning rate
python train.py --lr 1e-4
```

**4. Slow Training**
```bash
# Increase batch size (if memory allows)
python train.py --batch-size 64

# More workers
python train.py --num-workers 8

# Disable deterministic mode
python train.py --no-deterministic
```

---

## 🏆 Expected Results

### CIFAR-10 (ImageNet Pretrained)

| Architecture | Model | Accuracy | F1-Score | Training Time* |
|--------------|-------|----------|----------|----------------|
| CNN | ResNet-50 | ~95% | ~95% | ~4 hours |
| CNN | ResNet-34 | ~94% | ~94% | ~3.5 hours |
| CNN | ResNet-18 | ~93% | ~93% | ~3 hours |
| ViT | ViT-Base | ~96% | ~96% | ~8 hours |
| ViT | ViT-Small | ~95% | ~95% | ~5 hours |
| ViT | ViT-Tiny | ~93% | ~93% | ~3 hours |

*RTX 4050 (6GB VRAM)

---

## 📚 Additional Resources

- **CNN Pipeline**: See `cnn_distillation/README.md`
- **ViT Pipeline**: See `vit_distillation/README.md`
- **Phase 3 Summary**: See `PHASE3_SUMMARY.md`
- **Phase 4 Summary**: See `PHASE4_SUMMARY.md`
- **Repository**: [GitHub](https://github.com/Sourodyuti/Efficient-Knowledge-Distillation-for-Lightweight-Image-Classification)

---

## ❓ Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the README files in each module
3. Run `python validate_project.py` to check setup
4. Open an issue on GitHub

---

**Happy Distilling! 🎓**
