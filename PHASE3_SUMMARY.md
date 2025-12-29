# PHASE 3 COMPLETE — CNN DISTILLATION PIPELINE

## Overview

Complete implementation of CNN knowledge distillation pipeline:
**ResNet-50 (Teacher) → ResNet-34 (Assistant) → ResNet-18 (Student)**

---

## ✅ Deliverables

### 1. ResNet Models (`src/models/cnn/resnet.py`)

**Features:**
- ResNet-18, ResNet-34, ResNet-50 support
- ImageNet pretrained weights loading
- Custom classifier for any number of classes
- Feature extraction capability
- Freezing backbone option

**Usage:**
```python
from src.models.cnn.resnet import get_resnet_model

model = get_resnet_model(
    model_name='resnet18',
    num_classes=10,
    pretrained=True
)
```

---

### 2. CNN Training Script (`cnn_distillation/train.py`)

**Complete Training Pipeline:**
- Teacher training (standalone)
- Assistant distillation (teacher → assistant)
- Student distillation (assistant → student)
- Sequential or individual stage training

**Key Features:**
- ✅ Mixed precision training (AMP)
- ✅ Multiple optimizers (Adam, AdamW, SGD)
- ✅ Learning rate schedulers (Cosine, Step)
- ✅ Knowledge distillation loss
- ✅ Comprehensive metrics (Accuracy, Precision, Recall, F1)
- ✅ CSV logging per epoch
- ✅ GPU memory monitoring
- ✅ Automatic checkpoint saving
- ✅ Reproducibility (seed setting)
- ✅ Progress bars (tqdm)

**Training Modes:**
```bash
# Train full pipeline sequentially
python train.py --dataset cifar10 --pretrained --stage all

# Train individual stages
python train.py --dataset cifar10 --pretrained --stage teacher --epochs 300
python train.py --dataset cifar10 --pretrained --stage assistant --epochs 300
python train.py --dataset cifar10 --pretrained --stage student --epochs 300

# Train from scratch
python train.py --dataset cifar10 --stage all --epochs 300

# Custom configuration
python train.py --dataset cifar10 --pretrained --stage teacher \
    --epochs 500 --batch-size 64 --lr 0.0001 \
    --optimizer adamw --scheduler cosine \
    --mixed-precision --seed 42
```

**Command-Line Arguments:**
```
Dataset:
  --dataset              Dataset name (cifar10, pathmnist, etc.)
  --num-workers         DataLoader workers (default: 4)

Model:
  --stage               Training stage (teacher, assistant, student, all)
  --model-teacher       Teacher model (default: resnet50)
  --model-assistant     Assistant model (default: resnet34)
  --model-student       Student model (default: resnet18)
  --pretrained          Use ImageNet pretrained weights

Training:
  --epochs              Number of epochs (default: 300)
  --batch-size          Batch size (default: 32)
  --lr                  Learning rate (default: 0.001)
  --optimizer           Optimizer (adam, adamw, sgd)
  --scheduler           LR scheduler (cosine, step, none)
  --weight-decay        Weight decay (default: 1e-4)

Knowledge Distillation:
  --temperature         KD temperature (default: 4.0)
  --alpha               KD alpha / soft target weight (default: 0.7)

Hardware:
  --device              Device (cuda, cpu)
  --mixed-precision     Enable AMP

Reproducibility:
  --seed                Random seed (default: 42)
  --no-deterministic    Disable deterministic mode

Logging:
  --log-interval        Log interval in batches (default: 10)
  --save-interval       Checkpoint save interval in epochs (default: 50)
```

---

### 3. Feature Map Visualization (`src/visualization/cnn_viz.py`)

**Capabilities:**
- Extract features from any CNN layer
- Visualize multiple feature maps in grid layout
- Save visualizations to disk
- Batch visualization for multiple samples

**Functions:**
```python
from src.visualization.cnn_viz import (
    visualize_feature_maps,
    extract_and_visualize_features,
    save_feature_maps
)

# Visualize pre-extracted features
visualize_feature_maps(
    feature_maps=features,
    num_maps=16,
    title="Layer 4 Features",
    save_path="visualizations/features.png"
)

# Extract and visualize from model
extract_and_visualize_features(
    model=model,
    input_tensor=image,
    layer_name='layer4',
    num_maps=16,
    save_path="visualizations/layer4.png"
)

# Batch visualization
save_feature_maps(
    model=model,
    data_loader=val_loader,
    layer_name='layer4',
    save_dir='visualizations/resnet18',
    num_samples=5
)
```

---

### 4. Memory Cleanup Scripts

#### Main Cleanup (`cnn_distillation/clean.sh`)
```bash
#!/bin/bash
# Clears CUDA cache, Python GC, optional system cache
bash clean.sh
```

#### Per-Experiment Cleanup
Automatically created for each experiment:
```
cnn_distillation/
├── imagenet_pretrained/
│   └── cifar10/
│       ├── teacher/
│       │   └── clean.sh      # Auto-generated
│       ├── assistant/
│       │   └── clean.sh
│       └── student/
│           └── clean.sh
└── from_scratch/
    └── ...
```

---

### 5. Directory Structure

```
cnn_distillation/
├── imagenet_pretrained/          # ImageNet pretrained experiments
│   ├── cifar10/
│   │   ├── teacher/              # ResNet-50 training
│   │   │   ├── logs/
│   │   │   │   ├── resnet50_teacher_log.csv
│   │   │   │   └── resnet50_teacher_config.json
│   │   │   ├── checkpoints/
│   │   │   │   ├── latest.pth
│   │   │   │   └── best_acc_95.23.pth
│   │   │   ├── visualizations/
│   │   │   │   └── feature_maps_*.png
│   │   │   └── clean.sh
│   │   ├── assistant/            # ResNet-34 distillation
│   │   │   └── ...
│   │   └── student/              # ResNet-18 distillation
│   │       └── ...
│   └── medmnist/
│       └── ...
├── from_scratch/                 # Train from scratch experiments
│   ├── cifar10/
│   └── medmnist/
├── logs/                         # Consolidated logs
├── results/                      # Final results
├── visualize/                    # Visualization outputs
├── clean.sh                      # Main cleanup script
├── train.py                      # Training script
└── README.md                     # Documentation
```

---

## 📊 Logged Metrics (Per Epoch)

CSV format with the following columns:

```csv
epoch,train_loss,train_accuracy,train_precision,train_recall,train_f1_score,
val_loss,val_accuracy,val_precision,val_recall,val_f1_score,
learning_rate,epoch_time_sec,gpu_memory_mb
```

**Example Row:**
```
50,0.3421,92.35,92.10,92.05,92.08,0.4123,91.20,90.95,90.88,90.92,0.000523,45.2,2048.5
```

---

## 🚀 Training Workflow

### Step 1: Prepare Datasets
```bash
python dataset_prepare.py --all
```

### Step 2: Train Teacher Model
```bash
cd cnn_distillation
python train.py --dataset cifar10 --pretrained --stage teacher --epochs 300
```

**Output:**
- Logs: `imagenet_pretrained/cifar10/teacher/logs/`
- Checkpoints: `imagenet_pretrained/cifar10/teacher/checkpoints/`
- Best model: `best_acc_*.pth`

### Step 3: Train Assistant (Distillation)
```bash
python train.py --dataset cifar10 --pretrained --stage assistant --epochs 300
```

**Process:**
- Loads teacher checkpoint automatically
- Distills knowledge to ResNet-34
- Uses KD loss (70% soft, 30% hard)
- Temperature: 4.0

### Step 4: Train Student (Distillation)
```bash
python train.py --dataset cifar10 --pretrained --stage student --epochs 300
```

**Process:**
- Loads assistant checkpoint automatically
- Distills knowledge to ResNet-18
- Same KD configuration

### Step 5: Visualize Results
```python
from src.visualization.cnn_viz import save_feature_maps

save_feature_maps(
    model=student_model,
    data_loader=val_loader,
    layer_name='layer4',
    save_dir='visualize/student',
    num_samples=10
)
```

---

## 🔬 Technical Details

### Knowledge Distillation Loss

```
L_total = α × L_KD + (1 - α) × L_CE

where:
  L_KD = KL(Teacher_soft || Student_soft) × T²
  L_CE = CrossEntropy(Student_logits, True_labels)
  T = Temperature (4.0)
  α = Balance parameter (0.7)
```

### Model Sizes

| Model | Parameters | Memory (FP32) | Memory (FP16) |
|-------|------------|---------------|---------------|
| ResNet-50 | 23.5M | ~94 MB | ~47 MB |
| ResNet-34 | 21.3M | ~85 MB | ~43 MB |
| ResNet-18 | 11.2M | ~45 MB | ~23 MB |

### Training Configuration

**Default Hyperparameters:**
- Epochs: 300
- Batch size: 32
- Learning rate: 0.001
- Optimizer: AdamW
- Weight decay: 1e-4
- LR scheduler: Cosine annealing
- KD temperature: 4.0
- KD alpha: 0.7

**Mixed Precision Training:**
- Enables AMP for ~2x speedup
- Reduces memory usage by ~40%
- Minimal accuracy impact

---

## 📈 Expected Performance

### CIFAR-10 (ImageNet Pretrained)

| Stage | Model | Accuracy | Precision | Recall | F1-Score | Training Time* |
|-------|-------|----------|-----------|--------|----------|----------------|
| Teacher | ResNet-50 | ~95% | ~95% | ~95% | ~95% | ~4 hours |
| Assistant | ResNet-34 | ~94% | ~94% | ~94% | ~94% | ~3.5 hours |
| Student | ResNet-18 | ~93% | ~93% | ~93% | ~93% | ~3 hours |

*On RTX 4050 (6GB VRAM)

### CIFAR-10 (From Scratch)

| Stage | Model | Accuracy | Precision | Recall | F1-Score |
|-------|-------|----------|-----------|--------|----------|
| Teacher | ResNet-50 | ~92% | ~92% | ~92% | ~92% |
| Assistant | ResNet-34 | ~90% | ~90% | ~90% | ~90% |
| Student | ResNet-18 | ~88% | ~88% | ~88% | ~88% |

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Solutions:**
```bash
# Reduce batch size
python train.py --batch-size 16

# Enable mixed precision
python train.py --mixed-precision

# Clear memory before training
bash clean.sh

# Train on smaller model first
python train.py --stage student  # ResNet-18 only
```

### Teacher Checkpoint Not Found

**Error:** `FileNotFoundError: Teacher checkpoint not found`

**Solution:**
```bash
# Train teacher first
python train.py --stage teacher --epochs 300

# Then train assistant
python train.py --stage assistant --epochs 300
```

### Low Accuracy

**Solutions:**
```bash
# Increase epochs
python train.py --epochs 500

# Adjust learning rate
python train.py --lr 0.0001

# Try different KD temperature
python train.py --temperature 6.0 --alpha 0.8

# Use pretrained weights
python train.py --pretrained
```

---

## ✅ Integration with Shared Core

CNN pipeline fully integrates with Phase 2 components:

- ✅ **KD Loss**: `KnowledgeDistillationLoss` from `src.distillation.kd_loss`
- ✅ **Metrics**: `MetricsCalculator` from `src.distillation.metrics`
- ✅ **Logging**: `EpochLogger` from `src.utils.logger`
- ✅ **Reproducibility**: `setup_reproducibility` from `src.utils.reproducibility`
- ✅ **GPU Monitoring**: `get_gpu_memory_usage` from `src.utils.gpu_monitor`
- ✅ **Memory Cleanup**: `full_cleanup` from `src.utils.memory_cleanup`

---

## 📝 Files Created

1. `src/models/__init__.py` - Models module init
2. `src/models/cnn/__init__.py` - CNN module init
3. `src/models/cnn/resnet.py` - ResNet models (3.5KB)
4. `cnn_distillation/README.md` - Comprehensive documentation (7KB)
5. `cnn_distillation/train.py` - Training script (15KB)
6. `cnn_distillation/clean.sh` - Memory cleanup script
7. `src/visualization/__init__.py` - Visualization module init
8. `src/visualization/cnn_viz.py` - Feature map visualization (7KB)

---

## 🎯 Next Steps (PHASE 4)

- ViT model definitions
- ViT distillation pipeline
- ViT attention map visualization
- Memory-efficient ViT training for 6GB VRAM

---

**Repository:** [View on GitHub](https://github.com/Sourodyuti/Efficient-Knowledge-Distillation-for-Lightweight-Image-Classification)

**Status:** ✅ CNN Pipeline Complete | Ready for PHASE 4 (ViT Pipeline)
