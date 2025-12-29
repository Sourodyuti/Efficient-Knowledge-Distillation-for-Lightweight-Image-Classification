# CNN Knowledge Distillation

ResNet-based knowledge distillation pipeline:
**Teacher (ResNet-50) → Assistant (ResNet-34) → Student (ResNet-18)**

## Directory Structure

```
cnn_distillation/
├── imagenet_pretrained/     # Experiments with ImageNet pretrained models
│   ├── cifar10/
│   │   ├── teacher/         # ResNet-50 training logs
│   │   ├── assistant/       # ResNet-34 distillation logs
│   │   └── student/         # ResNet-18 distillation logs
│   └── medmnist/
│       └── ...
├── from_scratch/            # Experiments training from scratch
│   ├── cifar10/
│   └── medmnist/
├── logs/                    # Consolidated logs
├── results/                 # Final results and analysis
├── visualize/               # Visualization outputs
├── clean.sh                 # Memory cleanup script
└── train.py                 # Main training script
```

## Quick Start

### 1. Prepare Datasets

```bash
# From project root
python dataset_prepare.py --all
```

### 2. Train CNN Pipeline

```bash
cd cnn_distillation

# Train with ImageNet pretrained weights
python train.py --dataset cifar10 --pretrained --stage all

# Train from scratch
python train.py --dataset cifar10 --stage all

# Train specific stage
python train.py --dataset cifar10 --pretrained --stage teacher
python train.py --dataset cifar10 --pretrained --stage assistant
python train.py --dataset cifar10 --pretrained --stage student
```

### 3. Monitor Training

```bash
# View logs
tail -f imagenet_pretrained/cifar10/teacher/logs/resnet50_teacher_log.csv

# TensorBoard (if enabled)
tensorboard --logdir=./logs
```

### 4. Cleanup Memory

```bash
# Clear CUDA cache and Python GC
bash clean.sh

# Or per-experiment cleanup
bash imagenet_pretrained/cifar10/teacher/clean.sh
```

## Training Stages

### Stage 1: Teacher (ResNet-50)
- Train teacher model from scratch or fine-tune pretrained
- Save best checkpoint
- Logs: accuracy, precision, recall, F1, loss, LR, GPU memory

### Stage 2: Assistant (ResNet-34)
- Load trained teacher model
- Distill knowledge to ResNet-34
- Uses KD loss (70% soft, 30% hard)
- Temperature: 4.0

### Stage 3: Student (ResNet-18)
- Load trained assistant model
- Distill knowledge to ResNet-18
- Same KD configuration as Stage 2

## Configuration

Edit `train.py` or use command-line arguments:

```python
# Model
--model-teacher resnet50
--model-assistant resnet34
--model-student resnet18

# Training
--epochs 300
--batch-size 32
--lr 0.001
--optimizer adamw
--scheduler cosine

# Knowledge Distillation
--temperature 4.0
--alpha 0.7

# Dataset
--dataset cifar10  # or pathmnist, chestmnist, etc.
--num-workers 4

# Hardware
--device cuda
--mixed-precision  # Enable AMP

# Logging
--log-interval 10  # Log every 10 batches
--save-interval 50  # Save checkpoint every 50 epochs
```

## Visualization

Feature maps are automatically saved during training:

```
visualize/
├── teacher/
│   ├── feature_maps_epoch_50.png
│   └── feature_maps_epoch_100.png
├── assistant/
└── student/
```

## Results

Final results are saved to `results/`:

```
results/
├── teacher_metrics.json
├── assistant_metrics.json
├── student_metrics.json
├── comparison.csv
└── plots/
    ├── accuracy_comparison.png
    └── loss_curves.png
```

## Memory Management

### Automatic Cleanup

Each experiment folder has a `clean.sh` script:

```bash
#!/bin/bash
# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache(); print('CUDA cache cleared')"

# Python GC
python -c "import gc; gc.collect(); print('Python GC completed')"

# System cache (optional, requires sudo)
# sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

### During Training

Memory is automatically managed:
- CUDA cache cleared after each epoch
- Python GC after validation
- Peak memory logged to CSV

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
python train.py --batch-size 16

# Enable mixed precision
python train.py --mixed-precision

# Clear memory before training
bash clean.sh
```

### Low Accuracy

```bash
# Increase epochs
python train.py --epochs 500

# Adjust learning rate
python train.py --lr 0.0001

# Try different KD temperature
python train.py --temperature 6.0
```

### Slow Training

```bash
# Increase num workers
python train.py --num-workers 8

# Enable cudnn benchmark (non-deterministic)
python train.py --no-deterministic
```

## Expected Performance

### CIFAR-10 (ImageNet Pretrained)

| Model | Params | Accuracy | F1-Score |
|-------|--------|----------|----------|
| ResNet-50 (Teacher) | 23.5M | ~95% | ~95% |
| ResNet-34 (Assistant) | 21.3M | ~94% | ~94% |
| ResNet-18 (Student) | 11.2M | ~93% | ~93% |

### CIFAR-10 (From Scratch)

| Model | Params | Accuracy | F1-Score |
|-------|--------|----------|----------|
| ResNet-50 (Teacher) | 23.5M | ~92% | ~92% |
| ResNet-34 (Assistant) | 21.3M | ~90% | ~90% |
| ResNet-18 (Student) | 11.2M | ~88% | ~88% |

*Note: Actual performance may vary based on hyperparameters and training duration.*

## Citation

If you use this pipeline, please cite:

```bibtex
@misc{biswas2025kd_cnn,
  author = {Biswas Sanyal, Sourodyuti},
  title = {CNN Knowledge Distillation for Lightweight Image Classification},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Sourodyuti/Efficient-Knowledge-Distillation-for-Lightweight-Image-Classification}
}
```
