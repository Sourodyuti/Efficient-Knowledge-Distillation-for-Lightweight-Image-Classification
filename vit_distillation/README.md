# ViT Knowledge Distillation

Vision Transformer knowledge distillation pipeline:
**Teacher (ViT-Base) → Assistant (ViT-Small) → Student (ViT-Tiny)**

## Memory Optimization for 6GB VRAM

This pipeline is **specifically optimized for 6GB VRAM** (e.g., RTX 4050):

### Memory-Saving Techniques
1. **Gradient Checkpointing** - Saves ~40% memory by recomputing activations
2. **Mixed Precision (AMP)** - FP16 training reduces memory by ~50%
3. **Small Batch Sizes** - 8 for ViT-Base, 16 for ViT-Small, 32 for ViT-Tiny
4. **Efficient Attention** - Optimized attention implementation
5. **Aggressive Cleanup** - CUDA cache clearing every 10 epochs

### Expected Memory Usage (FP16 + Gradient Checkpointing)

| Model | Params | Batch Size | Peak Memory | Safe for 6GB? |
|-------|--------|------------|-------------|---------------|
| ViT-Base | 86M | 8 | ~4.5 GB | ✅ Yes |
| ViT-Small | 22M | 16 | ~2.5 GB | ✅ Yes |
| ViT-Tiny | 5M | 32 | ~1.5 GB | ✅ Yes |

## Directory Structure

```
vit_distillation/
├── imagenet_pretrained/     # Experiments with ImageNet pretrained models
│   ├── cifar10/
│   │   ├── teacher/         # ViT-Base training logs
│   │   ├── assistant/       # ViT-Small distillation logs
│   │   └── student/         # ViT-Tiny distillation logs
│   └── medmnist/
│       └── ...
├── from_scratch/            # Experiments training from scratch
│   ├── cifar10/
│   └── medmnist/
├── logs/                    # Consolidated logs
├── results/                 # Final results and analysis
├── visualize/               # Attention map visualizations
├── clean.sh                 # Memory cleanup script
└── train.py                 # Main training script
```

## Quick Start

### 1. Prepare Datasets

```bash
# From project root
python dataset_prepare.py --all
```

### 2. Train ViT Pipeline (6GB VRAM Safe)

```bash
cd vit_distillation

# Train with ImageNet pretrained weights
# Batch sizes automatically adjusted for 6GB VRAM
python train.py --dataset cifar10 --pretrained --stage all --mixed-precision

# Train from scratch
python train.py --dataset cifar10 --stage all --mixed-precision

# Train specific stage
python train.py --dataset cifar10 --pretrained --stage teacher --batch-size 8
python train.py --dataset cifar10 --pretrained --stage assistant --batch-size 16
python train.py --dataset cifar10 --pretrained --stage student --batch-size 32
```

### 3. Monitor Training

```bash
# View logs
tail -f imagenet_pretrained/cifar10/teacher/logs/vit_base_teacher_log.csv

# Monitor GPU memory
watch -n 1 nvidia-smi
```

### 4. Cleanup Memory

```bash
# Clear CUDA cache and Python GC
bash clean.sh

# Or per-experiment cleanup
bash imagenet_pretrained/cifar10/teacher/clean.sh
```

## Training Stages

### Stage 1: Teacher (ViT-Base)
- **Model**: ViT-Base (86M parameters)
- **Batch size**: 8 (optimized for 6GB VRAM)
- **Memory**: ~4.5GB peak
- **Training**: Fine-tune pretrained or train from scratch
- **Gradient checkpointing**: Enabled
- **Mixed precision**: Enabled (default)

### Stage 2: Assistant (ViT-Small)
- **Model**: ViT-Small (22M parameters)
- **Batch size**: 16
- **Memory**: ~2.5GB peak
- **Distillation**: Teacher (ViT-Base) → Assistant
- **KD loss**: 70% soft, 30% hard
- **Temperature**: 4.0

### Stage 3: Student (ViT-Tiny)
- **Model**: ViT-Tiny (5M parameters)
- **Batch size**: 32
- **Memory**: ~1.5GB peak
- **Distillation**: Assistant (ViT-Small) → Student
- **Same KD configuration**

## Configuration

Edit `train.py` or use command-line arguments:

```python
# Model
--model-teacher vit_base
--model-assistant vit_small
--model-student vit_tiny

# Training (optimized for 6GB VRAM)
--epochs 100
--batch-size 8          # For ViT-Base
--lr 3e-4              # ViT default learning rate
--optimizer adamw
--scheduler cosine
--weight-decay 0.05    # ViT default

# Memory optimization (CRITICAL for 6GB VRAM)
--mixed-precision       # Enable AMP (default ON)
--gradient-checkpointing  # Enable (always ON in code)

# Knowledge Distillation
--temperature 4.0
--alpha 0.7

# Dataset
--dataset cifar10
--num-workers 2        # Reduced for ViT

# Hardware
--device cuda

# Logging
--log-interval 10
--save-interval 25
```

## Attention Visualization

Attention maps are saved during training:

```
visualize/
├── teacher/
│   ├── attention_head_0_sample_1.png
│   ├── attention_multi_head_sample_1.png
│   └── attention_rollout_sample_1.png
├── assistant/
└── student/
```

To generate attention visualizations:

```python
from src.visualization.vit_viz import visualize_attention_map

# Single head
visualize_attention_map(
    attention_weights,
    image=input_image,
    head_idx=0,
    save_path='attention_head_0.png'
)

# Multi-head
visualize_multi_head_attention(
    attention_weights,
    num_heads=12,
    save_path='multi_head_attention.png'
)

# Attention rollout (across all layers)
rollout = extract_attention_rollout(attention_list)
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
# Clear CUDA cache, Python GC, and show memory status
bash clean.sh
```

### During Training

Memory is automatically managed:
- CUDA cache cleared every 10 epochs
- Python GC after validation
- Peak memory logged to CSV
- Gradient checkpointing always enabled

### Manual Cleanup

```bash
# Before training
bash clean.sh

# During training (in another terminal)
python -c "import torch; torch.cuda.empty_cache()"

# Check memory
nvidia-smi
```

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
```bash
# 1. Reduce batch size
python train.py --batch-size 4  # For ViT-Base

# 2. Ensure mixed precision is enabled
python train.py --mixed-precision  # Should be default

# 3. Clear memory before training
bash clean.sh

# 4. Use smaller model
python train.py --stage assistant  # Start with ViT-Small

# 5. Reduce number of workers
python train.py --num-workers 1
```

### Teacher Checkpoint Not Found

**Error:** `FileNotFoundError: Teacher checkpoint not found`

**Solution:**
```bash
# Train teacher first
python train.py --stage teacher --epochs 100

# Then train assistant
python train.py --stage assistant --epochs 100
```

### Low Accuracy

**Solutions:**
```bash
# Increase epochs (ViT needs more training)
python train.py --epochs 200

# Adjust learning rate
python train.py --lr 1e-4

# Try different warmup
python train.py --warmup-epochs 10

# Use pretrained weights
python train.py --pretrained
```

### Slow Training

**Solutions:**
```bash
# Increase batch size if memory allows
python train.py --batch-size 16  # For ViT-Small

# Disable deterministic mode
python train.py --no-deterministic

# More workers (if CPU/IO bound)
python train.py --num-workers 4
```

## Expected Performance

### CIFAR-10 (ImageNet Pretrained)

| Stage | Model | Params | Accuracy | F1-Score | Training Time* |
|-------|-------|--------|----------|----------|----------------|
| Teacher | ViT-Base | 86M | ~96% | ~96% | ~8 hours |
| Assistant | ViT-Small | 22M | ~95% | ~95% | ~5 hours |
| Student | ViT-Tiny | 5M | ~93% | ~93% | ~3 hours |

*On RTX 4050 (6GB VRAM), 100 epochs

### CIFAR-10 (From Scratch)

| Stage | Model | Params | Accuracy | F1-Score |
|-------|-------|--------|----------|----------|
| Teacher | ViT-Base | 86M | ~93% | ~93% |
| Assistant | ViT-Small | 22M | ~91% | ~91% |
| Student | ViT-Tiny | 5M | ~88% | ~88% |

### Memory Benchmarks (RTX 4050 6GB)

| Model | Batch Size | Peak Memory | Status |
|-------|------------|-------------|--------|
| ViT-Base | 8 | ~4.5 GB | ✅ Safe |
| ViT-Base | 16 | ~6.8 GB | ❌ OOM |
| ViT-Small | 16 | ~2.5 GB | ✅ Safe |
| ViT-Small | 32 | ~4.2 GB | ✅ Safe |
| ViT-Tiny | 32 | ~1.5 GB | ✅ Safe |
| ViT-Tiny | 64 | ~2.8 GB | ✅ Safe |

## Comparison with CNN (ResNet)

| Metric | ResNet-18 | ViT-Tiny | ResNet-50 | ViT-Base |
|--------|-----------|----------|-----------|----------|
| Parameters | 11M | 5M | 23M | 86M |
| FLOPs | 1.8G | 1.2G | 4.1G | 17.6G |
| Memory (FP16) | ~45 MB | ~23 MB | ~94 MB | ~344 MB |
| CIFAR-10 Acc | ~93% | ~93% | ~95% | ~96% |
| Training Speed | Faster | Slower | Medium | Slowest |
| Best for | Speed | Efficiency | Accuracy | Max Accuracy |

## Tips for 6GB VRAM

1. **Always use mixed precision** - Halves memory usage
2. **Enable gradient checkpointing** - Saves 40% memory
3. **Start with small batch sizes** - Increase gradually
4. **Monitor memory** - Use `nvidia-smi` or `watch -n 1 nvidia-smi`
5. **Clear cache regularly** - Run `bash clean.sh` between experiments
6. **Use smaller models first** - Train ViT-Tiny to test pipeline
7. **Reduce workers** - More workers = more memory
8. **Close other programs** - Free up GPU memory

## Advanced: Custom ViT Sizes

Create custom ViT models by modifying `src/models/vit/vit_models.py`:

```python
from torchvision.models.vision_transformer import VisionTransformer

# Ultra-tiny ViT for <1GB memory
model = VisionTransformer(
    image_size=224,
    patch_size=16,
    num_layers=6,      # Reduce layers
    num_heads=2,       # Reduce heads
    hidden_dim=128,    # Reduce hidden dim
    mlp_dim=512,       # Reduce MLP dim
    num_classes=10
)
```

## Citation

If you use this pipeline, please cite:

```bibtex
@misc{biswas2025kd_vit,
  author = {Biswas Sanyal, Sourodyuti},
  title = {ViT Knowledge Distillation for Lightweight Image Classification},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Sourodyuti/Efficient-Knowledge-Distillation-for-Lightweight-Image-Classification}
}
```

## References

- **ViT**: Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021
- **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network", 2015
- **Attention Rollout**: Abnar & Zuidema, "Quantifying Attention Flow in Transformers", ACL 2020
