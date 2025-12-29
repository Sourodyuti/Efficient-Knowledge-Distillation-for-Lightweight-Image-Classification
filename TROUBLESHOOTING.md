# 🔧 Troubleshooting Guide

## Quick Fixes for Common Issues

### 🐛 MedMNIST Verification Error

**Problem:**
```
❌ Verification failed: pic should be Tensor or ndarray. Got <class 'PIL.Image.Image'>.
```

**Cause:**
Your local repository has an outdated version of `src/data/medmnist_loader.py` that incorrectly uses `ToPILImage()` transform.

**Solution:**

```bash
# Pull the latest fixes from GitHub
git pull origin main

# Verify you have the latest version
grep -n "ToPILImage" src/data/medmnist_loader.py
# Should return NO results (transforms removed)

# Re-run verification
python dataset_prepare.py --verify
```

**Alternative (if git pull doesn't work):**

```bash
# Download the fixed file directly
wget https://raw.githubusercontent.com/Sourodyuti/Efficient-Knowledge-Distillation-for-Lightweight-Image-Classification/main/src/data/medmnist_loader.py -O src/data/medmnist_loader.py

# Or manually edit src/data/medmnist_loader.py:
# Remove "transforms.ToPILImage()," from line 87 (train_transforms)
# Remove "transforms.ToPILImage()," from line 96 (test_transforms)
```

**Why this happens:**
The MedMNIST library returns PIL Images directly, not numpy arrays. Adding `ToPILImage()` tries to convert PIL Image → PIL Image, which fails.

---

### ⚠️ Deprecation Warning: pynvml

**Problem:**
```
FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead.
```

**Solution:**

```bash
# Remove old package
pip uninstall pynvml -y

# Install new package
pip install nvidia-ml-py>=11.5.0

# Or update all requirements
git pull origin main
pip install -r requirements.txt --upgrade
```

---

### 💾 CUDA Out of Memory

**Problem:**
```
RuntimeError: CUDA out of memory
```

**Solution:**

```bash
# For ViT models on 6GB VRAM:
cd vit_distillation

# Clear GPU memory
bash clean.sh

# Use smaller batch size
python train.py --dataset cifar10 --stage teacher --batch-size 4 --mixed-precision

# For ViT-Base (86M params), MUST use batch-size 8 or less
python train.py --dataset cifar10 --stage teacher --model vit_base --batch-size 8
```

**Batch size recommendations (6GB VRAM):**
- ViT-Base: 8 (maximum)
- ViT-Small: 16
- ViT-Tiny: 32
- ResNet-50: 32
- ResNet-18: 64

---

### 📦 Dataset Download Fails

**Problem:**
```
ConnectionError: Failed to download dataset
```

**Solution:**

```bash
# Check internet connection
ping google.com

# Try again (automatic resume)
python dataset_prepare.py --all

# Force re-download
python dataset_prepare.py --all --force

# If firewall issues, check ports 80 and 443
```

---

### 📁 Permission Denied

**Problem:**
```
PermissionError: [Errno 13] Permission denied: 'datasets/'
```

**Solution:**

```bash
# Fix permissions
chmod -R u+w datasets/

# Or use different directory
python dataset_prepare.py --all --root ~/my_datasets
```

---

### 🔍 Dataset Already Exists But Verification Fails

**Problem:**
```
ℹ️  pathmnist already exists
❌ Verification failed
```

**Solution:**

```bash
# Option 1: Re-download (recommended)
python dataset_prepare.py --medmnist pathmnist --force

# Option 2: Delete and re-download
rm -rf datasets/medmnist/pathmnist
python dataset_prepare.py --medmnist pathmnist

# Option 3: Just delete metadata
rm datasets/medmnist/pathmnist/metadata.json
python dataset_prepare.py --medmnist pathmnist
```

---

### 🐍 Import Errors

**Problem:**
```
ModuleNotFoundError: No module named 'medmnist'
```

**Solution:**

```bash
# Verify virtual environment is activated
which python
# Should show path to venv

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import medmnist; print(medmnist.__version__)"
```

---

### 📊 Training Accuracy Stuck at Low Value

**Problem:**
```
Epoch 50/100 - Accuracy: 15.2%
```

**Solution:**

```bash
# Check if using pretrained weights (recommended)
python train.py --dataset cifar10 --stage teacher --pretrained

# Increase epochs
python train.py --dataset cifar10 --stage teacher --epochs 300

# Adjust learning rate
python train.py --dataset cifar10 --stage teacher --lr 1e-4

# Verify dataset is correct
python dataset_prepare.py --verify
```

---

### 🔄 Git Conflicts After Pull

**Problem:**
```
error: Your local changes to the following files would be overwritten by merge
```

**Solution:**

```bash
# Stash local changes
git stash

# Pull updates
git pull origin main

# Re-apply your changes (optional)
git stash pop

# Or discard local changes
git reset --hard origin/main
```

---

### 📉 NaN Loss During Training

**Problem:**
```
ValueError: Invalid loss value: nan detected
```

**Solution:**

```bash
# Reduce learning rate
python train.py --dataset cifar10 --stage teacher --lr 1e-4

# Use gradient clipping (already enabled by default)
# Check your code has: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Reduce batch size
python train.py --dataset cifar10 --stage teacher --batch-size 16

# Verify dataset normalization
python dataset_prepare.py --verify
```

---

### 🖥️ No GPU Detected

**Problem:**
```
Using device: cpu
Warning: CUDA not available
```

**Solution:**

```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True

# If False, reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

### 📝 Log File Not Created

**Problem:**
```
No log file found at: cnn_distillation/.../logs/
```

**Solution:**

```bash
# Check if training started
ls -la cnn_distillation/imagenet_pretrained/cifar10/teacher/logs/

# If directory doesn't exist, training didn't start
# Check for errors in training output

# Try running with verbose output
python train.py --dataset cifar10 --stage teacher --verbose
```

---

## 🔍 Diagnostic Commands

### Check System Status

```bash
# GPU status
nvidia-smi

# Disk space
df -h

# Python environment
which python
pip list | grep -E "torch|medmnist|numpy"

# Dataset status
python dataset_prepare.py --verify

# Project validation
python validate_project.py
```

### Clean Everything and Start Fresh

```bash
# WARNING: This deletes all trained models and datasets!

# Clean training outputs
cd cnn_distillation && bash clean.sh && cd ..
cd vit_distillation && bash clean.sh && cd ..

# Delete datasets (will need to re-download)
rm -rf datasets/

# Pull latest code
git pull origin main

# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Prepare datasets
python dataset_prepare.py --all

# Verify
python validate_project.py
```

---

## 📞 Getting Help

If you're still stuck:

1. **Check the logs:**
   - Training logs: `*/imagenet_pretrained/cifar10/*/logs/*.csv`
   - Error messages in terminal output

2. **Run validation:**
   ```bash
   python validate_project.py
   ```

3. **Check documentation:**
   - [DATASET_SETUP.md](DATASET_SETUP.md)
   - [FINAL_USAGE_GUIDE.md](FINAL_USAGE_GUIDE.md)
   - [CRITICAL_FIXES.md](CRITICAL_FIXES.md)

4. **Open a GitHub issue:**
   - Include error message
   - Include system info (GPU, Python version, PyTorch version)
   - Include steps to reproduce

---

## 🎯 Quick Reference

### Most Common Issues

| Issue | Quick Fix |
|-------|----------|
| MedMNIST verification fails | `git pull origin main` |
| CUDA OOM (ViT) | `--batch-size 8` |
| Dataset download fails | `--force` flag |
| Import errors | `pip install -r requirements.txt` |
| Low accuracy | `--pretrained` flag |
| NaN loss | Reduce `--lr` |
| No GPU | Check `nvidia-smi` |
| pynvml warning | `pip install nvidia-ml-py` |

---

**Last Updated:** December 30, 2025
