# 📦 Dataset Setup Guide

**Automatic download and preprocessing for all datasets**

This guide explains how to automatically download and prepare datasets for training. The `dataset_prepare.py` script handles everything automatically.

---

## ⚡ Quick Start (Recommended)

```bash
# Prepare CIFAR-10 + PathMNIST (most common setup)
python dataset_prepare.py --all
```

That's it! The script will:
- ✅ Automatically download datasets from official sources
- ✅ Verify data integrity
- ✅ Prepare datasets for training
- ✅ Display progress and status

**Total download size:** ~200-400 MB  
**Time:** ~5-10 minutes (depending on internet speed)

---

## 📚 Available Datasets

### CIFAR-10
**Natural images classification**
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Images:** 60,000 (50K train, 10K test)
- **Resolution:** 32x32x3 (RGB)
- **Size:** ~170 MB
- **Source:** [University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html)

### MedMNIST
**Medical imaging datasets**
- **12 different datasets** available
- **Classes:** Varies (2-11 classes per dataset)
- **Resolution:** 28x28 or 64x64
- **Size:** ~50-500 MB per dataset
- **Source:** [MedMNIST.com](https://medmnist.com/)

**Available MedMNIST datasets:**
1. `pathmnist` - Colon pathology images (9 classes)
2. `chestmnist` - Chest X-ray images (14 classes)
3. `dermamnist` - Dermatoscopic images (7 classes)
4. `octmnist` - Retinal OCT images (4 classes)
5. `pneumoniamnist` - Chest X-ray pneumonia (2 classes)
6. `retinamnist` - Fundus camera images (5 classes)
7. `breastmnist` - Breast ultrasound (2 classes)
8. `bloodmnist` - Blood cell images (8 classes)
9. `tissuemnist` - Kidney cortex images (8 classes)
10. `organamnist` - Abdominal CT (11 classes)
11. `organcmnist` - Abdominal CT coronal (11 classes)
12. `organsmnist` - Abdominal CT sagittal (11 classes)

---

## 🛠️ Usage

### Basic Commands

```bash
# List available datasets
python dataset_prepare.py --list

# Prepare CIFAR-10 only
python dataset_prepare.py --cifar10

# Prepare specific MedMNIST dataset
python dataset_prepare.py --medmnist pathmnist

# Prepare multiple MedMNIST datasets
python dataset_prepare.py --medmnist pathmnist chestmnist dermamnist

# Prepare all (CIFAR-10 + PathMNIST)
python dataset_prepare.py --all

# Verify prepared datasets
python dataset_prepare.py --verify
```

### Advanced Options

```bash
# Force re-download (even if exists)
python dataset_prepare.py --all --force

# Custom root directory
python dataset_prepare.py --all --root /path/to/datasets

# Prepare everything for comprehensive experiments
python dataset_prepare.py --cifar10 --medmnist pathmnist chestmnist
```

---

## 💾 System Requirements

### Disk Space

| Dataset | Download Size | Extracted Size |
|---------|--------------|----------------|
| CIFAR-10 | ~170 MB | ~170 MB |
| PathMNIST | ~50 MB | ~50 MB |
| ChestMNIST | ~200 MB | ~200 MB |
| DermaMNIST | ~100 MB | ~100 MB |
| Other MedMNIST | ~50-500 MB | ~50-500 MB |

**Recommended:** At least 2 GB free disk space

### Internet Connection
- Required for first-time download
- Automatic retry on connection failures
- Resume support for interrupted downloads

### Python Dependencies
- Installed via `pip install -r requirements.txt`
- Core: `torch`, `torchvision`, `medmnist`, `Pillow`

---

## 🔄 What the Script Does

### 1. Disk Space Check
```
💾 Disk space check:
   Available: 45.23 GB
   Required: 2.00 GB
   ✓ Sufficient space available
```

### 2. Automatic Download
```
[1/3] Initializing CIFAR-10 processor...
[2/3] Downloading CIFAR-10...
   Source: https://www.cs.toronto.edu/~kriz/cifar.html
   Size: ~170 MB
   This may take a few minutes...
   ✓ Download complete!
```

### 3. Verification
```
[3/3] Verifying CIFAR-10...
📊 CIFAR-10 Dataset Info:
   ✓ Training images: 50,000 (32x32x3)
   ✓ Test images: 10,000 (32x32x3)
   ✓ Classes: 10
   ✓ Location: /path/to/datasets/cifar-10-batches-py

✅ CIFAR-10 ready for training!
```

### 4. Final Summary
```
🔍 RUNNING FINAL VERIFICATION

📊 VERIFICATION SUMMARY
  ✅ CIFAR-10           READY
  ✅ pathmnist          READY

✅ All 2 dataset(s) verified successfully!

🚀 Next steps:
   1. Datasets are ready for training
   2. Run training scripts in cnn_distillation/ or vit_distillation/
   3. Example: cd cnn_distillation && python train.py --dataset cifar10 --stage teacher
```

---

## 📁 Directory Structure

After running the preparation script:

```
project/
├── datasets/                      # Root dataset directory
│   ├── cifar-10-batches-py/       # CIFAR-10 data
│   │   ├── data_batch_1
│   │   ├── data_batch_2
│   │   ├── data_batch_3
│   │   ├── data_batch_4
│   │   ├── data_batch_5
│   │   ├── test_batch
│   │   └── batches.meta
│   │
│   └── medmnist/                 # MedMNIST datasets
│       ├── pathmnist/
│       │   ├── train_images.npy
│       │   ├── train_labels.npy
│       │   ├── val_images.npy
│       │   ├── val_labels.npy
│       │   ├── test_images.npy
│       │   └── test_labels.npy
│       │
│       ├── chestmnist/
│       └── ... (other MedMNIST datasets)
│
├── dataset_prepare.py          # This preparation script
└── ...
```

---

## ✅ Verification

### Verify Datasets Anytime

```bash
python dataset_prepare.py --verify
```

**Output:**
```
🔍 VERIFYING ALL DATASETS

[1/2] Checking CIFAR-10...
   ✓ CIFAR-10 verified

[2/2] Checking MedMNIST datasets...
   ✓ pathmnist verified
   ✓ chestmnist verified

📊 VERIFICATION SUMMARY
  ✅ CIFAR-10           READY
  ✅ pathmnist          READY
  ✅ chestmnist         READY

✅ All 3 dataset(s) verified successfully!
```

---

## 🔧 Troubleshooting

### Problem: Download fails

**Symptom:**
```
❌ CIFAR-10 preparation failed: Connection timeout
```

**Solutions:**
1. Check internet connection
2. Try again (script will resume)
3. Use `--force` to restart download
4. Check firewall settings

### Problem: Insufficient disk space

**Symptom:**
```
⚠️  WARNING: Low disk space!
   Please free up at least 1.23 GB
```

**Solutions:**
1. Free up disk space
2. Use `--root` to specify different location
3. Prepare fewer datasets at once

### Problem: Dataset already exists

**Symptom:**
```
ℹ️  CIFAR-10 already exists
   Skipping download. Use --force to re-download.
```

**Solutions:**
1. This is normal! Dataset is already prepared
2. Use `--force` if you want to re-download
3. Use `--verify` to check integrity

### Problem: Verification fails

**Symptom:**
```
❌ CIFAR-10 verification failed!
```

**Solutions:**
1. Re-download with `--force`:
   ```bash
   python dataset_prepare.py --cifar10 --force
   ```
2. Check disk space (corruption may occur if disk full)
3. Delete partial downloads and try again

### Problem: Import errors

**Symptom:**
```
❌ Import error: No module named 'medmnist'
```

**Solutions:**
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Verify Python version (3.8+ required)
3. Check virtual environment activated

---

## 🔒 Dataset Immutability

**Important:** Datasets are **read-only** during training.

- Training scripts **never modify** dataset files
- Safe to share datasets across multiple experiments
- No need to re-download for each experiment
- Validation script checks this (see `validate_project.py`)

---

## 🚀 Next Steps

After preparing datasets:

### 1. Verify Setup
```bash
python dataset_prepare.py --verify
```

### 2. Validate Project
```bash
python validate_project.py
```

### 3. Start Training

**CNN Pipeline:**
```bash
cd cnn_distillation
python train.py --dataset cifar10 --stage teacher --pretrained --epochs 100
```

**ViT Pipeline:**
```bash
cd vit_distillation
python train.py --dataset cifar10 --stage teacher --pretrained --epochs 100
```

See [FINAL_USAGE_GUIDE.md](FINAL_USAGE_GUIDE.md) for complete training instructions.

---

## 📊 Dataset Statistics

### CIFAR-10 Details

| Split | Images | Classes | Size per Image | Total Size |
|-------|--------|---------|----------------|------------|
| Train | 50,000 | 10 | 32x32x3 | ~150 MB |
| Test | 10,000 | 10 | 32x32x3 | ~30 MB |

**Class Distribution:** 5,000 images per class (balanced)

### MedMNIST Details

| Dataset | Classes | Train | Val | Test | Size |
|---------|---------|-------|-----|------|------|
| PathMNIST | 9 | 89,996 | 10,004 | 7,180 | 28x28 |
| ChestMNIST | 14 | 78,468 | 11,219 | 22,433 | 28x28 |
| DermaMNIST | 7 | 7,007 | 1,003 | 2,005 | 28x28 |
| OctMNIST | 4 | 97,477 | 10,832 | 1,000 | 28x28 |

*See [MedMNIST.com](https://medmnist.com/) for complete statistics*

---

## 🔗 Official Sources

### CIFAR-10
- **Website:** https://www.cs.toronto.edu/~kriz/cifar.html
- **Paper:** Learning Multiple Layers of Features from Tiny Images (Krizhevsky, 2009)
- **License:** Open source

### MedMNIST
- **Website:** https://medmnist.com/
- **GitHub:** https://github.com/MedMNIST/MedMNIST
- **Paper:** MedMNIST v2 (Yang et al., 2023)
- **License:** CC BY 4.0

---

## ❓ FAQ

**Q: Do I need to re-download for each experiment?**  
A: No! Download once, use forever. Datasets are read-only.

**Q: Can I use my own dataset directory?**  
A: Yes! Use `--root /path/to/datasets`

**Q: How long does download take?**  
A: 5-10 minutes for CIFAR-10 + PathMNIST on average internet.

**Q: Can I prepare multiple datasets at once?**  
A: Yes! Use `--medmnist pathmnist chestmnist dermamnist`

**Q: What if download is interrupted?**  
A: Re-run the script. Most downloads will resume automatically.

**Q: Can I delete datasets later?**  
A: Yes! Just delete the `datasets/` folder. Run preparation script again when needed.

**Q: Are datasets modified during training?**  
A: No! Training only reads datasets (immutable).

---

## 📝 Summary

**Automatic dataset preparation with one command:**

```bash
python dataset_prepare.py --all
```

**Features:**
- ✅ Automatic download from official sources
- ✅ Integrity verification
- ✅ Progress tracking
- ✅ Disk space checking
- ✅ Error handling and retry
- ✅ Multiple dataset support
- ✅ Resume capability

**Total time:** ~5-10 minutes  
**Total size:** ~200-400 MB (CIFAR-10 + PathMNIST)

---

**Ready to train!** See [FINAL_USAGE_GUIDE.md](FINAL_USAGE_GUIDE.md) for next steps.
