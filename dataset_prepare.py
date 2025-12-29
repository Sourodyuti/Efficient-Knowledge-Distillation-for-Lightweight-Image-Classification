#!/usr/bin/env python3
"""
Automatic Dataset Preparation Script

Automatically downloads and preprocesses datasets for knowledge distillation.
Run this ONCE before training any models.

Features:
- Automatic download of CIFAR-10
- Automatic download of MedMNIST datasets
- Data validation and verification
- Progress bars and status updates
- Disk space checking
- Error handling and retry logic

Usage:
    # Prepare all datasets (recommended)
    python dataset_prepare.py --all
    
    # Prepare only CIFAR-10
    python dataset_prepare.py --cifar10
    
    # Prepare specific MedMNIST dataset
    python dataset_prepare.py --medmnist pathmnist
    
    # Prepare multiple MedMNIST datasets
    python dataset_prepare.py --medmnist pathmnist chestmnist dermamnist
    
    # Verify all prepared datasets
    python dataset_prepare.py --verify
    
    # Custom root directory
    python dataset_prepare.py --all --root /path/to/datasets
"""

import argparse
import sys
import shutil
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.data.cifar10_loader import CIFAR10Processor
    from src.data.medmnist_loader import MedMNISTProcessor
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("Please ensure you're running from the project root directory.")
    sys.exit(1)


# Available MedMNIST datasets
AVAILABLE_MEDMNIST = [
    'pathmnist', 'chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist',
    'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist',
    'organamnist', 'organcmnist', 'organsmnist'
]


def check_disk_space(path: str, required_gb: float = 2.0) -> bool:
    """
    Check if sufficient disk space is available.
    
    Args:
        path (str): Path to check
        required_gb (float): Required space in GB
    
    Returns:
        bool: True if sufficient space available
    """
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024**3)
        
        print(f"\n💾 Disk space check:")
        print(f"   Available: {free_gb:.2f} GB")
        print(f"   Required: {required_gb:.2f} GB")
        
        if free_gb < required_gb:
            print(f"\n⚠️  WARNING: Low disk space!")
            print(f"   Please free up at least {required_gb - free_gb:.2f} GB")
            return False
        
        print(f"   ✓ Sufficient space available")
        return True
        
    except Exception as e:
        print(f"\n⚠️  Could not check disk space: {e}")
        return True  # Proceed anyway


def prepare_cifar10(root_dir: str = './datasets', force: bool = False) -> bool:
    """
    Automatically download and prepare CIFAR-10 dataset.
    
    Args:
        root_dir (str): Root directory for datasets
        force (bool): Force re-download even if exists
    
    Returns:
        bool: True if successful
    """
    try:
        print("\n" + "="*70)
        print("📦 PREPARING CIFAR-10 DATASET")
        print("="*70)
        
        # Check disk space (~170 MB needed)
        if not check_disk_space(root_dir, required_gb=0.5):
            return False
        
        print("\n[1/3] Initializing CIFAR-10 processor...")
        processor = CIFAR10Processor(root_dir=root_dir)
        
        # Check if already exists
        cifar_path = Path(root_dir) / 'cifar-10-batches-py'
        if cifar_path.exists() and not force:
            print(f"\nℹ️  CIFAR-10 already exists at {cifar_path}")
            print("   Skipping download. Use --force to re-download.")
        else:
            print("\n[2/3] Downloading CIFAR-10...")
            print("   Source: https://www.cs.toronto.edu/~kriz/cifar.html")
            print("   Size: ~170 MB")
            print("   This may take a few minutes...")
            
            processor.download_and_prepare()
            print("\n   ✓ Download complete!")
        
        print("\n[3/3] Verifying CIFAR-10...")
        if not processor.verify_dataset():
            print("\n❌ CIFAR-10 verification failed!")
            return False
        
        # Print dataset info
        print("\n📊 CIFAR-10 Dataset Info:")
        print(f"   ✓ Training images: 50,000 (32x32x3)")
        print(f"   ✓ Test images: 10,000 (32x32x3)")
        print(f"   ✓ Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)")
        print(f"   ✓ Location: {cifar_path.absolute()}")
        
        print("\n✅ CIFAR-10 ready for training!")
        return True
        
    except Exception as e:
        print(f"\n❌ CIFAR-10 preparation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def prepare_medmnist(
    dataset_names: List[str],
    root_dir: str = './datasets',
    force: bool = False
) -> Dict[str, bool]:
    """
    Automatically download and prepare MedMNIST datasets.
    
    Args:
        dataset_names (List[str]): List of MedMNIST dataset names
        root_dir (str): Root directory for datasets
        force (bool): Force re-download even if exists
    
    Returns:
        Dict[str, bool]: Results for each dataset
    """
    results = {}
    
    for i, dataset_name in enumerate(dataset_names, 1):
        try:
            print("\n" + "="*70)
            print(f"📦 PREPARING MEDMNIST: {dataset_name.upper()} ({i}/{len(dataset_names)})")
            print("="*70)
            
            # Validate dataset name
            if dataset_name.lower() not in AVAILABLE_MEDMNIST:
                print(f"\n❌ Unknown MedMNIST dataset: {dataset_name}")
                print(f"   Available datasets: {', '.join(AVAILABLE_MEDMNIST)}")
                results[dataset_name] = False
                continue
            
            # Check disk space (~50-500 MB per dataset)
            if not check_disk_space(root_dir, required_gb=1.0):
                results[dataset_name] = False
                continue
            
            print(f"\n[1/3] Initializing {dataset_name} processor...")
            processor = MedMNISTProcessor(
                dataset_name=dataset_name,
                root_dir=root_dir
            )
            
            # Check if already exists
            medmnist_path = Path(root_dir) / 'medmnist' / dataset_name
            if medmnist_path.exists() and not force:
                print(f"\nℹ️  {dataset_name} already exists at {medmnist_path}")
                print("   Skipping download. Use --force to re-download.")
            else:
                print(f"\n[2/3] Downloading {dataset_name}...")
                print("   Source: https://medmnist.com/")
                print("   This may take a few minutes...")
                
                processor.download_and_prepare()
                print("\n   ✓ Download complete!")
            
            print(f"\n[3/3] Verifying {dataset_name}...")
            if not processor.verify_dataset():
                print(f"\n❌ {dataset_name} verification failed!")
                results[dataset_name] = False
                continue
            
            # Get dataset info
            loaders = processor.get_dataloaders(batch_size=32)
            num_classes = loaders['num_classes']
            
            print(f"\n📊 {dataset_name.upper()} Dataset Info:")
            print(f"   ✓ Classes: {num_classes}")
            print(f"   ✓ Location: {medmnist_path.absolute()}")
            
            print(f"\n✅ {dataset_name.upper()} ready for training!")
            results[dataset_name] = True
            
        except Exception as e:
            print(f"\n❌ {dataset_name} preparation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results[dataset_name] = False
    
    return results


def verify_all_datasets(root_dir: str = './datasets') -> bool:
    """
    Verify all prepared datasets.
    
    Args:
        root_dir (str): Root directory for datasets
    
    Returns:
        bool: True if all datasets verified successfully
    """
    print("\n" + "="*70)
    print("🔍 VERIFYING ALL DATASETS")
    print("="*70)
    
    results = {}
    
    # Check CIFAR-10
    print("\n[1/2] Checking CIFAR-10...")
    try:
        processor = CIFAR10Processor(root_dir=root_dir)
        results['CIFAR-10'] = processor.verify_dataset()
        if results['CIFAR-10']:
            print("   ✓ CIFAR-10 verified")
        else:
            print("   ❌ CIFAR-10 verification failed")
    except Exception as e:
        print(f"   ❌ CIFAR-10 check failed: {str(e)}")
        results['CIFAR-10'] = False
    
    # Check MedMNIST datasets
    print("\n[2/2] Checking MedMNIST datasets...")
    medmnist_dir = Path(root_dir) / 'medmnist'
    
    if medmnist_dir.exists():
        medmnist_datasets = sorted([d.name for d in medmnist_dir.iterdir() if d.is_dir()])
        
        if not medmnist_datasets:
            print("   ℹ️  No MedMNIST datasets found.")
        else:
            for dataset_name in medmnist_datasets:
                try:
                    processor = MedMNISTProcessor(
                        dataset_name=dataset_name,
                        root_dir=root_dir
                    )
                    results[dataset_name] = processor.verify_dataset()
                    if results[dataset_name]:
                        print(f"   ✓ {dataset_name} verified")
                    else:
                        print(f"   ❌ {dataset_name} verification failed")
                except Exception as e:
                    print(f"   ❌ {dataset_name} check failed: {str(e)}")
                    results[dataset_name] = False
    else:
        print("   ℹ️  No MedMNIST datasets found.")
    
    # Summary
    print("\n" + "="*70)
    print("📊 VERIFICATION SUMMARY")
    print("="*70)
    
    if not results:
        print("\nℹ️  No datasets found. Run with --all or --cifar10 to prepare datasets.")
        return False
    
    for dataset, status in sorted(results.items()):
        symbol = "✅" if status else "❌"
        status_str = "READY" if status else "FAILED"
        print(f"  {symbol} {dataset:20s} {status_str}")
    
    passed = sum(results.values())
    total = len(results)
    
    print("\n" + "="*70)
    
    if passed == total:
        print(f"✅ All {total} dataset(s) verified successfully!")
        return True
    elif passed > 0:
        print(f"⚠️  {passed}/{total} dataset(s) verified. {total - passed} failed.")
        return False
    else:
        print(f"❌ All {total} dataset(s) failed verification.")
        return False


def print_dataset_info():
    """Print information about available datasets."""
    print("\n" + "="*70)
    print("📚 AVAILABLE DATASETS")
    print("="*70)
    
    print("\n🖼️  CIFAR-10:")
    print("   - Natural images dataset")
    print("   - 60,000 images (50K train, 10K test)")
    print("   - 10 classes (airplane, automobile, bird, cat, etc.)")
    print("   - Image size: 32x32x3")
    print("   - Download size: ~170 MB")
    
    print("\n🎨 MedMNIST Datasets:")
    print("   - Medical imaging datasets")
    print("   - Various sizes and classes per dataset")
    print("   - Standardized to 28x28 or 64x64")
    print("   - Download size: ~50-500 MB per dataset")
    
    print("\n   Available MedMNIST datasets:")
    for i, dataset in enumerate(AVAILABLE_MEDMNIST, 1):
        print(f"      {i:2d}. {dataset}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Automatically download and prepare datasets for knowledge distillation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare all datasets (CIFAR-10 + PathMNIST) - RECOMMENDED
  python dataset_prepare.py --all
  
  # Prepare only CIFAR-10
  python dataset_prepare.py --cifar10
  
  # Prepare specific MedMNIST dataset
  python dataset_prepare.py --medmnist pathmnist
  
  # Prepare multiple MedMNIST datasets
  python dataset_prepare.py --medmnist pathmnist chestmnist dermamnist
  
  # Force re-download (even if exists)
  python dataset_prepare.py --all --force
  
  # Verify all prepared datasets
  python dataset_prepare.py --verify
  
  # Custom root directory
  python dataset_prepare.py --all --root /path/to/datasets
  
  # List available datasets
  python dataset_prepare.py --list

Notes:
  - Downloads are automatic from official sources
  - CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
  - MedMNIST: https://medmnist.com/
  - Total download size: ~200 MB - 2 GB (depending on datasets selected)
  - Internet connection required for first-time setup
  - Datasets are stored in ./datasets/ by default
        """
    )
    
    parser.add_argument('--all', action='store_true',
                        help='Prepare all datasets (CIFAR-10 + PathMNIST)')
    parser.add_argument('--cifar10', action='store_true',
                        help='Prepare CIFAR-10 dataset')
    parser.add_argument('--medmnist', nargs='+', metavar='DATASET',
                        help='Prepare MedMNIST dataset(s) (e.g., pathmnist chestmnist)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify all prepared datasets')
    parser.add_argument('--list', action='store_true',
                        help='List available datasets')
    parser.add_argument('--root', type=str, default='./datasets',
                        help='Root directory for datasets (default: ./datasets)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download even if dataset exists')
    
    args = parser.parse_args()
    
    # List datasets
    if args.list:
        print_dataset_info()
        sys.exit(0)
    
    # Check if any action specified
    if not any([args.all, args.cifar10, args.medmnist, args.verify]):
        parser.print_help()
        print("\nℹ️  Hint: Use --list to see available datasets")
        print("        Use --all to prepare CIFAR-10 + PathMNIST (recommended)")
        sys.exit(1)
    
    root_dir = args.root
    
    # Create root directory
    Path(root_dir).mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Dataset directory: {Path(root_dir).absolute()}")
    
    # Verify only
    if args.verify:
        success = verify_all_datasets(root_dir)
        sys.exit(0 if success else 1)
    
    # Track success
    overall_success = True
    
    # Prepare datasets
    if args.all or args.cifar10:
        if not prepare_cifar10(root_dir, force=args.force):
            overall_success = False
    
    if args.all:
        # Default to pathmnist for --all
        results = prepare_medmnist(['pathmnist'], root_dir, force=args.force)
        if not all(results.values()):
            overall_success = False
    
    if args.medmnist:
        results = prepare_medmnist(args.medmnist, root_dir, force=args.force)
        if not all(results.values()):
            overall_success = False
    
    # Final verification
    print("\n" + "="*70)
    print("🔍 RUNNING FINAL VERIFICATION")
    print("="*70)
    verify_success = verify_all_datasets(root_dir)
    
    if overall_success and verify_success:
        print("\n" + "="*70)
        print("✅ DATASET PREPARATION COMPLETE!")
        print("="*70)
        print(f"\n📁 Datasets location: {Path(root_dir).absolute()}")
        print("\n🚀 Next steps:")
        print("   1. Datasets are ready for training")
        print("   2. Run training scripts in cnn_distillation/ or vit_distillation/")
        print("   3. Example: cd cnn_distillation && python train.py --dataset cifar10 --stage teacher")
        print("\n⚠️  Note: Datasets will NOT be modified during training (read-only access)")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌ DATASET PREPARATION INCOMPLETE")
        print("="*70)
        print("\n⚠️  Some datasets failed preparation or verification.")
        print("   Please check errors above and try again.")
        print("   You can re-run this script with --force to re-download.")
        print("="*70)
        sys.exit(1)


if __name__ == '__main__':
    main()
