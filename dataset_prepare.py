#!/usr/bin/env python3
"""
Dataset Preparation Script

Prepares all datasets for knowledge distillation experiments.
Run this ONCE before training any models.

Usage:
    python dataset_prepare.py --all
    python dataset_prepare.py --cifar10
    python dataset_prepare.py --medmnist pathmnist
    python dataset_prepare.py --verify
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.cifar10_loader import CIFAR10Processor
from src.data.medmnist_loader import MedMNISTProcessor


def prepare_cifar10(root_dir='./datasets'):
    """
    Prepare CIFAR-10 dataset.
    
    Args:
        root_dir (str): Root directory for datasets
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print("\n" + "="*70)
        print("PREPARING CIFAR-10")
        print("="*70)
        
        processor = CIFAR10Processor(root_dir=root_dir)
        processor.download_and_prepare()
        
        # Verify
        if not processor.verify_dataset():
            print("❌ CIFAR-10 verification failed!")
            return False
        
        print("\n✓ CIFAR-10 ready for training!")
        return True
        
    except Exception as e:
        print(f"\n❌ CIFAR-10 preparation failed: {str(e)}")
        return False


def prepare_medmnist(dataset_name='pathmnist', root_dir='./datasets'):
    """
    Prepare MedMNIST dataset.
    
    Args:
        dataset_name (str): Name of MedMNIST dataset
        root_dir (str): Root directory for datasets
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print("\n" + "="*70)
        print(f"PREPARING MEDMNIST: {dataset_name.upper()}")
        print("="*70)
        
        processor = MedMNISTProcessor(dataset_name=dataset_name, root_dir=root_dir)
        processor.download_and_prepare()
        
        # Verify
        if not processor.verify_dataset():
            print(f"❌ {dataset_name.upper()} verification failed!")
            return False
        
        print(f"\n✓ {dataset_name.upper()} ready for training!")
        return True
        
    except Exception as e:
        print(f"\n❌ {dataset_name.upper()} preparation failed: {str(e)}")
        return False


def verify_all_datasets(root_dir='./datasets'):
    """
    Verify all prepared datasets.
    
    Args:
        root_dir (str): Root directory for datasets
    """
    print("\n" + "="*70)
    print("VERIFYING ALL DATASETS")
    print("="*70)
    
    results = {}
    
    # Check CIFAR-10
    print("\n[1/2] Checking CIFAR-10...")
    try:
        processor = CIFAR10Processor(root_dir=root_dir)
        results['cifar10'] = processor.verify_dataset()
    except Exception as e:
        print(f"❌ CIFAR-10 check failed: {str(e)}")
        results['cifar10'] = False
    
    # Check MedMNIST datasets
    print("\n[2/2] Checking MedMNIST datasets...")
    medmnist_dir = Path(root_dir) / 'medmnist'
    
    if medmnist_dir.exists():
        medmnist_datasets = [d.name for d in medmnist_dir.iterdir() if d.is_dir()]
        
        for dataset_name in medmnist_datasets:
            print(f"\n  Checking {dataset_name}...")
            try:
                processor = MedMNISTProcessor(dataset_name=dataset_name, root_dir=root_dir)
                results[dataset_name] = processor.verify_dataset()
            except Exception as e:
                print(f"  ❌ {dataset_name} check failed: {str(e)}")
                results[dataset_name] = False
    else:
        print("  No MedMNIST datasets found.")
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    for dataset, status in results.items():
        symbol = "✓" if status else "❌"
        print(f"  {symbol} {dataset}: {'READY' if status else 'FAILED'}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All datasets verified successfully!")
    else:
        print("\n❌ Some datasets failed verification. Please re-run preparation.")
    
    print("="*70)
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description='Prepare datasets for knowledge distillation experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare all datasets (CIFAR-10 + PathMNIST)
  python dataset_prepare.py --all
  
  # Prepare only CIFAR-10
  python dataset_prepare.py --cifar10
  
  # Prepare specific MedMNIST dataset
  python dataset_prepare.py --medmnist pathmnist
  python dataset_prepare.py --medmnist chestmnist
  
  # Verify all prepared datasets
  python dataset_prepare.py --verify
  
  # Custom root directory
  python dataset_prepare.py --all --root /path/to/datasets

Available MedMNIST datasets:
  pathmnist, chestmnist, dermamnist, octmnist, pneumoniamnist,
  retinamnist, breastmnist, bloodmnist, tissuemnist,
  organamnist, organcmnist, organsmnist
        """
    )
    
    parser.add_argument('--all', action='store_true',
                        help='Prepare all datasets (CIFAR-10 + PathMNIST)')
    parser.add_argument('--cifar10', action='store_true',
                        help='Prepare CIFAR-10 dataset')
    parser.add_argument('--medmnist', type=str, metavar='DATASET',
                        help='Prepare specific MedMNIST dataset (e.g., pathmnist)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify all prepared datasets')
    parser.add_argument('--root', type=str, default='./datasets',
                        help='Root directory for datasets (default: ./datasets)')
    
    args = parser.parse_args()
    
    # Check if any action specified
    if not any([args.all, args.cifar10, args.medmnist, args.verify]):
        parser.print_help()
        sys.exit(1)
    
    root_dir = args.root
    
    # Create root directory
    Path(root_dir).mkdir(parents=True, exist_ok=True)
    
    success = True
    
    # Verify only
    if args.verify:
        success = verify_all_datasets(root_dir)
        sys.exit(0 if success else 1)
    
    # Prepare datasets
    if args.all or args.cifar10:
        if not prepare_cifar10(root_dir):
            success = False
    
    if args.all:
        # Default to pathmnist for --all
        if not prepare_medmnist('pathmnist', root_dir):
            success = False
    
    if args.medmnist:
        if not prepare_medmnist(args.medmnist, root_dir):
            success = False
    
    # Final verification
    print("\n" + "="*70)
    print("RUNNING FINAL VERIFICATION")
    print("="*70)
    verify_all_datasets(root_dir)
    
    if success:
        print("\n" + "="*70)
        print("✓ DATASET PREPARATION COMPLETE!")
        print("="*70)
        print(f"\nDatasets location: {Path(root_dir).absolute()}")
        print("\nYou can now run training experiments.")
        print("Datasets will NOT be modified during training.")
        print("="*70)
    else:
        print("\n❌ Some datasets failed preparation. Please check errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
