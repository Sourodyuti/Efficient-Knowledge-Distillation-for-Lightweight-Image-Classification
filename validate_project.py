#!/usr/bin/env python3
"""
Project Validation Script

Validates:
- Folder isolation (CNN and ViT experiments are separate)
- Dataset immutability (datasets are not modified)
- Log consistency (all required columns present)
- Code structure and imports
"""

import sys
from pathlib import Path
import pandas as pd
from typing import List, Tuple


class ProjectValidator:
    """
    Validates project structure and integrity.
    """
    
    def __init__(self, project_root: str = '.'):
        self.project_root = Path(project_root)
        self.errors = []
        self.warnings = []
        self.checks_passed = 0
        self.checks_total = 0
    
    def check(self, condition: bool, message: str, is_warning: bool = False):
        """Record a validation check."""
        self.checks_total += 1
        if condition:
            self.checks_passed += 1
            print(f"✓ {message}")
        else:
            if is_warning:
                self.warnings.append(message)
                print(f"⚠ WARNING: {message}")
            else:
                self.errors.append(message)
                print(f"❌ ERROR: {message}")
    
    def validate_folder_isolation(self) -> bool:
        """
        Validate that CNN and ViT experiments are isolated.
        """
        print("\n[1/5] Validating Folder Isolation...")
        print("-" * 60)
        
        # Check CNN folder exists
        cnn_dir = self.project_root / 'cnn_distillation'
        self.check(
            cnn_dir.exists(),
            "CNN distillation directory exists"
        )
        
        # Check ViT folder exists
        vit_dir = self.project_root / 'vit_distillation'
        self.check(
            vit_dir.exists(),
            "ViT distillation directory exists"
        )
        
        # Check folders are separate (no overlap)
        if cnn_dir.exists() and vit_dir.exists():
            # CNN shouldn't have ViT files
            cnn_files = set(p.name for p in cnn_dir.rglob('*') if p.is_file())
            vit_indicators = ['vit', 'attention', 'transformer']
            
            has_vit_in_cnn = any(
                any(indicator in f.lower() for indicator in vit_indicators)
                for f in cnn_files
            )
            
            self.check(
                not has_vit_in_cnn,
                "CNN folder doesn't contain ViT-specific files",
                is_warning=True
            )
        
        # Check separate log directories
        cnn_logs = cnn_dir / 'imagenet_pretrained' / 'cifar10' / 'teacher' / 'logs'
        vit_logs = vit_dir / 'imagenet_pretrained' / 'cifar10' / 'teacher' / 'logs'
        
        if cnn_logs.exists() and vit_logs.exists():
            self.check(
                cnn_logs != vit_logs,
                "CNN and ViT logs are in separate directories"
            )
        
        return len(self.errors) == 0
    
    def validate_dataset_immutability(self) -> bool:
        """
        Validate that datasets are not modified during experiments.
        """
        print("\n[2/5] Validating Dataset Immutability...")
        print("-" * 60)
        
        dataset_dir = self.project_root / 'datasets'
        
        # Check dataset directory exists
        self.check(
            dataset_dir.exists(),
            "Datasets directory exists"
        )
        
        if not dataset_dir.exists():
            return False
        
        # Check for expected dataset structure
        expected_datasets = ['cifar10', 'medmnist']
        for dataset in expected_datasets:
            dataset_path = dataset_dir / dataset
            self.check(
                dataset_path.exists(),
                f"{dataset} directory exists",
                is_warning=True
            )
        
        # Check datasets are read-only (no write logs in dataset dir)
        write_indicators = ['.log', '.tmp', '.lock']
        write_files = []
        for indicator in write_indicators:
            write_files.extend(dataset_dir.rglob(f'*{indicator}'))
        
        self.check(
            len(write_files) == 0,
            "No write logs found in dataset directory"
        )
        
        return len(self.errors) == 0
    
    def validate_log_consistency(self) -> bool:
        """
        Validate that all log files have consistent structure.
        """
        print("\n[3/5] Validating Log Consistency...")
        print("-" * 60)
        
        required_columns = [
            'epoch',
            'train_loss', 'train_accuracy', 'train_precision', 'train_recall', 'train_f1_score',
            'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1_score',
            'learning_rate', 'epoch_time_sec', 'gpu_memory_mb'
        ]
        
        # Find all log CSV files
        log_files = list(self.project_root.rglob('*_log.csv'))
        
        self.check(
            len(log_files) >= 0,
            f"Found {len(log_files)} log file(s)",
            is_warning=len(log_files) == 0
        )
        
        for log_file in log_files:
            try:
                df = pd.read_csv(log_file)
                
                # Check all required columns exist
                missing_cols = set(required_columns) - set(df.columns)
                
                self.check(
                    len(missing_cols) == 0,
                    f"{log_file.name}: All required columns present"
                )
                
                if missing_cols:
                    print(f"  Missing columns: {', '.join(missing_cols)}")
                
                # Check no NaN values in critical columns
                critical_cols = ['train_accuracy', 'val_accuracy']
                has_nan = df[critical_cols].isnull().any().any()
                
                self.check(
                    not has_nan,
                    f"{log_file.name}: No NaN values in critical columns"
                )
                
            except Exception as e:
                self.errors.append(f"Failed to validate {log_file.name}: {e}")
                print(f"❌ ERROR: Failed to validate {log_file.name}: {e}")
        
        return len(self.errors) == 0
    
    def validate_code_structure(self) -> bool:
        """
        Validate code structure and imports.
        """
        print("\n[4/5] Validating Code Structure...")
        print("-" * 60)
        
        # Check src directory structure
        required_modules = [
            'src/data',
            'src/models/cnn',
            'src/models/vit',
            'src/distillation',
            'src/utils',
            'src/visualization',
            'src/analysis'
        ]
        
        for module in required_modules:
            module_path = self.project_root / module
            self.check(
                module_path.exists(),
                f"{module} module exists"
            )
            
            # Check __init__.py exists
            init_file = module_path / '__init__.py'
            self.check(
                init_file.exists(),
                f"{module}/__init__.py exists"
            )
        
        # Check main training scripts exist
        training_scripts = [
            'cnn_distillation/train.py',
            'vit_distillation/train.py'
        ]
        
        for script in training_scripts:
            script_path = self.project_root / script
            self.check(
                script_path.exists(),
                f"{script} exists"
            )
        
        return len(self.errors) == 0
    
    def validate_documentation(self) -> bool:
        """
        Validate documentation files.
        """
        print("\n[5/5] Validating Documentation...")
        print("-" * 60)
        
        # Check README files
        readme_files = [
            'README.md',
            'cnn_distillation/README.md',
            'vit_distillation/README.md'
        ]
        
        for readme in readme_files:
            readme_path = self.project_root / readme
            self.check(
                readme_path.exists(),
                f"{readme} exists"
            )
        
        # Check phase summaries
        phase_summaries = [
            'PHASE3_SUMMARY.md',
            'PHASE4_SUMMARY.md'
        ]
        
        for summary in phase_summaries:
            summary_path = self.project_root / summary
            self.check(
                summary_path.exists(),
                f"{summary} exists"
            )
        
        return len(self.errors) == 0
    
    def run_all_validations(self) -> bool:
        """
        Run all validation checks.
        
        Returns:
            bool: True if all validations pass
        """
        print("\n" + "="*60)
        print("PROJECT VALIDATION")
        print("="*60)
        
        self.validate_folder_isolation()
        self.validate_dataset_immutability()
        self.validate_log_consistency()
        self.validate_code_structure()
        self.validate_documentation()
        
        # Print summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Checks passed: {self.checks_passed}/{self.checks_total}")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        
        if self.errors:
            print("\nERRORS:")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print("\nWARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        success = len(self.errors) == 0
        
        if success:
            print("\n✓ All validations passed!")
        else:
            print("\n❌ Some validations failed!")
        
        print("="*60 + "\n")
        
        return success


def main():
    """Main validation function."""
    validator = ProjectValidator()
    success = validator.run_all_validations()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
