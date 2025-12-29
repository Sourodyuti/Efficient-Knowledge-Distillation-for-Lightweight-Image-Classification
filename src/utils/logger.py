"""
Logging Utilities for Knowledge Distillation

Provides:
- CSVLogger: Per-epoch CSV logging
- EpochLogger: Comprehensive epoch-level logging
- Automatic log file management

Logs include: accuracy, precision, recall, F1, loss, learning rate,
epoch time, GPU memory usage.
"""

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime


class CSVLogger:
    """
    CSV Logger for per-epoch metrics.
    
    Automatically creates and manages CSV log files.
    Thread-safe for single-process training.
    """
    
    def __init__(
        self,
        log_dir: str,
        filename: str = 'training_log.csv',
        resume: bool = False
    ):
        """
        Args:
            log_dir (str): Directory to save log files
            filename (str): Name of CSV file
            resume (bool): If True, append to existing file. Otherwise, create new.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_path = self.log_dir / filename
        self.resume = resume
        
        self.fieldnames: Optional[List[str]] = None
        self.file_handle = None
        self.writer = None
        
        # Initialize file if not resuming
        if not resume and self.log_path.exists():
            # Backup existing file
            backup_path = self.log_dir / f"{filename}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.log_path.rename(backup_path)
    
    def _initialize_writer(self, data: Dict[str, Any]):
        """Initialize CSV writer with fieldnames from first data entry."""
        self.fieldnames = list(data.keys())
        
        # Open file
        mode = 'a' if self.resume and self.log_path.exists() else 'w'
        self.file_handle = open(self.log_path, mode, newline='')
        self.writer = csv.DictWriter(self.file_handle, fieldnames=self.fieldnames)
        
        # Write header if new file
        if mode == 'w' or self.file_handle.tell() == 0:
            self.writer.writeheader()
            self.file_handle.flush()
    
    def log(self, data: Dict[str, Any]):
        """
        Log a single row of data.
        
        Args:
            data (Dict[str, Any]): Dictionary of metric_name -> value
        
        Example:
            >>> logger.log({
            ...     'epoch': 1,
            ...     'train_loss': 0.523,
            ...     'train_acc': 85.3,
            ...     'val_loss': 0.612,
            ...     'val_acc': 82.1,
            ...     'lr': 0.001,
            ...     'epoch_time': 45.2
            ... })
        """
        # Initialize writer on first call
        if self.writer is None:
            self._initialize_writer(data)
        
        # Ensure all fieldnames are present
        if set(data.keys()) != set(self.fieldnames):
            missing = set(self.fieldnames) - set(data.keys())
            extra = set(data.keys()) - set(self.fieldnames)
            
            if missing:
                print(f"Warning: Missing fields in log data: {missing}")
            if extra:
                print(f"Warning: Extra fields in log data: {extra}")
        
        # Write row
        self.writer.writerow(data)
        self.file_handle.flush()  # Ensure data is written immediately
    
    def close(self):
        """Close log file."""
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None
            self.writer = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor to ensure file is closed."""
        self.close()


class EpochLogger:
    """
    Comprehensive epoch-level logger.
    
    Tracks:
    - Training metrics (loss, accuracy, etc.)
    - Validation metrics
    - Test metrics (optional)
    - Learning rate
    - Epoch time
    - GPU memory usage
    
    Automatically logs to CSV and optionally to console.
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str = 'experiment',
        console_log: bool = True
    ):
        """
        Args:
            log_dir (str): Directory to save logs
            experiment_name (str): Name of experiment (for log filename)
            console_log (bool): If True, print logs to console
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.console_log = console_log
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV logger
        csv_filename = f"{experiment_name}_log.csv"
        self.csv_logger = CSVLogger(log_dir=log_dir, filename=csv_filename, resume=False)
        
        # Save experiment config
        self.config_path = self.log_dir / f"{experiment_name}_config.json"
        
        # Tracking
        self.epoch_start_time = None
        self.best_metrics = {}
    
    def save_config(self, config: Dict[str, Any]):
        """
        Save experiment configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    def start_epoch(self):
        """Mark start of epoch (for timing)."""
        self.epoch_start_time = time.time()
    
    def end_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        test_metrics: Optional[Dict[str, float]] = None,
        learning_rate: Optional[float] = None,
        gpu_memory_mb: Optional[float] = None,
        extra_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log end of epoch with all metrics.
        
        Args:
            epoch (int): Current epoch number
            train_metrics (Dict[str, float]): Training metrics
            val_metrics (Dict[str, float], optional): Validation metrics
            test_metrics (Dict[str, float], optional): Test metrics
            learning_rate (float, optional): Current learning rate
            gpu_memory_mb (float, optional): GPU memory usage in MB
            extra_metrics (Dict[str, float], optional): Additional metrics
        """
        # Compute epoch time
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0.0
        
        # Build log dictionary
        log_data = {'epoch': epoch}
        
        # Add training metrics
        for key, value in train_metrics.items():
            log_data[f'train_{key}'] = value
        
        # Add validation metrics
        if val_metrics:
            for key, value in val_metrics.items():
                log_data[f'val_{key}'] = value
        
        # Add test metrics
        if test_metrics:
            for key, value in test_metrics.items():
                log_data[f'test_{key}'] = value
        
        # Add learning rate
        if learning_rate is not None:
            log_data['learning_rate'] = learning_rate
        
        # Add timing
        log_data['epoch_time_sec'] = epoch_time
        
        # Add GPU memory
        if gpu_memory_mb is not None:
            log_data['gpu_memory_mb'] = gpu_memory_mb
        
        # Add extra metrics
        if extra_metrics:
            log_data.update(extra_metrics)
        
        # Log to CSV
        self.csv_logger.log(log_data)
        
        # Console logging
        if self.console_log:
            self._print_epoch_summary(epoch, log_data, epoch_time)
        
        # Track best metrics
        self._update_best_metrics(log_data)
    
    def _print_epoch_summary(self, epoch: int, log_data: Dict[str, Any], epoch_time: float):
        """Print formatted epoch summary to console."""
        print(f"\n{'='*70}")
        print(f"Epoch {epoch} Summary")
        print(f"{'='*70}")
        print(f"Time: {epoch_time:.2f}s")
        
        # Training metrics
        if 'train_loss' in log_data:
            print(f"\nTraining:")
            print(f"  Loss:      {log_data.get('train_loss', 0):.4f}")
            print(f"  Accuracy:  {log_data.get('train_accuracy', 0):.2f}%")
            if 'train_precision' in log_data:
                print(f"  Precision: {log_data.get('train_precision', 0):.2f}%")
                print(f"  Recall:    {log_data.get('train_recall', 0):.2f}%")
                print(f"  F1-Score:  {log_data.get('train_f1_score', 0):.2f}%")
        
        # Validation metrics
        if 'val_loss' in log_data:
            print(f"\nValidation:")
            print(f"  Loss:      {log_data.get('val_loss', 0):.4f}")
            print(f"  Accuracy:  {log_data.get('val_accuracy', 0):.2f}%")
            if 'val_precision' in log_data:
                print(f"  Precision: {log_data.get('val_precision', 0):.2f}%")
                print(f"  Recall:    {log_data.get('val_recall', 0):.2f}%")
                print(f"  F1-Score:  {log_data.get('val_f1_score', 0):.2f}%")
        
        # Learning rate
        if 'learning_rate' in log_data:
            print(f"\nLearning Rate: {log_data['learning_rate']:.6f}")
        
        # GPU memory
        if 'gpu_memory_mb' in log_data:
            print(f"GPU Memory: {log_data['gpu_memory_mb']:.1f} MB")
        
        print(f"{'='*70}\n")
    
    def _update_best_metrics(self, log_data: Dict[str, Any]):
        """Track best metrics across epochs."""
        metrics_to_track = ['val_accuracy', 'val_f1_score', 'train_accuracy']
        
        for metric in metrics_to_track:
            if metric in log_data:
                if metric not in self.best_metrics or log_data[metric] > self.best_metrics[metric]:
                    self.best_metrics[metric] = log_data[metric]
                    self.best_metrics[f'{metric}_epoch'] = log_data['epoch']
    
    def get_best_metrics(self) -> Dict[str, Any]:
        """Get best metrics achieved during training."""
        return self.best_metrics.copy()
    
    def close(self):
        """Close all loggers."""
        self.csv_logger.close()
        
        # Save best metrics
        best_path = self.log_dir / f"{self.experiment_name}_best_metrics.json"
        with open(best_path, 'w') as f:
            json.dump(self.best_metrics, f, indent=4)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Test CSV logger
    print("Testing CSV Logger...\n")
    
    with CSVLogger(log_dir='./test_logs', filename='test.csv') as logger:
        for epoch in range(5):
            logger.log({
                'epoch': epoch,
                'train_loss': 1.0 - epoch * 0.15,
                'train_acc': 60.0 + epoch * 5.0,
                'val_loss': 1.1 - epoch * 0.12,
                'val_acc': 58.0 + epoch * 4.5,
                'lr': 0.001,
            })
    
    print("CSV log saved to: ./test_logs/test.csv\n")
    
    # Test Epoch logger
    print("Testing Epoch Logger...\n")
    
    with EpochLogger(log_dir='./test_logs', experiment_name='test_exp') as logger:
        logger.save_config({'model': 'ResNet-18', 'lr': 0.001})
        
        for epoch in range(3):
            logger.start_epoch()
            time.sleep(0.1)  # Simulate training
            
            train_metrics = {
                'loss': 1.0 - epoch * 0.2,
                'accuracy': 70.0 + epoch * 5.0,
                'precision': 68.0 + epoch * 5.0,
                'recall': 69.0 + epoch * 5.0,
                'f1_score': 68.5 + epoch * 5.0,
            }
            
            val_metrics = {
                'loss': 1.1 - epoch * 0.18,
                'accuracy': 68.0 + epoch * 4.5,
                'precision': 66.0 + epoch * 4.5,
                'recall': 67.0 + epoch * 4.5,
                'f1_score': 66.5 + epoch * 4.5,
            }
            
            logger.end_epoch(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=0.001,
                gpu_memory_mb=2048.5
            )
    
    print("\nLogs saved to: ./test_logs/")
    print("\n✓ All tests passed!")
