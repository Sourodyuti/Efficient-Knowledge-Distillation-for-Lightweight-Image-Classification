#!/usr/bin/env python3
"""
ViT Knowledge Distillation Training Script

Implements Teacher → Assistant → Student distillation pipeline
for ViT-Base → ViT-Small → ViT-Tiny.

Memory-optimized for 6GB VRAM:
- Gradient checkpointing
- Mixed precision (AMP)
- Small batch sizes
- Efficient attention

Usage:
    # Train full pipeline
    python train.py --dataset cifar10 --pretrained --stage all
    
    # Train specific stage
    python train.py --dataset cifar10 --pretrained --stage teacher --batch-size 8
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time
import gc

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cifar10_loader import CIFAR10Processor
from src.data.medmnist_loader import MedMNISTProcessor
from src.models.vit.vit_models import get_vit_model, get_model_info
from src.distillation.kd_loss import KnowledgeDistillationLoss
from src.distillation.metrics import MetricsCalculator
from src.utils.logger import EpochLogger
from src.utils.reproducibility import setup_reproducibility
from src.utils.gpu_monitor import get_gpu_memory_usage, GPUMonitor
from src.utils.memory_cleanup import full_cleanup


class ViTTrainer:
    """
    ViT Knowledge Distillation Trainer.
    
    Optimized for 6GB VRAM:
    - Gradient checkpointing enabled by default
    - Mixed precision training (AMP)
    - Memory-efficient batch sizes
    - Automatic memory cleanup
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # Setup reproducibility
        setup_reproducibility(seed=args.seed, deterministic=not args.no_deterministic)
        
        # GPU monitor
        self.gpu_monitor = GPUMonitor(device_id=0)
        
        # Initialize components
        self.model = None
        self.teacher_model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.kd_criterion = None
        self.scaler = GradScaler() if args.mixed_precision else None
        
        # Setup paths
        self.setup_paths()
        
        # Load data
        self.load_data()
        
        # Initialize metrics and logger
        self.metrics_train = MetricsCalculator(self.num_classes, self.class_names)
        self.metrics_val = MetricsCalculator(self.num_classes, self.class_names)
        
        self.logger = EpochLogger(
            log_dir=self.log_dir,
            experiment_name=f"{args.model}_{args.stage}",
            console_log=True
        )
        
        # Save configuration
        config = vars(args).copy()
        self.logger.save_config(config)
    
    def setup_paths(self):
        """Setup directory paths for logs and checkpoints."""
        pretrained_str = 'imagenet_pretrained' if self.args.pretrained else 'from_scratch'
        base_dir = Path('vit_distillation') / pretrained_str / self.args.dataset / self.args.stage
        
        self.log_dir = base_dir / 'logs'
        self.checkpoint_dir = base_dir / 'checkpoints'
        self.viz_dir = base_dir / 'visualizations'
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cleanup script for this experiment
        self.create_cleanup_script(base_dir)
    
    def create_cleanup_script(self, base_dir):
        """Create cleanup script for this experiment."""
        cleanup_path = base_dir / 'clean.sh'
        
        cleanup_script = f"""#!/bin/bash
echo "Cleaning up {base_dir}..."
python3 << EOF
import sys
sys.path.insert(0, '{Path.cwd().parent}')
from src.utils.memory_cleanup import full_cleanup
full_cleanup(device_id=0, clear_system=False, verbose=True)
EOF
echo "Cleanup complete!"
"""
        
        with open(cleanup_path, 'w') as f:
            f.write(cleanup_script)
        
        os.chmod(cleanup_path, 0o755)
    
    def load_data(self):
        """Load dataset and create dataloaders."""
        print(f"\nLoading {self.args.dataset}...")
        
        if self.args.dataset == 'cifar10':
            processor = CIFAR10Processor(root_dir='../datasets')
            loaders = processor.get_dataloaders(
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                val_split=0.1
            )
        else:
            # MedMNIST dataset
            processor = MedMNISTProcessor(
                dataset_name=self.args.dataset,
                root_dir='../datasets'
            )
            loaders = processor.get_dataloaders(
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers
            )
        
        self.train_loader = loaders['train_loader']
        self.val_loader = loaders['val_loader']
        self.test_loader = loaders['test_loader']
        self.num_classes = loaders['num_classes']
        self.class_names = loaders['class_names']
        
        print(f"Dataset loaded: {self.num_classes} classes")
        print(f"Batch size: {self.args.batch_size} (optimized for 6GB VRAM)")
    
    def build_model(self):
        """Build student model and optionally load teacher."""
        print(f"\nBuilding {self.args.model}...")
        print("Memory optimization: Gradient checkpointing + Mixed precision")
        
        # Create student model with gradient checkpointing
        self.model = get_vit_model(
            model_name=self.args.model,
            num_classes=self.num_classes,
            pretrained=self.args.pretrained and self.args.stage == 'teacher',
            gradient_checkpointing=True  # Always enable for 6GB VRAM
        )
        self.model = self.model.to(self.device)
        
        # Print model info
        info = get_model_info(self.model)
        print(f"Total parameters: {info['total_params']:,}")
        print(f"Trainable parameters: {info['trainable_params']:,}")
        print(f"Estimated memory (FP16 + grad): ~{info['memory_fp16_mb'] * 3:.1f} MB")
        
        # Load teacher for distillation
        if self.args.stage in ['assistant', 'student']:
            teacher_model_name = self.args.model_teacher if self.args.stage == 'assistant' else self.args.model_assistant
            teacher_checkpoint = self.get_teacher_checkpoint(self.args.stage)
            
            if teacher_checkpoint is None:
                raise FileNotFoundError(f"Teacher checkpoint not found for {self.args.stage}")
            
            print(f"\nLoading teacher from {teacher_checkpoint}")
            self.teacher_model = get_vit_model(
                model_name=teacher_model_name,
                num_classes=self.num_classes,
                pretrained=False,
                gradient_checkpointing=True
            )
            self.teacher_model.load_state_dict(torch.load(teacher_checkpoint, map_location=self.device))
            self.teacher_model = self.teacher_model.to(self.device)
            self.teacher_model.eval()
            
            # Freeze teacher
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            
            print("Teacher model loaded and frozen")
    
    def get_teacher_checkpoint(self, stage):
        """Get path to teacher checkpoint."""
        pretrained_str = 'imagenet_pretrained' if self.args.pretrained else 'from_scratch'
        
        if stage == 'assistant':
            checkpoint_dir = Path('vit_distillation') / pretrained_str / self.args.dataset / 'teacher' / 'checkpoints'
        else:  # student
            checkpoint_dir = Path('vit_distillation') / pretrained_str / self.args.dataset / 'assistant' / 'checkpoints'
        
        # Find best checkpoint
        checkpoints = list(checkpoint_dir.glob('best_*.pth'))
        if checkpoints:
            return str(checkpoints[0])
        
        # Fallback to latest
        checkpoints = list(checkpoint_dir.glob('*.pth'))
        return str(checkpoints[-1]) if checkpoints else None
    
    def build_optimizer(self):
        """Build optimizer and scheduler."""
        # Use AdamW with weight decay for ViT
        if self.args.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.args.lr, 
                weight_decay=self.args.weight_decay,
                betas=(0.9, 0.999)
            )
        elif self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.args.lr, 
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=self.args.lr, 
                momentum=0.9, 
                weight_decay=self.args.weight_decay
            )
        
        # Learning rate scheduler
        if self.args.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.args.epochs,
                eta_min=self.args.lr * 0.01
            )
        elif self.args.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=self.args.epochs//3, 
                gamma=0.1
            )
        else:
            self.scheduler = None
    
    def build_criterion(self):
        """Build loss functions."""
        self.criterion = nn.CrossEntropyLoss()
        
        if self.args.stage in ['assistant', 'student']:
            self.kd_criterion = KnowledgeDistillationLoss(
                temperature=self.args.temperature,
                alpha=self.args.alpha
            )
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        self.metrics_train.reset()
        
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs} [Train]")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Mixed precision context
            with autocast(enabled=self.args.mixed_precision):
                # Forward pass
                outputs = self.model(images)
                
                # Compute loss
                if self.args.stage == 'teacher':
                    loss = self.criterion(outputs, labels)
                else:
                    # Distillation
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(images)
                    loss = self.kd_criterion(outputs, teacher_outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)  # More memory efficient
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            self.metrics_train.update(outputs, labels)
            total_loss += loss.item()
            
            # Update progress bar
            if batch_idx % self.args.log_interval == 0:
                current_memory = get_gpu_memory_usage()
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mem_mb': f'{current_memory:.0f}'
                })
            
            # Periodic memory cleanup
            if batch_idx % 50 == 0 and batch_idx > 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(self.train_loader)
        metrics = self.metrics_train.compute(average='macro')
        metrics['loss'] = avg_loss
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate model."""
        self.model.eval()
        self.metrics_val.reset()
        
        total_loss = 0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.args.epochs} [Val]  ")
        
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            self.metrics_val.update(outputs, labels)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        metrics = self.metrics_val.compute(average='macro')
        metrics['loss'] = avg_loss
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint."""
        # Save latest
        torch.save(self.model.state_dict(), self.checkpoint_dir / 'latest.pth')
        
        # Save best
        if is_best:
            torch.save(
                self.model.state_dict(), 
                self.checkpoint_dir / f'best_acc_{metrics["accuracy"]:.2f}.pth'
            )
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*70)
        print(f"Starting training: {self.args.stage.upper()} - {self.args.model.upper()}")
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print("="*70)
        
        # Print initial memory
        if torch.cuda.is_available():
            self.gpu_monitor.print_summary()
        
        # Build components
        self.build_model()
        self.build_optimizer()
        self.build_criterion()
        
        best_val_acc = 0.0
        
        for epoch in range(1, self.args.epochs + 1):
            self.logger.start_epoch()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Learning rate step
            if self.scheduler:
                self.scheduler.step()
            
            # Log epoch
            current_lr = self.optimizer.param_groups[0]['lr']
            gpu_memory = get_gpu_memory_usage()
            
            self.logger.end_epoch(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=current_lr,
                gpu_memory_mb=gpu_memory
            )
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > best_val_acc
            if is_best:
                best_val_acc = val_metrics['accuracy']
            
            if epoch % self.args.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Aggressive memory cleanup for ViT
            if epoch % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        print("\n" + "="*70)
        print(f"Training complete! Best val accuracy: {best_val_acc:.2f}%")
        print("="*70)
        
        self.logger.close()


def parse_args():
    parser = argparse.ArgumentParser(description='ViT Knowledge Distillation Training')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers (reduced for ViT)')
    
    # Model
    parser.add_argument('--stage', type=str, required=True, choices=['teacher', 'assistant', 'student', 'all'])
    parser.add_argument('--model', type=str, help='Model name (auto-selected based on stage)')
    parser.add_argument('--model-teacher', type=str, default='vit_base', help='Teacher model')
    parser.add_argument('--model-assistant', type=str, default='vit_small', help='Assistant model')
    parser.add_argument('--model-student', type=str, default='vit_tiny', help='Student model')
    parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained weights')
    
    # Training (optimized for 6GB VRAM)
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size (8 for ViT-Base on 6GB)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (ViT default)')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'none'])
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay (ViT default)')
    
    # Knowledge Distillation
    parser.add_argument('--temperature', type=float, default=4.0, help='KD temperature')
    parser.add_argument('--alpha', type=float, default=0.7, help='KD alpha (soft target weight)')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--mixed-precision', action='store_true', default=True, help='Enable AMP (default ON)')
    parser.add_argument('--no-mixed-precision', dest='mixed_precision', action='store_false')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-deterministic', action='store_true', help='Disable deterministic mode')
    
    # Logging
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval (batches)')
    parser.add_argument('--save-interval', type=int, default=25, help='Checkpoint save interval (epochs)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Auto-select model based on stage
    if args.model is None:
        if args.stage == 'teacher':
            args.model = args.model_teacher
        elif args.stage == 'assistant':
            args.model = args.model_assistant
        elif args.stage == 'student':
            args.model = args.model_student
    
    # Validate batch size for 6GB VRAM
    if args.model == 'vit_base' and args.batch_size > 8:
        print(f"Warning: Batch size {args.batch_size} may cause OOM on 6GB VRAM.")
        print("Recommended batch size for ViT-Base: 8 or less")
    
    # Train all stages sequentially
    if args.stage == 'all':
        for stage in ['teacher', 'assistant', 'student']:
            args.stage = stage
            if stage == 'teacher':
                args.model = args.model_teacher
                args.batch_size = 8 if args.model == 'vit_base' else 16
            elif stage == 'assistant':
                args.model = args.model_assistant
                args.batch_size = 16
            else:
                args.model = args.model_student
                args.batch_size = 32
            
            trainer = ViTTrainer(args)
            trainer.train()
            
            # Cleanup between stages
            full_cleanup(verbose=True)
    else:
        # Train single stage
        trainer = ViTTrainer(args)
        trainer.train()


if __name__ == '__main__':
    main()
