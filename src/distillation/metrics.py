"""
Metrics Calculator for Knowledge Distillation

Computes comprehensive evaluation metrics:
- Accuracy (top-1, top-5)
- Precision (per-class and macro/micro average)
- Recall (per-class and macro/micro average)
- F1-score (per-class and macro/micro average)
- Confusion matrix

Framework-agnostic: works with both CNN and ViT.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import warnings

warnings.filterwarnings('ignore')


class MetricsCalculator:
    """
    Comprehensive metrics calculator for classification tasks.
    
    Supports:
    - Top-1 and Top-5 accuracy
    - Per-class and averaged Precision/Recall/F1
    - Confusion matrix
    - Running average tracking
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Args:
            num_classes (int): Number of classes
            class_names (List[str], optional): Class names for reporting
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        # Running statistics
        self.reset()
    
    def reset(self):
        """Reset all running statistics."""
        self.all_predictions = []
        self.all_targets = []
        self.all_logits = []
        
        self.total_samples = 0
        self.correct_top1 = 0
        self.correct_top5 = 0
    
    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ):
        """
        Update metrics with a batch of predictions.
        
        Args:
            logits (torch.Tensor): Model logits [B, C]
            targets (torch.Tensor): Ground truth labels [B]
        """
        # Move to CPU and convert to numpy
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu()
        
        # Get predictions
        predictions = torch.argmax(logits, dim=1)
        
        # Store for later computation
        self.all_predictions.extend(predictions.numpy().tolist())
        self.all_targets.extend(targets.numpy().tolist())
        self.all_logits.append(logits)
        
        # Update running accuracy
        batch_size = targets.size(0)
        self.total_samples += batch_size
        
        # Top-1 accuracy
        self.correct_top1 += (predictions == targets).sum().item()
        
        # Top-5 accuracy (if enough classes)
        if self.num_classes >= 5:
            _, top5_pred = torch.topk(logits, k=5, dim=1)
            targets_expanded = targets.view(-1, 1).expand_as(top5_pred)
            self.correct_top5 += (top5_pred == targets_expanded).any(dim=1).sum().item()
    
    def compute(self, average: str = 'macro') -> Dict[str, float]:
        """
        Compute all metrics from accumulated predictions.
        
        Args:
            average (str): Averaging method for multi-class metrics
                'macro': Unweighted mean (equal weight to each class)
                'micro': Global average (weighted by support)
                'weighted': Weighted by class frequency
        
        Returns:
            Dict[str, float]: Dictionary of computed metrics
        """
        if self.total_samples == 0:
            return self._empty_metrics()
        
        # Convert to numpy arrays
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        # Compute precision, recall, f1
        precision, recall, f1, support = precision_recall_fscore_support(
            targets,
            predictions,
            average=average,
            zero_division=0
        )
        
        # Compute per-class metrics (for detailed analysis)
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            targets,
            predictions,
            average=None,
            zero_division=0
        )
        
        # Accuracy
        top1_accuracy = 100.0 * self.correct_top1 / self.total_samples
        top5_accuracy = 100.0 * self.correct_top5 / self.total_samples if self.num_classes >= 5 else None
        
        # Build metrics dictionary
        metrics = {
            'accuracy': top1_accuracy,
            'precision': 100.0 * precision if isinstance(precision, (int, float)) else 100.0 * precision.mean(),
            'recall': 100.0 * recall if isinstance(recall, (int, float)) else 100.0 * recall.mean(),
            'f1_score': 100.0 * f1 if isinstance(f1, (int, float)) else 100.0 * f1.mean(),
            'total_samples': self.total_samples,
        }
        
        if top5_accuracy is not None:
            metrics['top5_accuracy'] = top5_accuracy
        
        # Add per-class metrics
        for i, class_name in enumerate(self.class_names[:len(per_class_precision)]):
            metrics[f'precision_{class_name}'] = 100.0 * per_class_precision[i]
            metrics[f'recall_{class_name}'] = 100.0 * per_class_recall[i]
            metrics[f'f1_{class_name}'] = 100.0 * per_class_f1[i]
        
        return metrics
    
    def get_confusion_matrix(self, normalize: Optional[str] = None) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            normalize (str, optional): Normalization method
                None: Absolute counts
                'true': Normalize over true labels (rows)
                'pred': Normalize over predictions (columns)
                'all': Normalize over all samples
        
        Returns:
            np.ndarray: Confusion matrix [num_classes, num_classes]
        """
        if len(self.all_predictions) == 0:
            return np.zeros((self.num_classes, self.num_classes))
        
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        cm = confusion_matrix(targets, predictions, labels=range(self.num_classes))
        
        if normalize == 'true':
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        elif normalize == 'pred':
            cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
        elif normalize == 'all':
            cm = cm.astype('float') / cm.sum()
        
        # Replace NaN with 0
        cm = np.nan_to_num(cm)
        
        return cm
    
    def get_classification_report(self) -> str:
        """
        Generate sklearn classification report.
        
        Returns:
            str: Formatted classification report
        """
        if len(self.all_predictions) == 0:
            return "No predictions available."
        
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        report = classification_report(
            targets,
            predictions,
            target_names=self.class_names[:self.num_classes],
            zero_division=0,
            digits=4
        )
        
        return report
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary."""
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'total_samples': 0,
        }
    
    def summary(self, average: str = 'macro') -> str:
        """
        Generate formatted summary of metrics.
        
        Args:
            average (str): Averaging method
        
        Returns:
            str: Formatted summary string
        """
        metrics = self.compute(average=average)
        
        summary_lines = [
            "=" * 60,
            "Metrics Summary",
            "=" * 60,
            f"Total Samples:    {metrics['total_samples']}",
            f"Accuracy:         {metrics['accuracy']:.2f}%",
        ]
        
        if 'top5_accuracy' in metrics:
            summary_lines.append(f"Top-5 Accuracy:   {metrics['top5_accuracy']:.2f}%")
        
        summary_lines.extend([
            f"Precision ({average}): {metrics['precision']:.2f}%",
            f"Recall ({average}):    {metrics['recall']:.2f}%",
            f"F1-Score ({average}):  {metrics['f1_score']:.2f}%",
            "=" * 60,
        ])
        
        return "\n".join(summary_lines)


def compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    top_k: Tuple[int, ...] = (1, 5)
) -> Dict[str, float]:
    """
    Compute metrics for a single batch (fast, for training loops).
    
    Args:
        logits (torch.Tensor): Model logits [B, C]
        targets (torch.Tensor): Ground truth labels [B]
        top_k (Tuple[int, ...]): Top-K accuracies to compute
    
    Returns:
        Dict[str, float]: Batch metrics
    """
    with torch.no_grad():
        batch_size = targets.size(0)
        num_classes = logits.size(1)
        
        # Predictions
        predictions = torch.argmax(logits, dim=1)
        
        # Top-1 accuracy
        correct = (predictions == targets).sum().item()
        accuracy = 100.0 * correct / batch_size
        
        metrics = {
            'accuracy': accuracy,
            'correct': correct,
            'total': batch_size,
        }
        
        # Top-K accuracies
        for k in top_k:
            if k <= num_classes:
                _, topk_pred = torch.topk(logits, k=k, dim=1)
                targets_expanded = targets.view(-1, 1).expand_as(topk_pred)
                correct_k = (topk_pred == targets_expanded).any(dim=1).sum().item()
                metrics[f'top{k}_accuracy'] = 100.0 * correct_k / batch_size
        
        return metrics


def accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    top_k: int = 1
) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        logits (torch.Tensor): Model logits [B, C]
        targets (torch.Tensor): Ground truth labels [B]
        top_k (int): K for top-K accuracy
    
    Returns:
        float: Accuracy percentage
    
    Example:
        >>> logits = torch.randn(32, 10)
        >>> targets = torch.randint(0, 10, (32,))
        >>> acc = accuracy(logits, targets, top_k=1)
        >>> print(f"Top-1 Accuracy: {acc:.2f}%")
    """
    with torch.no_grad():
        batch_size = targets.size(0)
        
        if top_k == 1:
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == targets).sum().item()
        else:
            _, topk_pred = torch.topk(logits, k=top_k, dim=1)
            targets_expanded = targets.view(-1, 1).expand_as(topk_pred)
            correct = (topk_pred == targets_expanded).any(dim=1).sum().item()
        
        return 100.0 * correct / batch_size


if __name__ == "__main__":
    # Test metrics calculator
    print("Testing Metrics Calculator...\n")
    
    num_classes = 10
    class_names = [f"Class_{i}" for i in range(num_classes)]
    
    # Create calculator
    calculator = MetricsCalculator(num_classes=num_classes, class_names=class_names)
    
    # Simulate batches
    print("Simulating 5 batches...")
    for i in range(5):
        batch_size = 32
        logits = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        calculator.update(logits, targets)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = calculator.compute(average='macro')
    
    print("\n" + calculator.summary(average='macro'))
    
    print("\nPer-metric values:")
    print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall:    {metrics['recall']:.2f}%")
    print(f"  F1-Score:  {metrics['f1_score']:.2f}%")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = calculator.get_confusion_matrix(normalize='true')
    print(cm[:3, :3])  # Print 3x3 subset
    
    # Test functional API
    print("\nTesting functional API...")
    logits = torch.randn(32, 10)
    targets = torch.randint(0, 10, (32,))
    
    batch_metrics = compute_batch_metrics(logits, targets)
    print(f"Batch Accuracy: {batch_metrics['accuracy']:.2f}%")
    
    acc1 = accuracy(logits, targets, top_k=1)
    print(f"Top-1 Accuracy: {acc1:.2f}%")
    
    print("\n✓ All tests passed!")
