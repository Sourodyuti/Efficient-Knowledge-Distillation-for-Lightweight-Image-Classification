#!/usr/bin/env python3
"""
Plotting Scripts for Knowledge Distillation Analysis

Creates comprehensive visualizations:
- Accuracy vs Epoch
- Loss vs Epoch
- F1-Score vs Epoch
- Model comparisons
- Training/Validation curves
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_log_csv(log_path: str) -> pd.DataFrame:
    """
    Load training log CSV file.
    
    Args:
        log_path (str): Path to CSV log file
    
    Returns:
        pd.DataFrame: Loaded data
    """
    return pd.read_csv(log_path)


def plot_metric_vs_epoch(
    df: pd.DataFrame,
    metric: str = 'accuracy',
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show_val: bool = True
):
    """
    Plot metric vs epoch for training and validation.
    
    Args:
        df (pd.DataFrame): Training log dataframe
        metric (str): Metric to plot (accuracy, loss, f1_score, etc.)
        title (str, optional): Plot title
        save_path (str, optional): Path to save figure
        show_val (bool): Show validation curve
    
    Example:
        >>> df = load_log_csv('logs/resnet18_log.csv')
        >>> plot_metric_vs_epoch(df, metric='accuracy', save_path='accuracy.png')
    """
    plt.figure(figsize=(10, 6))
    
    # Plot training metric
    train_col = f'train_{metric}'
    if train_col in df.columns:
        plt.plot(df['epoch'], df[train_col], label=f'Train {metric.title()}', linewidth=2, marker='o', markersize=3)
    
    # Plot validation metric
    if show_val:
        val_col = f'val_{metric}'
        if val_col in df.columns:
            plt.plot(df['epoch'], df[val_col], label=f'Val {metric.title()}', linewidth=2, marker='s', markersize=3)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(title or f'{metric.replace("_", " ").title()} vs Epoch', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(
    log_path: str,
    save_dir: str = './results/plots',
    model_name: str = 'Model'
):
    """
    Create all training curves for a single model.
    
    Creates:
    - Accuracy vs Epoch
    - Loss vs Epoch
    - F1-Score vs Epoch
    - Precision vs Epoch
    - Recall vs Epoch
    
    Args:
        log_path (str): Path to training log CSV
        save_dir (str): Directory to save plots
        model_name (str): Model name for titles
    
    Example:
        >>> plot_training_curves(
        ...     'logs/resnet18_teacher_log.csv',
        ...     save_dir='results/plots/teacher',
        ...     model_name='ResNet-18 Teacher'
        ... )
    """
    df = load_log_csv(log_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating training curves for {model_name}...")
    
    # Accuracy
    plot_metric_vs_epoch(
        df,
        metric='accuracy',
        title=f'{model_name} - Accuracy vs Epoch',
        save_path=save_dir / 'accuracy_vs_epoch.png'
    )
    
    # Loss
    plot_metric_vs_epoch(
        df,
        metric='loss',
        title=f'{model_name} - Loss vs Epoch',
        save_path=save_dir / 'loss_vs_epoch.png'
    )
    
    # F1-Score
    plot_metric_vs_epoch(
        df,
        metric='f1_score',
        title=f'{model_name} - F1-Score vs Epoch',
        save_path=save_dir / 'f1_score_vs_epoch.png'
    )
    
    # Precision
    plot_metric_vs_epoch(
        df,
        metric='precision',
        title=f'{model_name} - Precision vs Epoch',
        save_path=save_dir / 'precision_vs_epoch.png'
    )
    
    # Recall
    plot_metric_vs_epoch(
        df,
        metric='recall',
        title=f'{model_name} - Recall vs Epoch',
        save_path=save_dir / 'recall_vs_epoch.png'
    )
    
    print(f"All plots saved to {save_dir}")


def plot_comparison(
    log_paths: Dict[str, str],
    metric: str = 'accuracy',
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    plot_train: bool = False
):
    """
    Compare multiple models on the same metric.
    
    Args:
        log_paths (Dict[str, str]): Dict mapping model names to log paths
        metric (str): Metric to compare
        title (str, optional): Plot title
        save_path (str, optional): Path to save figure
        plot_train (bool): Plot training curves (otherwise validation only)
    
    Example:
        >>> log_paths = {
        ...     'Teacher (ResNet-50)': 'logs/teacher_log.csv',
        ...     'Assistant (ResNet-34)': 'logs/assistant_log.csv',
        ...     'Student (ResNet-18)': 'logs/student_log.csv'
        ... }
        >>> plot_comparison(log_paths, metric='accuracy', save_path='comparison.png')
    """
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(log_paths)))
    
    for idx, (model_name, log_path) in enumerate(log_paths.items()):
        df = load_log_csv(log_path)
        
        col = f'val_{metric}' if not plot_train else f'train_{metric}'
        if col in df.columns:
            plt.plot(
                df['epoch'],
                df[col],
                label=model_name,
                linewidth=2.5,
                marker='o',
                markersize=4,
                color=colors[idx],
                alpha=0.8
            )
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    
    if title is None:
        mode = 'Training' if plot_train else 'Validation'
        title = f'Model Comparison - {mode} {metric.replace("_", " ").title()}'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_multi_metric_comparison(
    log_paths: Dict[str, str],
    save_path: str = './results/plots/multi_metric_comparison.png'
):
    """
    Create a 2x2 grid comparing multiple metrics across models.
    
    Args:
        log_paths (Dict[str, str]): Dict mapping model names to log paths
        save_path (str): Path to save figure
    
    Example:
        >>> log_paths = {
        ...     'Teacher': 'logs/teacher_log.csv',
        ...     'Assistant': 'logs/assistant_log.csv',
        ...     'Student': 'logs/student_log.csv'
        ... }
        >>> plot_multi_metric_comparison(log_paths)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics = ['accuracy', 'loss', 'f1_score', 'precision']
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(log_paths)))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        for model_idx, (model_name, log_path) in enumerate(log_paths.items()):
            df = load_log_csv(log_path)
            col = f'val_{metric}'
            
            if col in df.columns:
                ax.plot(
                    df['epoch'],
                    df[col],
                    label=model_name,
                    linewidth=2,
                    marker='o',
                    markersize=3,
                    color=colors[model_idx]
                )
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'Validation {metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def create_performance_summary(
    log_paths: Dict[str, str],
    save_path: str = './results/performance_summary.csv'
):
    """
    Create a summary table of final performance metrics.
    
    Args:
        log_paths (Dict[str, str]): Dict mapping model names to log paths
        save_path (str): Path to save CSV
    
    Returns:
        pd.DataFrame: Summary dataframe
    """
    summary = []
    
    for model_name, log_path in log_paths.items():
        df = load_log_csv(log_path)
        
        # Get best epoch (highest val accuracy)
        best_idx = df['val_accuracy'].idxmax()
        best_row = df.iloc[best_idx]
        
        summary.append({
            'Model': model_name,
            'Best Epoch': int(best_row['epoch']),
            'Val Accuracy (%)': f"{best_row['val_accuracy']:.2f}",
            'Val Precision (%)': f"{best_row['val_precision']:.2f}",
            'Val Recall (%)': f"{best_row['val_recall']:.2f}",
            'Val F1-Score (%)': f"{best_row['val_f1_score']:.2f}",
            'Val Loss': f"{best_row['val_loss']:.4f}",
            'Train Accuracy (%)': f"{best_row['train_accuracy']:.2f}",
            'Train Loss': f"{best_row['train_loss']:.4f}",
        })
    
    summary_df = pd.DataFrame(summary)
    
    # Save to CSV
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(save_path, index=False)
    print(f"\nPerformance summary saved to {save_path}")
    
    return summary_df


def create_all_plots(
    experiment_dir: str,
    architecture: str = 'cnn',
    dataset: str = 'cifar10',
    pretrained: bool = True
):
    """
    Create all plots for a complete experiment.
    
    Args:
        experiment_dir (str): Base experiment directory (cnn_distillation or vit_distillation)
        architecture (str): 'cnn' or 'vit'
        dataset (str): Dataset name
        pretrained (bool): Whether using pretrained weights
    
    Example:
        >>> create_all_plots(
        ...     experiment_dir='cnn_distillation',
        ...     architecture='cnn',
        ...     dataset='cifar10',
        ...     pretrained=True
        ... )
    """
    pretrained_str = 'imagenet_pretrained' if pretrained else 'from_scratch'
    base_dir = Path(experiment_dir) / pretrained_str / dataset
    results_dir = base_dir / 'results' / 'plots'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"Creating plots for {architecture.upper()} - {dataset.upper()}")
    print(f"Pretrained: {pretrained}")
    print("="*70)
    
    # Find log files
    log_paths = {}
    stages = ['teacher', 'assistant', 'student']
    
    for stage in stages:
        stage_dir = base_dir / stage / 'logs'
        if stage_dir.exists():
            # Find CSV log file
            csv_files = list(stage_dir.glob('*_log.csv'))
            if csv_files:
                log_paths[stage.title()] = str(csv_files[0])
    
    if not log_paths:
        print("\nNo log files found. Please train models first.")
        return
    
    print(f"\nFound {len(log_paths)} models: {', '.join(log_paths.keys())}")
    
    # Create individual training curves
    print("\n[1/4] Creating individual training curves...")
    for model_name, log_path in log_paths.items():
        model_plot_dir = results_dir / model_name.lower()
        plot_training_curves(
            log_path=log_path,
            save_dir=str(model_plot_dir),
            model_name=f"{model_name} ({architecture.upper()})"
        )
    
    # Create comparison plots
    print("\n[2/4] Creating comparison plots...")
    
    # Accuracy comparison
    plot_comparison(
        log_paths=log_paths,
        metric='accuracy',
        title=f'{architecture.upper()} Distillation - Validation Accuracy',
        save_path=results_dir / 'comparison_accuracy.png'
    )
    
    # Loss comparison
    plot_comparison(
        log_paths=log_paths,
        metric='loss',
        title=f'{architecture.upper()} Distillation - Validation Loss',
        save_path=results_dir / 'comparison_loss.png'
    )
    
    # F1-Score comparison
    plot_comparison(
        log_paths=log_paths,
        metric='f1_score',
        title=f'{architecture.upper()} Distillation - Validation F1-Score',
        save_path=results_dir / 'comparison_f1_score.png'
    )
    
    # Multi-metric comparison
    print("\n[3/4] Creating multi-metric comparison...")
    plot_multi_metric_comparison(
        log_paths=log_paths,
        save_path=results_dir / 'multi_metric_comparison.png'
    )
    
    # Performance summary
    print("\n[4/4] Creating performance summary...")
    summary_df = create_performance_summary(
        log_paths=log_paths,
        save_path=results_dir.parent / 'performance_summary.csv'
    )
    
    print("\n" + "="*70)
    print("Performance Summary:")
    print("="*70)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*70)
    print("✓ All plots created successfully!")
    print(f"Results saved to: {results_dir}")
    print("="*70)


if __name__ == "__main__":
    # Example usage
    print("Testing plotting functions...\n")
    
    # Create dummy data
    dummy_data = pd.DataFrame({
        'epoch': range(1, 51),
        'train_loss': np.linspace(2.3, 0.3, 50) + np.random.randn(50) * 0.1,
        'train_accuracy': np.linspace(20, 95, 50) + np.random.randn(50) * 2,
        'train_precision': np.linspace(18, 94, 50) + np.random.randn(50) * 2,
        'train_recall': np.linspace(19, 93, 50) + np.random.randn(50) * 2,
        'train_f1_score': np.linspace(18.5, 93.5, 50) + np.random.randn(50) * 2,
        'val_loss': np.linspace(2.5, 0.4, 50) + np.random.randn(50) * 0.1,
        'val_accuracy': np.linspace(18, 92, 50) + np.random.randn(50) * 2,
        'val_precision': np.linspace(16, 91, 50) + np.random.randn(50) * 2,
        'val_recall': np.linspace(17, 90, 50) + np.random.randn(50) * 2,
        'val_f1_score': np.linspace(16.5, 90.5, 50) + np.random.randn(50) * 2,
    })
    
    # Save dummy CSV
    dummy_data.to_csv('./test_log.csv', index=False)
    
    # Test plotting
    print("[1/2] Testing single metric plot...")
    plot_metric_vs_epoch(
        df=dummy_data,
        metric='accuracy',
        title='Test - Accuracy vs Epoch',
        save_path='./test_accuracy.png'
    )
    
    print("\n[2/2] Testing training curves...")
    plot_training_curves(
        log_path='./test_log.csv',
        save_dir='./test_plots',
        model_name='Test Model'
    )
    
    print("\n✓ All tests passed!")
