"""
Analysis Module for Knowledge Distillation

Provides:
- Training curve plotting
- Model comparison
- Performance analysis
- Results aggregation
"""

__version__ = "1.0.0"

from .plot_curves import (
    plot_training_curves,
    plot_comparison,
    plot_metric_vs_epoch,
    create_all_plots
)

__all__ = [
    'plot_training_curves',
    'plot_comparison',
    'plot_metric_vs_epoch',
    'create_all_plots'
]
