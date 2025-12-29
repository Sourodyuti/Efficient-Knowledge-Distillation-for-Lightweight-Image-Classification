"""
Knowledge Distillation Core Module

Framework-agnostic knowledge distillation components:
- KD loss functions
- Temperature scaling
- Hard + soft label blending
- Evaluation metrics

Reusable across CNN and ViT experiments.
"""

__version__ = "1.0.0"

from .kd_loss import (
    KnowledgeDistillationLoss,
    distillation_loss,
    soft_cross_entropy,
)
from .metrics import MetricsCalculator

__all__ = [
    'KnowledgeDistillationLoss',
    'distillation_loss',
    'soft_cross_entropy',
    'MetricsCalculator',
]
