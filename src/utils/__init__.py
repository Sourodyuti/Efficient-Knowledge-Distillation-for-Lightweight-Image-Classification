"""
Utilities Module for Knowledge Distillation

Provides:
- Logging and metrics tracking
- Reproducibility utilities
- GPU monitoring
- Memory management
- Training safety guards
"""

__version__ = "1.0.0"

from .logger import EpochLogger
from .reproducibility import setup_reproducibility
from .gpu_monitor import GPUMonitor, get_gpu_memory_usage
from .memory_cleanup import full_cleanup, clear_cuda_cache, clear_python_memory
from .training_guards import (
    freeze_teacher,
    enforce_gradient_checkpointing,
    cleanup_training_state,
    validate_loss,
    reset_peak_memory_stats,
    schedule_kd_alpha,
    get_model_specific_lr
)

__all__ = [
    'EpochLogger',
    'setup_reproducibility',
    'GPUMonitor',
    'get_gpu_memory_usage',
    'full_cleanup',
    'clear_cuda_cache',
    'clear_python_memory',
    'freeze_teacher',
    'enforce_gradient_checkpointing',
    'cleanup_training_state',
    'validate_loss',
    'reset_peak_memory_stats',
    'schedule_kd_alpha',
    'get_model_specific_lr'
]
