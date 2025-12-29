"""
Utility Functions for Knowledge Distillation

Common utilities:
- CSV logging
- Reproducibility (seed setting)
- GPU monitoring
- Memory cleanup
- Configuration management

Framework-agnostic.
"""

__version__ = "1.0.0"

from .logger import CSVLogger, EpochLogger
from .reproducibility import set_seed, set_deterministic
from .gpu_monitor import GPUMonitor, get_gpu_memory_usage

__all__ = [
    'CSVLogger',
    'EpochLogger',
    'set_seed',
    'set_deterministic',
    'GPUMonitor',
    'get_gpu_memory_usage',
]
