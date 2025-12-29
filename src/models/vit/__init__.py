"""
Vision Transformer (ViT) Models for Knowledge Distillation

Supports:
- ViT-Base (Teacher) - 86M params
- ViT-Small (Assistant) - 22M params  
- ViT-Tiny (Student) - 5M params

Memory-optimized for 6GB VRAM:
- Gradient checkpointing
- Mixed precision (AMP)
- Efficient attention
"""

__version__ = "1.0.0"

from .vit_models import get_vit_model, ViTConfig

__all__ = ['get_vit_model', 'ViTConfig']
