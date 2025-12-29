"""
Visualization Module for Knowledge Distillation

Provides:
- CNN feature map visualization
- ViT attention map visualization
- Attention rollout
- Loss curve plotting
"""

__version__ = "1.0.0"

from .cnn_viz import visualize_feature_maps, save_feature_maps
from .vit_viz import (
    visualize_attention_map,
    visualize_multi_head_attention,
    extract_attention_rollout,
    save_attention_visualizations
)

__all__ = [
    'visualize_feature_maps',
    'save_feature_maps',
    'visualize_attention_map',
    'visualize_multi_head_attention',
    'extract_attention_rollout',
    'save_attention_visualizations'
]
