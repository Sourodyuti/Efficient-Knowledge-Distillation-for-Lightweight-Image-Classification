"""
Visualization Module for Knowledge Distillation

Provides:
- CNN feature map visualization
- ViT attention map visualization
- Gradient flow visualization
- Loss curve plotting
"""

__version__ = "1.0.0"

from .cnn_viz import visualize_feature_maps, save_feature_maps

__all__ = ['visualize_feature_maps', 'save_feature_maps']
