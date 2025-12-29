"""
CNN Models for Knowledge Distillation

ResNet family:
- ResNet-50 (Teacher)
- ResNet-34 (Assistant)
- ResNet-18 (Student)

Supports:
- ImageNet pretrained weights
- Training from scratch
- Custom number of classes
"""

__version__ = "1.0.0"

from .resnet import get_resnet_model, ResNetConfig

__all__ = ['get_resnet_model', 'ResNetConfig']
