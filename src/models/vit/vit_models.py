#!/usr/bin/env python3
"""
Vision Transformer Models for Knowledge Distillation

Provides ViT-Tiny, ViT-Small, ViT-Base with:
- ImageNet pretrained weights support
- Gradient checkpointing for memory efficiency
- Custom classifier for any number of classes
- Attention weight extraction

Optimized for 6GB VRAM:
- Gradient checkpointing reduces memory by ~40%
- Compatible with mixed precision training
- Efficient implementation
"""

import torch
import torch.nn as nn
from torchvision import models
from dataclasses import dataclass
from typing import Optional, Literal
import warnings


@dataclass
class ViTConfig:
    """Configuration for ViT models."""
    model_name: Literal['vit_tiny', 'vit_small', 'vit_base']
    num_classes: int
    pretrained: bool = True
    gradient_checkpointing: bool = True  # Enable for 6GB VRAM
    freeze_backbone: bool = False
    
    def __post_init__(self):
        valid_models = ['vit_tiny', 'vit_small', 'vit_base']
        if self.model_name not in valid_models:
            raise ValueError(f"model_name must be one of {valid_models}")


def get_vit_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    gradient_checkpointing: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Get ViT model with custom classifier.
    
    Args:
        model_name (str): 'vit_tiny', 'vit_small', or 'vit_base'
        num_classes (int): Number of output classes
        pretrained (bool): Load ImageNet pretrained weights
        gradient_checkpointing (bool): Enable gradient checkpointing (saves ~40% memory)
        freeze_backbone (bool): Freeze backbone weights (fine-tuning mode)
    
    Returns:
        nn.Module: ViT model
    
    Memory Usage (approximate, batch_size=16, FP16):
        - ViT-Base: ~4.5GB with gradient checkpointing
        - ViT-Small: ~2.5GB with gradient checkpointing
        - ViT-Tiny: ~1.5GB with gradient checkpointing
    
    Example:
        >>> model = get_vit_model(
        ...     'vit_tiny', 
        ...     num_classes=10, 
        ...     pretrained=True,
        ...     gradient_checkpointing=True
        ... )
        >>> output = model(torch.randn(1, 3, 224, 224))
        >>> print(output.shape)  # torch.Size([1, 10])
    """
    # Get model constructor
    if model_name == 'vit_tiny':
        # ViT-Tiny: Custom implementation (not in torchvision)
        # We'll use vit_b_16 with reduced dimensions
        model = _create_vit_tiny(num_classes, pretrained)
    elif model_name == 'vit_small':
        # ViT-Small: Custom implementation
        model = _create_vit_small(num_classes, pretrained)
    elif model_name == 'vit_base':
        if pretrained:
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            model = models.vit_b_16(weights=None)
        
        # Replace classifier head
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Enable gradient checkpointing for memory efficiency
    if gradient_checkpointing:
        if hasattr(model, 'encoder'):
            for block in model.encoder.layers:
                if hasattr(block, 'set_grad_checkpointing'):
                    block.set_grad_checkpointing(True)
                else:
                    # Manual gradient checkpointing
                    _enable_gradient_checkpointing(block)
    
    # Freeze backbone if requested
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'heads' not in name:  # Don't freeze classifier
                param.requires_grad = False
    
    # Always train the classifier head
    if hasattr(model, 'heads'):
        for param in model.heads.parameters():
            param.requires_grad = True
    
    return model


def _create_vit_tiny(num_classes: int, pretrained: bool) -> nn.Module:
    """
    Create ViT-Tiny model.
    
    Architecture:
    - Patch size: 16x16
    - Hidden dim: 192
    - Depth: 12 layers
    - Heads: 3
    - MLP dim: 768
    - Parameters: ~5M
    """
    # Start with base model and modify
    if pretrained:
        warnings.warn(
            "ViT-Tiny pretrained weights not available. Using random initialization."
        )
    
    from torchvision.models.vision_transformer import VisionTransformer
    
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=3,
        hidden_dim=192,
        mlp_dim=768,
        num_classes=num_classes
    )
    
    return model


def _create_vit_small(num_classes: int, pretrained: bool) -> nn.Module:
    """
    Create ViT-Small model.
    
    Architecture:
    - Patch size: 16x16
    - Hidden dim: 384
    - Depth: 12 layers
    - Heads: 6
    - MLP dim: 1536
    - Parameters: ~22M
    """
    if pretrained:
        warnings.warn(
            "ViT-Small pretrained weights not available. Using random initialization."
        )
    
    from torchvision.models.vision_transformer import VisionTransformer
    
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=6,
        hidden_dim=384,
        mlp_dim=1536,
        num_classes=num_classes
    )
    
    return model


def _enable_gradient_checkpointing(module: nn.Module):
    """
    Enable gradient checkpointing for a module.
    
    This wraps the forward pass to use torch.utils.checkpoint,
    trading compute for memory (recompute activations during backward).
    """
    if hasattr(module, 'forward'):
        original_forward = module.forward
        
        def checkpointed_forward(*args, **kwargs):
            if module.training:
                return torch.utils.checkpoint.checkpoint(
                    original_forward, *args, **kwargs, use_reentrant=False
                )
            else:
                return original_forward(*args, **kwargs)
        
        module.forward = checkpointed_forward


class ViTWithAttention(nn.Module):
    """
    ViT wrapper that extracts attention weights.
    Useful for visualization and analysis.
    """
    
    def __init__(self, base_model: nn.Module):
        """
        Args:
            base_model (nn.Module): Base ViT model
        """
        super().__init__()
        self.model = base_model
        self.attention_weights = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture attention weights."""
        def get_hook(layer_idx):
            def hook(module, input, output):
                # Attention module outputs (output, attention_weights)
                if isinstance(output, tuple) and len(output) > 1:
                    self.attention_weights.append(output[1].detach())
            return hook
        
        # Register hooks on attention layers
        if hasattr(self.model, 'encoder'):
            for idx, layer in enumerate(self.model.encoder.layers):
                if hasattr(layer, 'self_attention'):
                    layer.self_attention.register_forward_hook(get_hook(idx))
    
    def forward(self, x):
        """Forward pass with attention extraction."""
        self.attention_weights = []  # Clear previous
        output = self.model(x)
        return output, self.attention_weights


def get_model_info(model: nn.Module) -> dict:
    """
    Get model information (parameters, memory, etc.).
    
    Args:
        model (nn.Module): Model
    
    Returns:
        dict: Model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory (rough)
    memory_fp32 = total_params * 4 / (1024**2)  # MB
    memory_fp16 = total_params * 2 / (1024**2)  # MB
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'memory_fp32_mb': memory_fp32,
        'memory_fp16_mb': memory_fp16,
    }


if __name__ == "__main__":
    # Test ViT models
    print("Testing ViT Models...\n")
    
    models_to_test = ['vit_tiny', 'vit_small', 'vit_base']
    
    for model_name in models_to_test:
        print(f"Testing {model_name.upper().replace('_', '-')}...")
        
        # Create model
        model = get_vit_model(
            model_name, 
            num_classes=10, 
            pretrained=False,
            gradient_checkpointing=True
        )
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Model info
        info = get_model_info(model)
        print(f"  Total params:     {info['total_params']:,}")
        print(f"  Trainable params: {info['trainable_params']:,}")
        print(f"  Memory (FP32):    {info['memory_fp32_mb']:.1f} MB")
        print(f"  Memory (FP16):    {info['memory_fp16_mb']:.1f} MB")
        print()
    
    print("✓ All tests passed!")
