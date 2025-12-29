"""
ResNet Models for Knowledge Distillation

Provides ResNet-18, ResNet-34, ResNet-50 with:
- ImageNet pretrained weights support
- Custom classifier for any number of classes
- Feature extraction capability
"""

import torch
import torch.nn as nn
from torchvision import models
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class ResNetConfig:
    """Configuration for ResNet models."""
    model_name: Literal['resnet18', 'resnet34', 'resnet50']
    num_classes: int
    pretrained: bool = True
    freeze_backbone: bool = False
    
    def __post_init__(self):
        valid_models = ['resnet18', 'resnet34', 'resnet50']
        if self.model_name not in valid_models:
            raise ValueError(f"model_name must be one of {valid_models}")


def get_resnet_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Get ResNet model with custom classifier.
    
    Args:
        model_name (str): 'resnet18', 'resnet34', or 'resnet50'
        num_classes (int): Number of output classes
        pretrained (bool): Load ImageNet pretrained weights
        freeze_backbone (bool): Freeze backbone weights (fine-tuning mode)
    
    Returns:
        nn.Module: ResNet model
    
    Example:
        >>> model = get_resnet_model('resnet18', num_classes=10, pretrained=True)
        >>> output = model(torch.randn(1, 3, 224, 224))
        >>> print(output.shape)  # torch.Size([1, 10])
    """
    # Get model constructor
    if model_name == 'resnet18':
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights=None)
    elif model_name == 'resnet34':
        if pretrained:
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet34(weights=None)
    elif model_name == 'resnet50':
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    # Always train the final layer
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet with feature extraction at multiple layers.
    Useful for visualization and analysis.
    """
    
    def __init__(self, model: nn.Module, extract_layers: list = None):
        """
        Args:
            model (nn.Module): Base ResNet model
            extract_layers (list): Layer names to extract features from
        """
        super().__init__()
        self.model = model
        self.extract_layers = extract_layers or ['layer4', 'avgpool']
        self.features = {}
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to extract features."""
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output.detach()
            return hook
        
        for name, module in self.model.named_modules():
            if name in self.extract_layers:
                module.register_forward_hook(get_hook(name))
    
    def forward(self, x):
        """Forward pass with feature extraction."""
        self.features = {}  # Clear previous features
        output = self.model(x)
        return output, self.features


if __name__ == "__main__":
    # Test ResNet models
    print("Testing ResNet Models...\n")
    
    models_to_test = ['resnet18', 'resnet34', 'resnet50']
    
    for model_name in models_to_test:
        print(f"Testing {model_name.upper()}...")
        
        # Create model
        model = get_resnet_model(model_name, num_classes=10, pretrained=False)
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable:    {trainable_params:,}")
        print()
    
    print("✓ All tests passed!")
