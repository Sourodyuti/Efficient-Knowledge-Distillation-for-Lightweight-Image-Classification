#!/usr/bin/env python3
"""
CNN Feature Map Visualization

Provides tools to visualize CNN feature maps from ResNet models.
Useful for understanding what the model learns at different layers.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple


def visualize_feature_maps(
    feature_maps: torch.Tensor,
    num_maps: int = 16,
    title: str = "Feature Maps",
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
):
    """
    Visualize feature maps from a CNN layer.
    
    Args:
        feature_maps (torch.Tensor): Feature maps [B, C, H, W]
        num_maps (int): Number of feature maps to visualize
        title (str): Plot title
        figsize (Tuple[int, int]): Figure size
        save_path (str, optional): Path to save figure
    
    Example:
        >>> # During forward pass
        >>> features = model.layer4(x)  # Get features from layer4
        >>> visualize_feature_maps(features[0], num_maps=16, title="Layer 4 Features")
    """
    # Move to CPU and convert to numpy
    if isinstance(feature_maps, torch.Tensor):
        feature_maps = feature_maps.detach().cpu().numpy()
    
    # If batch dimension exists, take first sample
    if feature_maps.ndim == 4:
        feature_maps = feature_maps[0]  # [C, H, W]
    
    num_channels = feature_maps.shape[0]
    num_maps = min(num_maps, num_channels)
    
    # Calculate grid size
    ncols = int(np.ceil(np.sqrt(num_maps)))
    nrows = int(np.ceil(num_maps / ncols))
    
    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Plot feature maps
    for idx in range(num_maps):
        ax = axes[idx]
        feature_map = feature_maps[idx]
        
        # Normalize for better visualization
        vmin, vmax = feature_map.min(), feature_map.max()
        if vmax - vmin > 0:
            feature_map = (feature_map - vmin) / (vmax - vmin)
        
        ax.imshow(feature_map, cmap='viridis')
        ax.set_title(f"Map {idx}", fontsize=8)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_maps, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature maps saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def extract_and_visualize_features(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_name: str = 'layer4',
    num_maps: int = 16,
    save_path: Optional[str] = None
):
    """
    Extract and visualize features from a specific layer.
    
    Args:
        model (nn.Module): Model to extract features from
        input_tensor (torch.Tensor): Input image [1, 3, 224, 224]
        layer_name (str): Name of layer to extract features from
        num_maps (int): Number of feature maps to visualize
        save_path (str, optional): Path to save figure
    
    Returns:
        torch.Tensor: Extracted features
    
    Example:
        >>> model = get_resnet_model('resnet18', num_classes=10)
        >>> image = torch.randn(1, 3, 224, 224)
        >>> features = extract_and_visualize_features(
        ...     model, image, layer_name='layer4',
        ...     save_path='visualizations/features.png'
        ... )
    """
    model.eval()
    features = {}
    
    # Register hook
    def hook_fn(module, input, output):
        features['output'] = output.detach()
    
    # Find layer and register hook
    target_layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_layer = module
            break
    
    if target_layer is None:
        raise ValueError(f"Layer {layer_name} not found in model")
    
    handle = target_layer.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Remove hook
    handle.remove()
    
    # Visualize
    if 'output' in features:
        visualize_feature_maps(
            features['output'],
            num_maps=num_maps,
            title=f"{layer_name} Feature Maps",
            save_path=save_path
        )
        return features['output']
    else:
        print(f"Warning: No features extracted from {layer_name}")
        return None


def save_feature_maps(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    layer_name: str = 'layer4',
    save_dir: str = './visualizations',
    num_samples: int = 5,
    num_maps: int = 16,
    device: str = 'cuda'
):
    """
    Save feature map visualizations for multiple samples.
    
    Args:
        model (nn.Module): Model
        data_loader (DataLoader): DataLoader
        layer_name (str): Layer to visualize
        save_dir (str): Directory to save visualizations
        num_samples (int): Number of samples to visualize
        num_maps (int): Number of feature maps per sample
        device (str): Device
    
    Example:
        >>> save_feature_maps(
        ...     model=model,
        ...     data_loader=val_loader,
        ...     layer_name='layer4',
        ...     save_dir='visualizations/resnet18',
        ...     num_samples=5
        ... )
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for images, labels in data_loader:
        if count >= num_samples:
            break
        
        image = images[0:1].to(device)  # Take first image
        label = labels[0].item()
        
        save_path = save_dir / f"sample_{count}_class_{label}.png"
        
        extract_and_visualize_features(
            model=model,
            input_tensor=image,
            layer_name=layer_name,
            num_maps=num_maps,
            save_path=str(save_path)
        )
        
        count += 1
    
    print(f"\nSaved {count} feature map visualizations to {save_dir}")


if __name__ == "__main__":
    # Test feature map visualization
    print("Testing CNN Feature Map Visualization...\n")
    
    # Create dummy feature maps
    feature_maps = torch.randn(1, 64, 28, 28)  # [B, C, H, W]
    
    print("[1/2] Testing visualize_feature_maps()...")
    visualize_feature_maps(
        feature_maps,
        num_maps=16,
        title="Test Feature Maps",
        save_path="./test_feature_maps.png"
    )
    
    print("\n[2/2] Testing with real model...")
    from ..models.cnn.resnet import get_resnet_model
    
    model = get_resnet_model('resnet18', num_classes=10, pretrained=False)
    model.eval()
    
    # Random input
    input_image = torch.randn(1, 3, 224, 224)
    
    # Extract and visualize
    features = extract_and_visualize_features(
        model=model,
        input_tensor=input_image,
        layer_name='layer4',
        num_maps=16,
        save_path="./test_resnet_features.png"
    )
    
    print(f"\nExtracted features shape: {features.shape}")
    print("\n✓ All tests passed!")
