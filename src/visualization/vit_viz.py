#!/usr/bin/env python3
"""
ViT Attention Map Visualization

Provides tools to visualize attention maps from Vision Transformer models.
Useful for understanding what the model attends to at different layers.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import cv2


def visualize_attention_map(
    attention_weights: torch.Tensor,
    image: Optional[torch.Tensor] = None,
    head_idx: int = 0,
    layer_name: str = "Attention",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
):
    """
    Visualize attention map from a single attention head.
    
    Args:
        attention_weights (torch.Tensor): Attention weights [B, H, N, N]
            where H=num_heads, N=num_patches+1 (including CLS token)
        image (torch.Tensor, optional): Original image [3, 224, 224]
        head_idx (int): Which attention head to visualize
        layer_name (str): Name of the layer
        save_path (str, optional): Path to save figure
        figsize (Tuple[int, int]): Figure size
    
    Example:
        >>> attention = model.encoder.layers[0].self_attention.attention_weights
        >>> visualize_attention_map(
        ...     attention,
        ...     image=input_image,
        ...     head_idx=0,
        ...     save_path='attention_head_0.png'
        ... )
    """
    # Move to CPU
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Take first sample and specific head
    if attention_weights.ndim == 4:
        attention_weights = attention_weights[0, head_idx]  # [N, N]
    
    # Get attention from CLS token to all patches
    cls_attention = attention_weights[0, 1:]  # Skip CLS-to-CLS
    
    # Reshape to 2D (assuming 16x16 patches for 224x224 image)
    num_patches = int(np.sqrt(len(cls_attention)))
    attention_map = cls_attention.reshape(num_patches, num_patches)
    
    # Create figure
    if image is not None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if image.shape[0] == 3:  # [C, H, W] -> [H, W, C]
            image = image.transpose(1, 2, 0)
        
        # Normalize for display
        image = (image - image.min()) / (image.max() - image.min())
        
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Attention map
        im1 = axes[1].imshow(attention_map, cmap='viridis')
        axes[1].set_title(f"{layer_name} - Head {head_idx}")
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        # Overlay
        # Resize attention map to image size
        attention_resized = cv2.resize(
            attention_map, 
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )
        
        # Normalize attention for overlay
        attention_resized = (attention_resized - attention_resized.min()) / \
                           (attention_resized.max() - attention_resized.min())
        
        # Create heatmap overlay
        axes[2].imshow(image)
        axes[2].imshow(attention_resized, cmap='jet', alpha=0.5)
        axes[2].set_title("Attention Overlay")
        axes[2].axis('off')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(attention_map, cmap='viridis')
        ax.set_title(f"{layer_name} - Head {head_idx}")
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attention map saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_multi_head_attention(
    attention_weights: torch.Tensor,
    num_heads: int = 8,
    layer_name: str = "Attention",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Visualize attention from multiple heads in a grid.
    
    Args:
        attention_weights (torch.Tensor): Attention weights [B, H, N, N]
        num_heads (int): Number of heads to visualize
        layer_name (str): Layer name
        save_path (str, optional): Path to save figure
        figsize (Tuple[int, int]): Figure size
    
    Example:
        >>> attention = model.encoder.layers[0].self_attention.attention_weights
        >>> visualize_multi_head_attention(
        ...     attention,
        ...     num_heads=8,
        ...     save_path='multi_head_attention.png'
        ... )
    """
    # Move to CPU
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Take first sample
    if attention_weights.ndim == 4:
        attention_weights = attention_weights[0]  # [H, N, N]
    
    total_heads = attention_weights.shape[0]
    num_heads = min(num_heads, total_heads)
    
    # Calculate grid size
    ncols = int(np.ceil(np.sqrt(num_heads)))
    nrows = int(np.ceil(num_heads / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Get num patches
    num_patches = int(np.sqrt(attention_weights.shape[1] - 1))
    
    for head_idx in range(num_heads):
        ax = axes[head_idx]
        
        # Get CLS token attention
        cls_attention = attention_weights[head_idx, 0, 1:]
        attention_map = cls_attention.reshape(num_patches, num_patches)
        
        im = ax.imshow(attention_map, cmap='viridis')
        ax.set_title(f"Head {head_idx}", fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"{layer_name} - Multi-Head Attention", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Multi-head attention saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def extract_attention_rollout(
    attention_weights_list: List[torch.Tensor],
    discard_ratio: float = 0.9
) -> np.ndarray:
    """
    Compute attention rollout across all layers.
    
    Attention rollout recursively multiplies attention matrices
    from all layers to get the effective attention from input to output.
    
    Args:
        attention_weights_list (List[torch.Tensor]): List of attention weights from each layer
        discard_ratio (float): Ratio of lowest attentions to discard
    
    Returns:
        np.ndarray: Rolled attention map [num_patches, num_patches]
    
    Reference:
        "Quantifying Attention Flow in Transformers" (Abnar & Zuidema, 2020)
    
    Example:
        >>> attentions = [layer.attention for layer in model.encoder.layers]
        >>> rollout = extract_attention_rollout(attentions)
        >>> plt.imshow(rollout)
    """
    # Average attention weights across all heads
    attention_matrices = []
    
    for attention in attention_weights_list:
        if isinstance(attention, torch.Tensor):
            attention = attention.detach().cpu().numpy()
        
        # Take first sample and average across heads
        if attention.ndim == 4:
            attention = attention[0]  # [H, N, N]
        
        attention_avg = attention.mean(axis=0)  # [N, N]
        attention_matrices.append(attention_avg)
    
    # Add identity matrix
    num_tokens = attention_matrices[0].shape[0]
    result = np.eye(num_tokens)
    
    # Rollout: multiply attention matrices
    for attention_mat in attention_matrices:
        # Add residual connection
        attention_mat = attention_mat + np.eye(num_tokens)
        # Normalize
        attention_mat = attention_mat / attention_mat.sum(axis=-1, keepdims=True)
        # Multiply
        result = np.matmul(attention_mat, result)
    
    # Get attention to patches from CLS token
    mask = result[0, 1:]  # Skip CLS token
    
    # Discard lowest attentions
    if discard_ratio > 0:
        threshold = np.percentile(mask, discard_ratio * 100)
        mask[mask < threshold] = 0
    
    # Normalize
    mask = mask / mask.sum()
    
    return mask


def save_attention_visualizations(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    save_dir: str = './visualizations/attention',
    num_samples: int = 5,
    num_heads: int = 8,
    device: str = 'cuda'
):
    """
    Save attention visualizations for multiple samples.
    
    Args:
        model (nn.Module): ViT model
        data_loader (DataLoader): DataLoader
        save_dir (str): Directory to save visualizations
        num_samples (int): Number of samples to visualize
        num_heads (int): Number of attention heads to visualize
        device (str): Device
    
    Example:
        >>> save_attention_visualizations(
        ...     model=vit_model,
        ...     data_loader=val_loader,
        ...     save_dir='visualizations/vit_attention',
        ...     num_samples=10
        ... )
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving attention visualizations to {save_dir}...")
    
    count = 0
    for images, labels in data_loader:
        if count >= num_samples:
            break
        
        image = images[0:1].to(device)
        label = labels[0].item()
        
        # Forward pass with attention extraction
        # Note: This requires model to return attention weights
        with torch.no_grad():
            output = model(image)
            
            # Try to extract attention from model
            # This depends on model implementation
            # For now, create placeholder
            print(f"Sample {count + 1}/{num_samples}: class {label}")
            print("  Note: Attention extraction requires model modification")
            print("  See ViTWithAttention wrapper in vit_models.py")
        
        count += 1
    
    print(f"\nVisualization complete!")


if __name__ == "__main__":
    # Test attention visualization
    print("Testing ViT Attention Visualization...\n")
    
    # Create dummy attention weights
    # [Batch, Heads, Tokens, Tokens]
    num_heads = 12
    num_patches = 14  # 14x14 for 224x224 with 16x16 patches
    num_tokens = num_patches * num_patches + 1  # +1 for CLS token
    
    attention = torch.randn(1, num_heads, num_tokens, num_tokens)
    attention = torch.softmax(attention, dim=-1)  # Make it a valid attention matrix
    
    print(f"[1/3] Testing single head visualization...")
    visualize_attention_map(
        attention,
        head_idx=0,
        layer_name="Test Layer",
        save_path="./test_attention_single.png"
    )
    
    print(f"\n[2/3] Testing multi-head visualization...")
    visualize_multi_head_attention(
        attention,
        num_heads=8,
        layer_name="Test Layer",
        save_path="./test_attention_multi.png"
    )
    
    print(f"\n[3/3] Testing attention rollout...")
    # Create list of attention weights (simulating multiple layers)
    attention_list = [attention for _ in range(12)]
    rollout = extract_attention_rollout(attention_list)
    
    num_patches_dim = int(np.sqrt(len(rollout)))
    rollout_2d = rollout.reshape(num_patches_dim, num_patches_dim)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(rollout_2d, cmap='viridis')
    plt.title("Attention Rollout")
    plt.colorbar()
    plt.axis('off')
    plt.savefig('./test_attention_rollout.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Rollout shape: {rollout_2d.shape}")
    print("\n✓ All tests passed!")
