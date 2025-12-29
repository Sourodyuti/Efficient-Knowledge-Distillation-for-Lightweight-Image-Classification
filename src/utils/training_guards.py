#!/usr/bin/env python3
"""
Training Safety Guards

Provides critical safety checks for training:
- Teacher model freezing
- Gradient checkpointing enforcement
- NaN/Inf detection
- Memory cleanup
- Loss validation
"""

import torch
import torch.nn as nn
import gc
from typing import Optional


def freeze_teacher(teacher_model: nn.Module, verbose: bool = True) -> None:
    """
    **CRITICAL**: Freeze teacher model for distillation.
    
    This function MUST be called before assistant/student training to:
    - Set model to eval mode
    - Disable gradient computation
    - Free VRAM from gradient graphs
    
    Args:
        teacher_model (nn.Module): Teacher model
        verbose (bool): Print status
    
    Example:
        >>> teacher = get_resnet_model('resnet50', num_classes=10)
        >>> freeze_teacher(teacher)
        >>> # Now safe to use for distillation
    """
    if teacher_model is None:
        return
    
    # Set to eval mode
    teacher_model.eval()
    
    # Freeze all parameters
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    if verbose:
        total_params = sum(p.numel() for p in teacher_model.parameters())
        frozen_params = sum(p.numel() for p in teacher_model.parameters() if not p.requires_grad)
        print(f"\n❄️  Teacher frozen: {frozen_params:,}/{total_params:,} parameters")
        print(f"   Mode: eval(), requires_grad=False")
        print(f"   VRAM saved: ~{(frozen_params * 4 / 1024**2):.1f} MB (FP32)")


def enforce_gradient_checkpointing(model: nn.Module, model_name: str = "", verbose: bool = True) -> bool:
    """
    **CRITICAL for ViT-Base on 6GB VRAM**: Enforce gradient checkpointing.
    
    Gradient checkpointing saves ~40% memory by recomputing activations
    during backward pass instead of storing them.
    
    Args:
        model (nn.Module): Model to enable checkpointing
        model_name (str): Model name for selective enforcement
        verbose (bool): Print status
    
    Returns:
        bool: True if checkpointing was enabled
    
    Example:
        >>> model = get_vit_model('vit_base', num_classes=10)
        >>> enforce_gradient_checkpointing(model, 'vit_base')
    """
    # Only enforce for ViT-Base (most memory-intensive)
    if 'vit_base' not in model_name.lower() and 'base' not in model_name.lower():
        return False
    
    enabled = False
    
    # Try multiple methods (different model implementations)
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        enabled = True
    elif hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
        enabled = True
    elif hasattr(model, 'set_grad_checkpointing'):
        model.set_grad_checkpointing(True)
        enabled = True
    
    if verbose and enabled:
        print(f"\n✅ Gradient checkpointing enforced for {model_name}")
        print(f"   Memory savings: ~40%")
        print(f"   Speed impact: ~30% slower (acceptable trade-off)")
    elif verbose and not enabled:
        print(f"\n⚠️  WARNING: Could not enable gradient checkpointing for {model_name}")
        print(f"   This may cause OOM on 6GB VRAM!")
    
    return enabled


def cleanup_training_state(
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    verbose: bool = True
) -> None:
    """
    **CRITICAL**: Fully cleanup training state between models.
    
    Deleting model alone is NOT enough. Optimizer, scheduler, and AMP scaler
    hold references that prevent full memory release.
    
    Args:
        model (nn.Module, optional): Model to delete
        optimizer (Optimizer, optional): Optimizer to delete
        scheduler (LRScheduler, optional): Scheduler to delete
        scaler (GradScaler, optional): AMP scaler to delete
        verbose (bool): Print cleanup status
    
    Example:
        >>> # After teacher training, before assistant training:
        >>> cleanup_training_state(
        ...     model=teacher_model,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     scaler=scaler
        ... )
    """
    if verbose:
        print("\n🧹 Cleaning up training state...")
    
    # Delete components
    components = {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'scaler': scaler
    }
    
    for name, component in components.items():
        if component is not None:
            del component
            if verbose:
                print(f"   ✓ Deleted {name}")
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if verbose:
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"   ✓ CUDA cache cleared")
            print(f"   Current memory: {memory_allocated:.1f} MB allocated, {memory_reserved:.1f} MB reserved")


def validate_loss(loss: torch.Tensor, step: int, raise_on_invalid: bool = True) -> bool:
    """
    **CRITICAL**: Validate loss for NaN/Inf values.
    
    Training should STOP immediately on invalid loss to avoid wasting compute.
    
    Args:
        loss (torch.Tensor): Loss value to validate
        step (int): Current training step (for error message)
        raise_on_invalid (bool): Raise RuntimeError on invalid loss
    
    Returns:
        bool: True if loss is valid
    
    Raises:
        RuntimeError: If loss is NaN/Inf and raise_on_invalid=True
    
    Example:
        >>> loss = criterion(output, target)
        >>> validate_loss(loss, step=batch_idx)
        >>> # Safe to proceed with backward pass
    """
    is_valid = torch.isfinite(loss).all().item()
    
    if not is_valid:
        error_msg = (
            f"\n❌ NON-FINITE LOSS DETECTED at step {step}\n"
            f"   Loss value: {loss.item() if loss.numel() == 1 else loss}\n"
            f"   This indicates:\n"
            f"   - Learning rate too high\n"
            f"   - Gradient explosion\n"
            f"   - Numerical instability\n"
            f"   - Bad initialization\n"
            f"   Action: Reduce learning rate or check data"
        )
        
        if raise_on_invalid:
            raise RuntimeError(error_msg)
        else:
            print(error_msg)
    
    return is_valid


def reset_peak_memory_stats(device: Optional[torch.device] = None) -> None:
    """
    Reset peak memory statistics for accurate per-epoch tracking.
    
    torch.cuda.max_memory_allocated() is cumulative by default.
    Call this at the start of each epoch for correct logging.
    
    Args:
        device (torch.device, optional): CUDA device
    
    Example:
        >>> for epoch in range(epochs):
        ...     reset_peak_memory_stats()
        ...     # Train epoch
        ...     peak_memory = torch.cuda.max_memory_allocated()
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)


def schedule_kd_alpha(epoch: int, max_epochs: int, alpha_start: float = 0.9, alpha_end: float = 0.5) -> float:
    """
    **RECOMMENDED**: Schedule KD loss weight (alpha) during training.
    
    Best practice:
    - More KD (soft targets) early: Learn from teacher
    - More CE (hard targets) later: Learn to be confident
    
    Args:
        epoch (int): Current epoch (1-indexed)
        max_epochs (int): Total epochs
        alpha_start (float): Initial alpha (high = more KD)
        alpha_end (float): Final alpha (low = more CE)
    
    Returns:
        float: Alpha value for current epoch
    
    Example:
        >>> for epoch in range(1, epochs + 1):
        ...     alpha = schedule_kd_alpha(epoch, epochs)
        ...     kd_loss = kd_criterion(output, teacher_output, target, alpha=alpha)
    """
    progress = (epoch - 1) / max_epochs
    alpha = alpha_start - (alpha_start - alpha_end) * progress
    return max(alpha_end, alpha)  # Ensure doesn't go below minimum


def get_model_specific_lr(model_name: str, base_lr: float = 0.001) -> float:
    """
    **RECOMMENDED**: Get model-specific learning rate.
    
    Smaller models often need lower learning rates:
    - Teacher (large): Higher LR
    - Assistant (medium): Medium LR
    - Student (small): Lower LR
    
    Args:
        model_name (str): Model name
        base_lr (float): Base learning rate
    
    Returns:
        float: Adjusted learning rate
    
    Example:
        >>> teacher_lr = get_model_specific_lr('resnet50', base_lr=0.001)  # 0.001
        >>> student_lr = get_model_specific_lr('resnet18', base_lr=0.001)  # 0.0005
    """
    model_name_lower = model_name.lower()
    
    # Teacher models (large)
    if 'resnet50' in model_name_lower or 'vit_base' in model_name_lower or 'base' in model_name_lower:
        return base_lr
    
    # Assistant models (medium)
    elif 'resnet34' in model_name_lower or 'vit_small' in model_name_lower or 'small' in model_name_lower:
        return base_lr * 0.75
    
    # Student models (small)
    elif 'resnet18' in model_name_lower or 'vit_tiny' in model_name_lower or 'tiny' in model_name_lower:
        return base_lr * 0.5
    
    # Default
    return base_lr


if __name__ == "__main__":
    # Test safety guards
    print("Testing training safety guards...\n")
    
    # Test 1: Teacher freezing
    print("[1/4] Testing teacher freezing...")
    dummy_model = nn.Linear(10, 10)
    for p in dummy_model.parameters():
        p.requires_grad = True
    
    freeze_teacher(dummy_model)
    assert all(not p.requires_grad for p in dummy_model.parameters())
    print("✓ Teacher freezing works\n")
    
    # Test 2: Loss validation
    print("[2/4] Testing loss validation...")
    valid_loss = torch.tensor(0.5)
    assert validate_loss(valid_loss, step=1, raise_on_invalid=False)
    print("✓ Valid loss passes\n")
    
    # Test 3: NaN detection
    print("[3/4] Testing NaN detection...")
    nan_loss = torch.tensor(float('nan'))
    try:
        validate_loss(nan_loss, step=1, raise_on_invalid=True)
        assert False, "Should have raised error"
    except RuntimeError as e:
        print(f"✓ NaN detected and caught: {str(e)[:50]}...\n")
    
    # Test 4: Alpha scheduling
    print("[4/4] Testing alpha scheduling...")
    alphas = [schedule_kd_alpha(e, 100) for e in [1, 25, 50, 75, 100]]
    print(f"Alpha schedule (epoch 1, 25, 50, 75, 100): {alphas}")
    assert alphas[0] > alphas[-1], "Alpha should decrease"
    print("✓ Alpha scheduling works\n")
    
    print("✅ All safety guards working!")
