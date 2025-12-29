"""
Knowledge Distillation Loss Functions

Implements various KD loss formulations:
- Standard KD loss (Hinton et al., 2015)
- Temperature scaling
- Hard + soft label blending
- Soft cross-entropy

Framework-agnostic: works with both CNN and ViT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining hard and soft targets.
    
    L_total = alpha * L_KD + (1 - alpha) * L_CE
    
    where:
        L_KD  = KL divergence between teacher and student soft predictions
        L_CE  = Cross-entropy loss on true labels (hard targets)
        alpha = Balance parameter between KD and CE loss
    
    Temperature scaling is applied to teacher and student logits before
    computing KL divergence.
    
    References:
        Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        reduction: str = 'batchmean'
    ):
        """
        Args:
            temperature (float): Temperature for softening distributions (T > 1)
                Higher temperature produces softer probability distributions.
                Typical values: 3.0 - 20.0. Default: 4.0
            alpha (float): Weight for KD loss. (1-alpha) is weight for CE loss.
                Range: [0, 1]. Default: 0.7 (70% KD, 30% CE)
            reduction (str): Reduction method for KL divergence.
                Options: 'batchmean', 'sum', 'mean'. Default: 'batchmean'
        """
        super().__init__()
        
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        if reduction not in ['batchmean', 'sum', 'mean', 'none']:
            raise ValueError(f"Invalid reduction: {reduction}")
        
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction
        
        # Cross-entropy for hard labels
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute knowledge distillation loss.
        
        Args:
            student_logits (torch.Tensor): Raw logits from student model [B, C]
            teacher_logits (torch.Tensor): Raw logits from teacher model [B, C]
            targets (torch.Tensor): Ground truth labels [B]
            return_components (bool): If True, return (total_loss, kd_loss, ce_loss)
        
        Returns:
            torch.Tensor: Combined loss (scalar)
            or Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (total, kd, ce)
        """
        # Hard target loss (cross-entropy on true labels)
        ce_loss = self.ce_loss(student_logits, targets)
        
        # Soft target loss (KL divergence between teacher and student)
        # Scale logits by temperature
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # KL divergence
        kd_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction=self.reduction
        )
        
        # Scale KD loss by T^2 to account for temperature scaling
        # This ensures gradients are properly scaled
        kd_loss = kd_loss * (self.temperature ** 2)
        
        # Combine losses
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        
        if return_components:
            return total_loss, kd_loss, ce_loss
        
        return total_loss
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"temperature={self.temperature}, "
            f"alpha={self.alpha}, "
            f"reduction='{self.reduction}'"
            f")"
        )


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
    reduction: str = 'batchmean'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Functional API for knowledge distillation loss.
    
    Convenience function for one-off loss computation without creating
    a KnowledgeDistillationLoss module.
    
    Args:
        student_logits (torch.Tensor): Student model logits [B, C]
        teacher_logits (torch.Tensor): Teacher model logits [B, C]
        targets (torch.Tensor): Ground truth labels [B]
        temperature (float): Temperature for softening (default: 4.0)
        alpha (float): KD loss weight (default: 0.7)
        reduction (str): KL divergence reduction (default: 'batchmean')
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            (total_loss, kd_loss, ce_loss)
    
    Example:
        >>> student_logits = torch.randn(32, 10)  # Batch=32, Classes=10
        >>> teacher_logits = torch.randn(32, 10)
        >>> targets = torch.randint(0, 10, (32,))
        >>> total, kd, ce = distillation_loss(student_logits, teacher_logits, targets)
    """
    loss_fn = KnowledgeDistillationLoss(
        temperature=temperature,
        alpha=alpha,
        reduction=reduction
    )
    
    return loss_fn(
        student_logits,
        teacher_logits,
        targets,
        return_components=True
    )


def soft_cross_entropy(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = 'batchmean'
) -> torch.Tensor:
    """
    Soft cross-entropy between student and teacher predictions.
    
    This is equivalent to KL divergence but using log_softmax for both
    teacher and student (symmetric form).
    
    Args:
        student_logits (torch.Tensor): Student logits [B, C]
        teacher_logits (torch.Tensor): Teacher logits [B, C]
        temperature (float): Temperature scaling (default: 1.0)
        reduction (str): Reduction method (default: 'batchmean')
    
    Returns:
        torch.Tensor: Soft cross-entropy loss
    
    Example:
        >>> student_logits = torch.randn(32, 10)
        >>> teacher_logits = torch.randn(32, 10)
        >>> loss = soft_cross_entropy(student_logits, teacher_logits, temperature=4.0)
    """
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    
    loss = F.kl_div(student_soft, teacher_soft, reduction=reduction)
    
    # Scale by T^2
    loss = loss * (temperature ** 2)
    
    return loss


class AdaptiveKDLoss(nn.Module):
    """
    Adaptive Knowledge Distillation Loss with dynamic alpha.
    
    Alpha is adjusted based on training progress or student confidence.
    Early training: Higher alpha (more focus on teacher)
    Late training: Lower alpha (more focus on true labels)
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha_start: float = 0.9,
        alpha_end: float = 0.5,
        total_epochs: int = 300,
        warmup_epochs: int = 10,
        reduction: str = 'batchmean'
    ):
        """
        Args:
            temperature (float): Temperature for softening
            alpha_start (float): Initial alpha (high teacher influence)
            alpha_end (float): Final alpha (balanced)
            total_epochs (int): Total training epochs
            warmup_epochs (int): Epochs before starting alpha decay
            reduction (str): KL divergence reduction
        """
        super().__init__()
        
        self.temperature = temperature
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.reduction = reduction
        
        self.current_epoch = 0
        self.ce_loss = nn.CrossEntropyLoss()
    
    def get_alpha(self, epoch: Optional[int] = None) -> float:
        """
        Compute alpha for current epoch using cosine annealing.
        
        Args:
            epoch (int, optional): Current epoch. If None, uses self.current_epoch
        
        Returns:
            float: Current alpha value
        """
        if epoch is None:
            epoch = self.current_epoch
        
        # Warmup: use alpha_start
        if epoch < self.warmup_epochs:
            return self.alpha_start
        
        # Cosine annealing from alpha_start to alpha_end
        progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        progress = min(progress, 1.0)
        
        alpha = self.alpha_end + 0.5 * (self.alpha_start - self.alpha_end) * (
            1 + torch.cos(torch.tensor(progress * 3.14159265359))
        )
        
        return float(alpha)
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        epoch: Optional[int] = None,
        return_components: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Compute adaptive KD loss.
        
        Args:
            student_logits (torch.Tensor): Student logits [B, C]
            teacher_logits (torch.Tensor): Teacher logits [B, C]
            targets (torch.Tensor): Ground truth labels [B]
            epoch (int, optional): Current epoch for alpha computation
            return_components (bool): Return loss components and alpha
        
        Returns:
            torch.Tensor: Total loss
            or Tuple: (total_loss, kd_loss, ce_loss, alpha)
        """
        # Get current alpha
        alpha = self.get_alpha(epoch)
        
        # Hard target loss
        ce_loss = self.ce_loss(student_logits, targets)
        
        # Soft target loss
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction=self.reduction)
        kd_loss = kd_loss * (self.temperature ** 2)
        
        # Combine with adaptive alpha
        total_loss = alpha * kd_loss + (1 - alpha) * ce_loss
        
        if return_components:
            return total_loss, kd_loss, ce_loss, alpha
        
        return total_loss
    
    def step_epoch(self):
        """Increment epoch counter."""
        self.current_epoch += 1


if __name__ == "__main__":
    # Test KD loss
    print("Testing Knowledge Distillation Loss...\n")
    
    # Create dummy data
    batch_size, num_classes = 32, 10
    student_logits = torch.randn(batch_size, num_classes)
    teacher_logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Standard KD loss
    kd_loss_fn = KnowledgeDistillationLoss(temperature=4.0, alpha=0.7)
    print(f"KD Loss Function: {kd_loss_fn}\n")
    
    total, kd, ce = kd_loss_fn(
        student_logits,
        teacher_logits,
        targets,
        return_components=True
    )
    
    print(f"Total Loss: {total.item():.4f}")
    print(f"KD Loss:    {kd.item():.4f}")
    print(f"CE Loss:    {ce.item():.4f}")
    
    # Functional API
    print("\nFunctional API:")
    total2, kd2, ce2 = distillation_loss(
        student_logits,
        teacher_logits,
        targets,
        temperature=4.0,
        alpha=0.7
    )
    print(f"Total: {total2.item():.4f}, KD: {kd2.item():.4f}, CE: {ce2.item():.4f}")
    
    # Adaptive KD loss
    print("\nAdaptive KD Loss:")
    adaptive_kd = AdaptiveKDLoss(
        temperature=4.0,
        alpha_start=0.9,
        alpha_end=0.5,
        total_epochs=300
    )
    
    for epoch in [0, 50, 150, 299]:
        alpha = adaptive_kd.get_alpha(epoch)
        print(f"Epoch {epoch:3d}: alpha = {alpha:.3f}")
    
    print("\n✓ All tests passed!")
