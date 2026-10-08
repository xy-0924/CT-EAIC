import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.utilities.helpers import softmax_helper_dim1


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in segmentation.
    
    Focuses learning on hard examples and down-weights easy negatives.
    
    Args:
        gamma: Focusing parameter (default: 2.0)
               Higher gamma = more focus on hard examples
        alpha: Balance parameter (default: 0.25)
               Can be float or list of floats per class
        apply_nonlin: Nonlinearity to apply to input (default: softmax)
        smooth: Smoothing factor to avoid log(0) (default: 1e-5)
    """
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, 
                 apply_nonlin=softmax_helper_dim1, smooth: float = 1e-5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W) or (B, C, H, W, D)
            y: Target tensor of shape (B, H, W) or (B, H, W, D) with class indices
        
        Returns:
            Focal loss value
        """
        # Apply nonlinearity (softmax)
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        
        # Add smoothing to avoid log(0)
        x = x.clamp(min=self.smooth, max=1.0 - self.smooth)
        
        # Convert target to one-hot encoding
        num_classes = x.shape[1]
        y_onehot = torch.zeros_like(x)
        y_onehot.scatter_(1, y.unsqueeze(1), 1)
        
        # Calculate focal loss
        # pt: probability of true class
        pt = (x * y_onehot).sum(dim=1)  # (B, H, W) or (B, H, W, D)
        
        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Cross entropy: -log(pt)
        ce_loss = -torch.log(pt)
        
        # Focal loss: alpha * (1 - pt)^gamma * CE
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return focal_loss.mean()


class FocalDiceLoss(nn.Module):
    """
    Combined Focal + Dice Loss for small object segmentation.
    
    This combines the benefits of:
    - Focal Loss: Handles class imbalance, focuses on hard examples
    - Dice Loss: Directly optimizes for overlap metric
    """
    
    def __init__(self, focal_weight: float = 0.5, dice_weight: float = 0.5,
                 gamma: float = 2.0, alpha: float = 0.25,
                 batch_dice: bool = False, smooth: float = 1e-5):
        super(FocalDiceLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        
        # Import dice loss
        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
        self.dice = MemoryEfficientSoftDiceLoss(batch=batch_dice, smooth=smooth)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(x, y)
        dice_loss = self.dice(x, y)
        
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss
