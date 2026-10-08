"""
Focal Loss for multi-class classification with class imbalance.

Focuses training on hard, misclassified examples by down-weighting
well-classified samples. Combined with optional class_weights for
synergy with sampling-level balancing strategies.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with optional class weights.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: focusing parameter (default: 2.0).
               gamma=0 reduces to standard cross-entropy.
               Higher gamma = stronger focus on hard examples.
               Recommended range: 1.0~3.0.
        class_weights: optional per-class weights (list or tensor).
                       Applied as alpha_t in the focal loss formula.
        reduction: 'mean' or 'sum' (default: 'mean').
    """

    def __init__(self, gamma=2.0, class_weights=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', w)
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) raw model outputs (before softmax).
            targets: (B,) integer class labels in [0, C-1].
        Returns:
            scalar loss.
        """
        # Compute cross-entropy per sample (no reduction)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Compute probabilities for focal term
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # p_t = prob of true class

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Apply focal weighting
        loss = focal_weight * ce_loss

        # Apply class weights (alpha_t)
        if self.class_weights is not None:
            alpha_t = self.class_weights[targets]
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
