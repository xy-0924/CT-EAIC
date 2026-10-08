#!/usr/bin/env python3
"""
Ordinal-aware Cross Entropy Loss for ordered categories (e.g. LR grades).

LR grades have a natural ordering: LR-1/2 < LR-3 < LR-4 < LR-5 < LR-M.
Standard CE treats all misclassifications equally, but predicting LR-1/2 when
the true label is LR-3 is a less severe error than predicting LR-5.

This module provides:
1. OrdinalCrossEntropyLoss: CE + soft-label smoothing towards adjacent classes
2. LabelDifferenceLoss: penalizes predictions far from the true class in ordinal space

Both support optional class_weights (sqrt-scaled, safe range) for imbalanced datasets.

Usage:
    # In train.py, replace standard CE with ordinal loss:
    loss_fn = OrdinalCrossEntropyLoss(num_classes=5, alpha=0.2, class_weights=[0.76, 1.53, 1.57, 0.72, 1.42])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalCrossEntropyLoss(nn.Module):
    """Ordinal-aware Cross Entropy: standard CE + soft-label smoothing towards adjacent classes.

    For a sample with true class c, the soft target distributes probability:
      - (1 - alpha) to class c
      - alpha / 2 to class c-1 (if exists)
      - alpha / 2 to class c+1 (if exists)

    This teaches the model that adjacent classes are "closer" and reduces the
    penalty for predicting LR-4 when true is LR-3 (vs predicting LR-1/2).

    Args:
        num_classes: number of ordered classes (default: 5 for LR grades)
        alpha: smoothing strength (0 = pure CE, 1 = fully smoothed). Recommended: 0.1~0.3
        class_weights: optional list of per-class weights (sqrt-scaled, safe range).
                       Pass None for uniform weights (default behavior).
    """

    def __init__(self, num_classes=5, alpha=0.2, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha

        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', w)
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) raw model outputs (before softmax)
            targets: (B,) integer class labels in [0, num_classes-1]
        Returns:
            scalar loss
        """
        # 1. Standard cross-entropy (with optional class weights)
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')

        # 2. Ordinal soft-label component
        if self.alpha > 0:
            with torch.no_grad():
                soft_targets = torch.zeros_like(logits)
                B = logits.size(0)
                C = self.num_classes
                for i in range(B):
                    c = targets[i].item()
                    # Distribute probability to self and immediate neighbors
                    soft_targets[i, c] = 1.0 - self.alpha
                    neighbors = []
                    if c > 0:
                        neighbors.append(c - 1)
                    if c < C - 1:
                        neighbors.append(c + 1)
                    if neighbors:
                        share = self.alpha / len(neighbors)
                        for n in neighbors:
                            soft_targets[i, n] = share

            # KL-divergence style: -sum(soft * log_softmax(pred))
            log_probs = F.log_softmax(logits, dim=-1)
            ordinal_loss = -(soft_targets * log_probs).sum(dim=-1)

            # Apply class weights to ordinal component too
            if self.class_weights is not None:
                w = self.class_weights[targets]
                ordinal_loss = ordinal_loss * w

            loss = (1.0 - self.alpha) * ce_loss + self.alpha * ordinal_loss
        else:
            loss = ce_loss

        return loss.mean()


class LabelDifferenceLoss(nn.Module):
    """Penalizes predictions based on ordinal distance from true class.

    Loss = CE_loss + lambda * mean(|pred_class - true_class|)

    The ordinal penalty encourages the model to at least predict a class
    close to the true one, even when it's wrong.

    Args:
        num_classes: number of ordered classes
        lam: ordinal penalty weight (recommended: 0.1~0.5)
        class_weights: optional per-class weights for CE component
    """

    def __init__(self, num_classes=5, lam=0.2, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.lam = lam
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32)
                                       if class_weights else None)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)

        # Expected ordinal distance: sum over all classes of |c - true| * p(c)
        with torch.no_grad():
            idx = torch.arange(self.num_classes, device=logits.device, dtype=logits.dtype)
            # (B, C) distance matrix
            dist = torch.abs(idx.unsqueeze(0) - targets.unsqueeze(1).float())

        probs = F.softmax(logits, dim=-1)
        ordinal_penalty = (probs * dist).sum(dim=-1).mean()

        return ce_loss + self.lam * ordinal_penalty


class FocalOrdinalLoss(nn.Module):
    """Focal Loss combined with ordinal soft-label smoothing.

    Addresses two issues simultaneously:
    1. Class imbalance (focal component: down-weights easy samples)
    2. Ordinal structure (soft-label: teaches that adjacent classes are closer)

    Args:
        num_classes: number of ordered classes
        gamma: focal exponent (higher = more focus on hard samples). Recommended: 1.0~2.0
        alpha_smooth: ordinal soft-label strength. Recommended: 0.1~0.3
        class_weights: optional per-class weights
    """

    def __init__(self, num_classes=5, gamma=1.5, alpha_smooth=0.2, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha_smooth = alpha_smooth

        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', w)
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        # Focal weight: (1 - p_true)^gamma
        p_true = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - p_true) ** self.gamma

        # Standard CE with focal weighting
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')
        focal_ce = focal_weight * ce_loss

        # Ordinal soft-label component
        if self.alpha_smooth > 0:
            with torch.no_grad():
                soft_targets = torch.zeros_like(logits)
                B, C = logits.size(0), self.num_classes
                for i in range(B):
                    c = targets[i].item()
                    soft_targets[i, c] = 1.0 - self.alpha_smooth
                    neighbors = []
                    if c > 0:
                        neighbors.append(c - 1)
                    if c < C - 1:
                        neighbors.append(c + 1)
                    if neighbors:
                        share = self.alpha_smooth / len(neighbors)
                        for n in neighbors:
                            soft_targets[i, n] = share

            log_probs = F.log_softmax(logits, dim=-1)
            ordinal_loss = -(soft_targets * log_probs).sum(dim=-1)

            if self.class_weights is not None:
                w = self.class_weights[targets]
                ordinal_loss = ordinal_loss * w

            loss = (1.0 - self.alpha_smooth) * focal_ce + self.alpha_smooth * ordinal_loss
        else:
            loss = focal_ce

        return loss.mean()
