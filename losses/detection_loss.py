"""
Detection Loss Functions for Object Detection
==============================================

Focal Loss (classification) + Smooth L1 (bounding box regression).

IMPORTANT: Uses sigmoid focal loss (standard RetinaNet) rather than softmax.
Each anchor has K binary classifiers (one per class). Background anchors
have all-zero targets. This avoids needing a dedicated background class
and is the standard approach for RetinaNet/FCOS-style detectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Sigmoid Focal Loss for anchor-based object detection.

    Uses per-class sigmoid (not softmax), which is the standard RetinaNet approach.
    Each anchor has K binary classifiers. For positive anchors, the target is a
    one-hot vector. For negative (background) anchors, the target is all zeros.

    Args:
        alpha: Weighting factor for positive class (default 0.25)
        gamma: Focusing parameter (default 2.0)

    Input:
        inputs: (N, K) raw logits, K = num_classes
        targets: (N,) with values:
            0 = background (all classes negative)
            1..K = object class (one-hot encoding applied internally)
            -1 = ignore (excluded from loss)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute sigmoid focal loss.

        Args:
            inputs: (N, K) raw logits
            targets: (N,) class indices. 0=bg, 1..K=classes, -1=ignore
        """
        num_classes = inputs.shape[-1]

        # Filter out ignored anchors
        valid_mask = targets >= 0
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]

        if inputs.numel() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        # Create one-hot targets: background (0) -> all zeros, class k -> index k-1
        target_onehot = torch.zeros_like(inputs)
        pos_mask = targets > 0
        if pos_mask.any():
            # targets are 1-indexed (1..K), convert to 0-indexed for scatter
            # .to(target_onehot.dtype) ensures AMP compatibility (float16/float32)
            target_onehot[pos_mask] = F.one_hot(
                targets[pos_mask] - 1, num_classes
            ).to(target_onehot.dtype)

        # Sigmoid activation
        p = torch.sigmoid(inputs)

        # Compute focal loss per element
        # For positive targets (t=1): pt = p, alpha_t = alpha
        # For negative targets (t=0): pt = 1-p, alpha_t = 1-alpha
        pt = torch.where(target_onehot == 1, p, 1 - p)
        alpha_t = torch.where(target_onehot == 1, self.alpha, 1 - self.alpha)

        # Focal term
        focal_weight = (1 - pt) ** self.gamma

        # Binary cross-entropy (numerically stable)
        bce = F.binary_cross_entropy_with_logits(inputs, target_onehot, reduction='none')

        loss = alpha_t * focal_weight * bce

        # Normalize by number of positive anchors (standard RetinaNet)
        num_pos = pos_mask.sum().clamp(min=1)
        return loss.sum() / num_pos


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss for bounding box regression.

    Only computed for positive anchors (those matched to a GT box).

    Args:
        beta: Threshold for switching from quadratic to linear (default 1/9)
    """

    def __init__(self, beta: float = 1.0 / 9.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (N, 4) predicted deltas for positive anchors
            target: (N, 4) encoded GT deltas for positive anchors
        """
        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        return loss.mean()


class DetectionLoss(nn.Module):
    """
    Combined Detection Loss = Focal Loss + Smooth L1 Loss.

    Args:
        num_classes: Number of object classes (15 for DOTA)
        cls_weight: Weight for classification loss
        reg_weight: Weight for regression loss (typically higher to balance scales)
    """

    def __init__(self, num_classes=16, cls_weight=1.0, reg_weight=50.0):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.smooth_l1 = SmoothL1Loss(beta=1.0 / 9.0)
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

    def forward(self, cls_logits, cls_targets, bbox_preds, bbox_targets):
        loss_cls = self.focal_loss(cls_logits, cls_targets)
        loss_reg = self.smooth_l1(bbox_preds, bbox_targets)
        loss_total = self.cls_weight * loss_cls + self.reg_weight * loss_reg
        return {'loss_cls': loss_cls, 'loss_reg': loss_reg, 'loss_det': loss_total}


if __name__ == "__main__":
    print("Testing Sigmoid Focal Loss...")

    # Simulate: 100 anchors, 15 classes
    logits = torch.randn(100, 15)
    # 10 positives (classes 1-15), 90 backgrounds (0)
    targets = torch.zeros(100, dtype=torch.long)
    targets[:10] = torch.randint(1, 16, (10,))

    fl = FocalLoss()
    loss = fl(logits, targets)
    print(f"Focal loss: {loss.item():.4f}")

    # Test with ignore
    targets[50:60] = -1
    loss = fl(logits, targets)
    print(f"Focal loss (with ignore): {loss.item():.4f}")

    # SmoothL1
    pred_boxes = torch.randn(10, 4)
    gt_boxes = torch.randn(10, 4)
    sl1 = SmoothL1Loss()
    loss_reg = sl1(pred_boxes, gt_boxes)
    print(f"SmoothL1 loss: {loss_reg.item():.4f}")

    print("✓ All loss tests passed!")
