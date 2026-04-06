"""
Losses Package for Resolution-Agnostic Knowledge Distillation
==============================================================

This package contains all loss functions used during training:

1. detection_loss.py
   ├─ FocalLoss: Sigmoid focal loss with hard negative mining
   ├─ SmoothL1Loss: Regression loss for bounding box coordinates
   └─ DetectionLoss: Combined loss for object detection

   Used in:
   - Phase 1: Teacher training
   - Phase 2a: Student baseline training

2. distillation.py
   ├─ LogitKDLoss: KL divergence on temperature-scaled soft targets
   ├─ FeatureKDLoss: MSE between teacher/student FPN features
   └─ DistillationLoss: Combined α·L_det + β·L_logit + γ·L_feature

   Used in:
   - Phase 2b: Student with knowledge distillation training

Training Loss Composition:
==========================

Phase 1 & 2a (Standalone Training):
    L_total = L_detection = L_cls + L_reg

Phase 2b (Knowledge Distillation):
    L_total = α·L_detection + β·L_logit_kd + γ·L_feature_kd

    Typical weights:
    - α = 1.0 (detection: primary task)
    - β = 0.5 (logit KD: class confusion transfer)
    - γ = 1.0 (feature KD: spatial reasoning transfer)

    NOTE: γ may need to be higher (2.0-4.0) for cross-resolution KD
    because adapters perform semantic translation, not just resizing.
    See experiments.txt EXP-3C for the gamma sweep.
"""

from .detection_loss import (
    FocalLoss,
    SmoothL1Loss,
    DetectionLoss
)

from .distillation import (
    LogitKDLoss,
    FeatureKDLoss,
    DistillationLoss,
    create_distillation_loss
)

__all__ = [
    'FocalLoss',
    'SmoothL1Loss',
    'DetectionLoss',
    'LogitKDLoss',
    'FeatureKDLoss',
    'DistillationLoss',
    'create_distillation_loss',
]
