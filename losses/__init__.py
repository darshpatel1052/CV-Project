"""
Losses Package for Resolution-Agnostic Knowledge Distillation
==============================================================

This package contains all loss functions used during training:

1. detection_loss.py
   ├─ FocalLoss: Classification loss with hard negative mining
   ├─ SmoothL1Loss: Regression loss for bounding box coordinates
   └─ DetectionLoss: Combined loss for object detection
   
   Used in:
   - Phase 1: Teacher training
   - Phase 2a: Student baseline training

2. distillation.py (TO BE IMPLEMENTED)
   ├─ LogitKDLoss: Knowledge distillation on classification outputs
   ├─ FeatureKDLoss: Knowledge distillation on intermediate features
   └─ DistillationLoss: Combined KD losses
   
   Used in:
   - Phase 2b: Student with knowledge distillation training

Training Loss Composition:
==========================

Phase 1 & 2a (Standalone Training):
    L_total = L_detection
            = L_cls + L_reg
            
Phase 2b (Knowledge Distillation):
    L_total = α·L_detection + β·L_logit_kd + γ·L_feature_kd
    
    where:
    - L_detection: Object detection loss (classification + regression)
    - L_logit_kd: Soft target distillation from teacher classification
    - L_feature_kd: Feature map alignment between student and teacher
    
    Typical weights:
    - α = 1.0 (detection is primary)
    - β = 0.5 (soft targets provide additional supervision)
    - γ = 1.0 (feature alignment helps representation transfer)

Import Examples:
================

```python
# For Phase 1/2a training (detection only)
from losses.detection_loss import DetectionLoss
loss_fn = DetectionLoss(num_classes=15)

# For Phase 2b training (with KD) - coming soon
from losses.distillation import DistillationLoss
kd_loss_fn = DistillationLoss(num_classes=15, temperature=4.0)
```
"""

from .detection_loss import (
    FocalLoss,
    SmoothL1Loss,
    DetectionLoss
)

__all__ = [
    'FocalLoss',
    'SmoothL1Loss',
    'DetectionLoss',
]
