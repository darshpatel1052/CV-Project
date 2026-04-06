"""
Models Package
==============

- teacher.py: Swin-T + FPN + RetinaNet (high-capacity, HR)
- student.py: MobileNetV2 + FPN + Adapters + RetinaNet (lightweight, LR)
- fpn.py: Feature Pyramid Network (shared design)
- detection_head.py: RetinaNet parallel classification/regression head
- adapters.py: Spatial Projection Adapters for cross-resolution feature alignment
"""

from .teacher import TeacherDetector
from .student import StudentDetector
from .fpn import FPN
from .detection_head import RetinaNetHead
from .adapters import MultiLevelAdapters, SpatialAdapter

__all__ = [
    'TeacherDetector',
    'StudentDetector',
    'FPN',
    'RetinaNetHead',
    'MultiLevelAdapters',
    'SpatialAdapter',
]
