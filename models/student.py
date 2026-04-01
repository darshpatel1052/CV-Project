"""
Lightweight Student Model for Resolution-Agnostic KD
======================================================

Implements the mobile-scale student network designed for Low-Resolution (LR) imagery.
Mimics the intermediate spatial representations of the high-capacity teacher model
via Spatial Projection Adapters.

Architecture Overview:
    Input: Low-Resolution Images (128×128, downsampled 8× from 1024×1024)
    Backbone: MobileNetV2
        ├─ Designed for embedded devices and drone platforms.
        └─ Extracts features at strides 2, 4, 8, 16.
    Neck: Feature Pyramid Network (FPN)
        ├─ Maps MobileNetV2 channel depths [16, 24, 32, 96] to unified 256-d planes.
        └─ Outputs P3..P6 maps at strides 4, 8, 16, 32 relative to the 128x128 input.
    Adapters: MultiLevelAdapters
        ├─ Bridges the cross-resolution representation gap.
        ├─ Student P3..P6 shapes: [32, 16, 8, 4]
        ├─ Teacher P3..P6 shapes: [128, 64, 32, 16]
        └─ Upsamples FPN features exactly 4× (using 2 deconv blocks: 2^2 = 4) 
           so spatial dimensions match the Teacher's representations identically.
    Head: RetinaNet detection head
        ├─ Parallel branches for classification (15 classes).
        └─ Bounding box regression for object localization.

Use Case:
    Phase 2: Train this student network (both KD phase and baseline Phase 2a).
             Outputs explicit 'adapted_features' mimicking the Teacher architecture.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from models.fpn import FPN
from models.detection_head import RetinaNetHead
from models.adapters import MultiLevelAdapters


class StudentDetector(nn.Module):
    """
    Lightweight Student Object Detector using MobileNetV2 + FPN + Adapters.
    
    Args:
        num_classes (int): Number of target object categories.
        pretrained (bool): Whether to load ImageNet pre-trained weights for MobileNetV2.
    """
    
    def __init__(self, num_classes: int = 15, pretrained: bool = True):
        super().__init__()
        
        # =========================================================================
        # 1. Feature Extraction (Backbone)
        # =========================================================================
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        mobilenet_backbone = mobilenet_v2(weights=weights)
        
        # MobileNetV2 produces features across sequential InvertedResidual blocks.
        # We extract at indices yielding channels [16, 24, 32, 96].
        # These correspond to spatial strides [2, 4, 8, 16] w.r.t the input image.
        self.return_nodes = {
            'features.1':  'stage1',  # Channels: 16 (Stride 2 -> 64x64 on 128 input)
            'features.3':  'stage2',  # Channels: 24 (Stride 4 -> 32x32)
            'features.6':  'stage3',  # Channels: 32 (Stride 8 -> 16x16)
            'features.13': 'stage4'   # Channels: 96 (Stride 16 -> 8x8)
        }
        
        self.backbone = create_feature_extractor(mobilenet_backbone, return_nodes=self.return_nodes)
        
        # =========================================================================
        # 2. Multi-Scale Context (Neck)
        # =========================================================================
        self.fpn = FPN(
            in_channels_list=[16, 24, 32, 96],
            out_channels=256,
            add_extra_levels=True
        )
        
        # =========================================================================
        # 3. Spatial Projection Adapters (For Teacher Alignment)
        # =========================================================================
        # Teacher FPN is 256 channels across all levels.
        # Student FPN also outputs 256 channels natively because we set out_channels=256.
        # To align the spatial resolution: 
        # Student FPN spatial outputs [32x32, 16x16, 8x8, 4x4] -> Teacher FPN [128x128, 64x64, ...].
        # This is a 4× spatial gap for the features.
        # So we use exactly 2 blocks (num_upsample_blocks = 2) for 2^2 = 4x upscaling.
        self.adapters = MultiLevelAdapters(
            student_channels=[256, 256, 256, 256],
            teacher_channels=[256, 256, 256, 256],
            num_levels=4,
            num_upsample_blocks=2  # 4× spatial scaling
        )
        
        # =========================================================================
        # 4. Detection Predictions (Head)
        # =========================================================================
        self.head = RetinaNetHead(
            in_channels=256,
            num_classes=num_classes,
            num_anchors=9
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the student detector.
        
        Args:
            x (torch.Tensor): Image tensor of shape (B, 3, H, W). 
                              Normally 128×128 LR images.
            
        Returns:
            Dict[str, torch.Tensor/List]:
                - 'fpn_features': Un-adapted student FPN maps (for baseline inference)
                - 'adapted_features': Upsampled maps matching teacher shapes (for Feature KD)
                - 'cls_logits': Student class probabilities
                - 'bbox_regs': Student box offsets
        """
        # Backbone map
        raw_features = self.backbone(x)
        
        backbone_features = [
            raw_features['stage1'],
            raw_features['stage2'],
            raw_features['stage3'],
            raw_features['stage4']
        ]
        
        # FPN Output (Spatial dimensions 32, 16, 8, 4)
        fpn_features = self.fpn(backbone_features)
        
        # Adapt Features for Knowledge Distillation (Spatial dimensions 128, 64, 32, 16)
        adapted_features = self.adapters(fpn_features)
        
        # It's crucial that predictions run on the final adapted features if the student
        # will act structurally identical to the teacher in logits.
        # Alternatively, we could run the head on 'fpn_features' for pure efficiency during
        # deployment, and only use 'adapted_features' for the KD loss.
        # In a generic KD pipeline, the student learns from KD to adjust its native 
        # parameters. Hence, standard inference often runs on native 'fpn_features'
        # while KD loss is computed on 'adapted_features'. Wait! The teacher logits
        # are mapped across 195,840 anchors (128x128 base). For Logit KD to match, the 
        # student head must process the heavily upsampled adapted_features too!
        cls_logits, bbox_regs = self.head(adapted_features)
        
        return {
            'fpn_features': fpn_features,
            'adapted_features': adapted_features,
            'cls_logits': cls_logits,
            'bbox_regs': bbox_regs
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print(f"{'='*60}")
    print("Testing StudentDetector (MobileNetV2 + FPN + Adapters)")
    print(f"{'='*60}")
    
    try:
        print("Initializing Student Model (Pretrained=False for speed)...")
        model = StudentDetector(num_classes=15, pretrained=False)
        model.eval()
        
        # Simulated DOTA 128x128 Low-Resolution Drone Image Extract
        dummy_input = torch.randn(1, 3, 128, 128)
        
        print("\nExecuting Forward Pass with 128x128 LR tensor...")
        with torch.no_grad():
            outputs = model(dummy_input)
            
        print("\n✓ PASS: Forward pass executed normally.\n")
        
        print("Verifying FPN Shapes (Native Student Feature Domain):")
        for idx, fmap in enumerate(outputs['fpn_features']):
            print(f"  Native Student P{idx + 3} shape: {tuple(fmap.shape)}")
            
        print("\nVerifying Adapted Shapes (Teacher Spatial Domain):")
        for idx, amap in enumerate(outputs['adapted_features']):
            print(f"  Adapted Student P{idx + 3} shape: {tuple(amap.shape)}")
        
        print(f"\nVerifying Detection Logits:")
        print(f"  cls_logits tensor shape: {tuple(outputs['cls_logits'].shape)}")
        print(f"  bbox_regs tensor shape : {tuple(outputs['bbox_regs'].shape)}")
        
        # Expected anchors constraint validation (Same as Teacher: 195840 default on 1024 base)
        assert outputs['cls_logits'].shape[1] == 195840, "Anchor count fails to replicate the Teacher!"
        print("\n✓ SUCCESS: Student successfully mimics exact spatial/anchor footprint of Teacher!")
        
    except Exception as e:
        print(f"❌ FAIL: Exception occurred: {e}")
