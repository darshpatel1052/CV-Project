"""
Teacher Model for Knowledge Distillation
=========================================

Implements the high-capacity teacher network designed for High-Resolution (HR) imagery.
Extracts semantically rich multi-scale features to supervise a lightweight student
model via Resolution-Agnostic Knowledge Distillation.

Architecture Overview:
    Input: Ultra-High-Resolution Images (e.g., 1024×1024)
    Backbone: Swin Transformer Tiny (Swin-T) 
        ├─ Extracts hierarchical spatial representations.
        └─ Provides local and global context using windowed attention.
    Neck: Feature Pyramid Network (FPN) 
        ├─ Unifies dimensionality to 256 channels.
        └─ Merges coarse semantics with fine-grained localizations.
    Head: RetinaNet detection head
        ├─ Parallel branches for classification (15 DOTA classes).
        └─ Bounding box regression for object localization.

Use Case:
    Phase 1: Train independently on HR cropped DOTA images to converge on high mAP.
    Phase 2: Freeze weights and deploy as the "teacher" to generate FPN activation maps
             and target logits for the mobile LR student.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from .fpn import FPN
from .detection_head import RetinaNetHead


class TeacherDetector(nn.Module):
    """
    Teacher Object Detector using Swin-T + FPN + RetinaNet Head.
    
    Args:
        num_classes (int): Number of target object categories (15 for DOTA).
        pretrained (bool): Whether to load ImageNet pre-trained weights for the backbone.
                           Essential for rapid convergence in Phase 1.
    """
    
    def __init__(self, num_classes: int = 15, pretrained: bool = True):
        super().__init__()
        
        # 1. Backbone Initialization: Swin-T
        # Swin-T balances high representation capacity (Transformer) with manageable
        # parameter counts (~28M) for extracting HR semantic features.
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        
        # We extract features using torchvision's standard feature extraction.
        swin_backbone = swin_t(weights=weights)
        
        # Map specific internal stages to readable keys.
        # Swin-T processes inputs hierarchically across 4 stages.
        self.return_nodes = {
            'features.1': 'stage1',  # Output channels: 96,  Stride: 4
            'features.3': 'stage2',  # Output channels: 192, Stride: 8
            'features.5': 'stage3',  # Output channels: 384, Stride: 16
            'features.7': 'stage4'   # Output channels: 768, Stride: 32
        }
        
        self.backbone = create_feature_extractor(swin_backbone, return_nodes=self.return_nodes)
        
        # 2. Neck Initialization: FPN
        # The student model will be trained to replicate these specific unified 256-d planes.
        self.fpn = FPN(
            in_channels_list=[96, 192, 384, 768],
            out_channels=256,
            add_extra_levels=True  # P6 is generated dynamically within the FPN
        )
        
        # 3. Detection Head Initialization: RetinaNet
        self.head = RetinaNetHead(
            in_channels=256,
            num_classes=num_classes,
            num_anchors=9  # Default 3 scales × 3 aspect ratios per spatial bin
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the teacher detector.
        
        During Distillation (Phase 2), we need BOTH the intermediate FPN features 
        and the final predictions.
        
        Args:
            x (torch.Tensor): Image tensor of shape (B, 3, H, W). 
                              Normally 1024×1024 HR images.
            
        Returns:
            Dict[str, torch.Tensor/List]:
                - 'fpn_features': List of P3..P6 feature maps (Feature KD Target)
                - 'cls_logits': Class probabilities (Logit KD / Standard loss)
                - 'bbox_regs': Box offsets (Logit KD / Standard loss)
        """
        # =========================================================================
        # 1. Feature Extraction (Backbone)
        # =========================================================================
        raw_features = self.backbone(x)
        
        backbone_features = []
        # Swin Transformer outputs feature batches formatted as (B, H, W, C).
        # We must explicitly project the channel dimension back to standard PyTorch
        # spatial dimensions (B, C, H, W) before passing into Conv2d-based FPN modules.
        for stage_name in ['stage1', 'stage2', 'stage3', 'stage4']:
            feat = raw_features[stage_name].permute(0, 3, 1, 2)
            backbone_features.append(feat)
            
        # =========================================================================
        # 2. Multi-Scale Context (Neck)
        # =========================================================================
        # Outputs a list: [P3, P4, P5, P6] all aligned to 256 channels.
        # These are the "Teacher Thoughts" the lightweight student must mimic via 
        # the SpatialProjectionAdapters.
        fpn_features = self.fpn(backbone_features)
        
        # =========================================================================
        # 3. Detection Predictions (Head)
        # =========================================================================
        # The teacher generates dense prior anchors mapped precisely across levels P3-P6.
        cls_logits, bbox_regs = self.head(fpn_features)
        
        # Package and return explicit components needed for multi-loss distillation.
        return {
            'fpn_features': fpn_features,
            'cls_logits': cls_logits,
            'bbox_regs': bbox_regs
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print(f"{'='*60}")
    print("Testing TeacherDetector (Swin-T + FPN + RetinaNet)")
    print(f"{'='*60}")
    
    try:
        # 1. Initialize Teacher
        print("Initializing Teacher Model (Pretrained=False for speed)...")
        model = TeacherDetector(num_classes=15, pretrained=False)
        model.eval()
        
        # 2. Generate Dummy 1024x1024 High-Resolution Image Tensor
        # Shape: (Batch_Size=1, Channels=3, Height=1024, Width=1024)
        dummy_input = torch.randn(1, 3, 1024, 1024)
        
        # 3. Simulate Forward Pass
        print("\nExecuting Forward Pass with 1024x1024 tensor...")
        with torch.no_grad():
            outputs = model(dummy_input)
            
        print("\n✓ PASS: Forward pass executed normally.\n")
        
        # 4. Verify Tensors Structure
        print("Verifying Output Tensors Shapes:")
        total_spatial_locations = 0
        for idx, fmap in enumerate(outputs['fpn_features']):
            h, w = fmap.shape[2:]
            curr_locations = h * w
            total_spatial_locations += curr_locations
            print(f"  FPN Level {idx + 3} (P{idx + 3}) shape: {tuple(fmap.shape)} "
                  f"--> {h}x{w} = {curr_locations} spatial bins")
        
        # Explain the total anchors math
        num_anchors_per_location = 9 # 3 scales x 3 aspect ratios
        expected_total_anchors = total_spatial_locations * num_anchors_per_location
        print(f"\n  Total Spatial Locations across all FPN levels: {total_spatial_locations}")
        print(f"  Anchors per location: {num_anchors_per_location}")
        print(f"  Expected Total Anchors: {total_spatial_locations} * {num_anchors_per_location} = {expected_total_anchors}\n")
            
        print(f"  cls_logits tensor shape: {tuple(outputs['cls_logits'].shape)}")
        print(f"  bbox_regs tensor shape : {tuple(outputs['bbox_regs'].shape)}")
        
        # Verify correctness programmatically
        assert outputs['cls_logits'].shape[1] == expected_total_anchors, "Mismatch in expected anchor count!"
        print("\nSUCCESS!\nDetection head output anchor counts perfectly match the expected FPN spatial dimensions.")
        
    except Exception as e:
        print(f"❌ FAIL: Exception occurred: {e}")
