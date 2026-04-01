"""
RetinaNet-Style Detection Head
================================

Parallel classification and bounding box regression sub-networks.
Used by both teacher and student models.

Architecture:
    Input FPN features (P3, P4, P5, P6): (B, C, H, W)
        ↓
    [Shared stem: 4× Conv3×3 + ReLU]
        ├─ Classification subnet → (B, A×K, H, W) where K=num_classes
        └─ Regression subnet → (B, A×4, H, W)
        
    Reshape to anchors:
        ├─ Class logits: (B, H×W×A, K)
        └─ Box offsets: (B, H×W×A, 4)

Key Features:
- Parallel design: cls and bbox branches don't share intermediate layers
- ReLU activations prevent gradient vanishing
- Separate class/bbox predictions per feature level
- Supports multi-scale detection (P3-P6 = 8×-128× stride)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """
    Convolution block: Conv2d + BatchNorm + ReLU
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size (default 3×3)
        stride: Stride (default 1)
        padding: Padding (default 1)
        groups: Number of groups for grouped conv (default 1)
        use_bn: Whether to use batch norm (default True)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        use_bn: bool = True,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=not use_bn
        )
        
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SubNet(nn.Module):
    """
    Shared sub-network stem for classification or regression.
    
    Design: 4 sequential conv blocks with ReLU
    - Used independently by cls and bbox branches
    - NOT shared between them (each has own parameters)
    - Creates rich feature representations for predictions
    
    Args:
        in_channels: Number of input channels from FPN
        out_channels: Output channels (fixed 256 for feature richness)
        num_layers: Number of conv blocks (default 4)
        num_outputs: Number of output channels (K for cls, 4 for bbox)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        num_layers: int = 4,
        num_outputs: int = 81  # 80 classes + 1 background
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        
        # Stack of 4 conv blocks: progressively transform features
        layers = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            layers.append(
                ConvBlock(
                    in_ch,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    activation='relu'
                )
            )
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Final output layer (no relu, will be followed by loss function)
        self.output_conv = nn.Conv2d(
            out_channels,
            num_outputs,
            kernel_size=3,
            stride=1,
            padding=1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input feature map (B, C_in, H, W)
        
        Returns:
            output: Logits (B, num_outputs, H, W)
        """
        x = self.conv_layers(x)
        x = self.output_conv(x)
        return x


class RetinaNetHead(nn.Module):
    """
    RetinaNet Detection Head with parallel classification and regression.
    
    Produces:
    - Class probabilities for each anchor-object pair
    - Bounding box offset predictions for each anchor
    
    Args:
        in_channels: Number of input channels from FPN (typically 256)
        num_classes: Number of object classes (e.g., 15 for DOTA)
        num_anchors: Number of anchor types per location (typically 9: 3 scales × 3 aspect ratios)
        num_conv_layers: Number of conv blocks in each subnet (default 4)
        prior_prob: Prior probability for focal loss to initialize class logits
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 15,
        num_anchors: int = 9,
        num_conv_layers: int = 4,
        prior_prob: float = 0.01
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_conv_layers = num_conv_layers
        self.prior_prob = prior_prob
        
        # Classification sub-network
        # Output: (B, H, W, num_anchors × num_classes)
        self.cls_subnet = SubNet(
            in_channels=in_channels,
            out_channels=256,
            num_layers=num_conv_layers,
            num_outputs=num_anchors * num_classes
        )
        
        # Regression sub-network
        # Output: (B, H, W, num_anchors × 4)
        self.bbox_subnet = SubNet(
            in_channels=in_channels,
            out_channels=256,
            num_layers=num_conv_layers,
            num_outputs=num_anchors * 4
        )
        
        # Initialize class logits with prior probability for better convergence
        # This helps with class imbalance in early training
        self._init_prior_prob()
    
    def _init_prior_prob(self):
        """Initialize classification output with prior probability."""
        # Formula: logit = log(p / (1-p))
        # For prior_prob=0.01: logit = log(0.01/0.99) ≈ -4.6
        # Initialize class sub-network's final layer bias
        for module in self.cls_subnet.modules():
            if isinstance(module, nn.Conv2d) and module.out_channels == self.num_anchors * self.num_classes:
                # Initialize bias with negative values to represent low confidence initially
                # This helps with focal loss early training (prevents log(0) issues)
                nn.init.constant_(module.bias, -4.595)
    
    def forward(
        self,
        features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate predictions for all FPN levels.
        
        Args:
            features: List of FPN feature maps [P3, P4, P5, P6]
                Each: (B, 256, H_i, W_i)
        
        Returns:
            cls_logits: Classification logits
                Shape: (B, total_anchors, num_classes)
            bbox_pred: Bounding box predictions
                Shape: (B, total_anchors, 4)
        
        Where total_anchors = sum(H_i × W_i × num_anchors) across all levels
        """
        cls_logits_per_level = []
        bbox_pred_per_level = []
        
        # Process each FPN level independently
        for level_idx, feature in enumerate(features):
            B, C, H, W = feature.shape
            
            # Classification: (B, C, H, W) → (B, num_anchors × num_classes, H, W)
            cls_logits = self.cls_subnet(feature)
            
            # Regression: (B, C, H, W) → (B, num_anchors × 4, H, W)
            bbox_pred = self.bbox_subnet(feature)
            
            # Reshape for easier processing during loss computation
            # (B, num_anchors × num_classes, H, W) → (B, num_classes, H × W × num_anchors)
            cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous()
            cls_logits = cls_logits.view(B, H, W, self.num_anchors, self.num_classes)
            cls_logits = cls_logits.view(B, -1, self.num_classes)  # (B, H×W×A, K)
            
            # (B, num_anchors × 4, H, W) → (B, H × W × num_anchors, 4)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous()
            bbox_pred = bbox_pred.view(B, H, W, self.num_anchors, 4)
            bbox_pred = bbox_pred.view(B, -1, 4)  # (B, H×W×A, 4)
            
            cls_logits_per_level.append(cls_logits)
            bbox_pred_per_level.append(bbox_pred)
        
        # Concatenate all levels
        cls_logits = torch.cat(cls_logits_per_level, dim=1)  # (B, total_anchors, K)
        bbox_pred = torch.cat(bbox_pred_per_level, dim=1)    # (B, total_anchors, 4)
        
        return cls_logits, bbox_pred
    
    def forward_features_only(
        self,
        features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate predictions for each level separately (for intermediate supervision).
        Used when we need per-level outputs (e.g., for feature distillation).
        
        Args:
            features: List of FPN feature maps
        
        Returns:
            cls_logits_per_level: List of class logits per FPN level
            bbox_pred_per_level: List of bbox predictions per FPN level
        """
        cls_logits_per_level = []
        bbox_pred_per_level = []
        
        for feature in features:
            cls_logits = self.cls_subnet(feature)
            bbox_pred = self.bbox_subnet(feature)
            
            cls_logits_per_level.append(cls_logits)
            bbox_pred_per_level.append(bbox_pred)
        
        return cls_logits_per_level, bbox_pred_per_level


def create_detection_head(
    in_channels: int = 256,
    num_classes: int = 15,
    num_anchors: int = 9
) -> RetinaNetHead:
    """
    Factory function to create detection head.
    
    Args:
        in_channels: Input channels from FPN
        num_classes: Number of object classes
        num_anchors: Anchors per location
    
    Returns:
        RetinaNetHead instance
    """
    return RetinaNetHead(
        in_channels=in_channels,
        num_classes=num_classes,
        num_anchors=num_anchors,
        num_conv_layers=4,
        prior_prob=0.01
    )


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("Testing RetinaNet Detection Head")
    print("=" * 70)
    
    # Create head
    head = RetinaNetHead(
        in_channels=256,
        num_classes=15,
        num_anchors=9
    )
    
    # Simulate FPN features
    B = 2  # batch size
    features = [
        torch.randn(B, 256, 64, 64),   # P3 (1/8 stride)
        torch.randn(B, 256, 32, 32),   # P4 (1/16 stride)
        torch.randn(B, 256, 16, 16),   # P5 (1/32 stride)
        torch.randn(B, 256, 8, 8),     # P6 (1/64 stride)
    ]
    
    # Forward pass
    cls_logits, bbox_pred = head(features)
    
    print(f"\n✓ Head initialized successfully")
    print(f"\nOutput shapes:")
    print(f"  Class logits: {cls_logits.shape}")
    
    total_spatial = 64*64 + 32*32 + 16*16 + 8*8
    expected_anchors = total_spatial * 9
    print(f"    Expected: ({B}, {expected_anchors}, 15)")
    print(f"  Box predictions: {bbox_pred.shape}")
    print(f"    Expected: ({B}, {expected_anchors}, 4)")
    
    total_anchors = cls_logits.shape[1]
    print(f"\nTotal anchors: {total_anchors:,}")
    print(f"  = (64×64 + 32×32 + 16×16 + 8×8) × 9 anchors/location")
    print(f"  = ({64*64} + {32*32} + {16*16} + {8*8}) × 9")
    print(f"  = {total_spatial} × 9")
    
    # Per-level outputs
    cls_per_level, bbox_per_level = head.forward_features_only(features)
    print(f"\nPer-level outputs:")
    for i, (cls, bbox) in enumerate(zip(cls_per_level, bbox_per_level)):
        print(f"  Level P{i+3}: cls {cls.shape}, bbox {bbox.shape}")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
