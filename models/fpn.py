"""
Feature Pyramid Network (FPN)
=============================

Multi-scale feature representation for object detection.
Creates semantically-rich feature pyramids from backbone outputs.

Architecture:
    Backbone outputs (C1, C2, C3, C4): Multi-scale features
        ├─ C1: 1/4 scale
        ├─ C2: 1/8 scale
        ├─ C3: 1/16 scale
        └─ C4: 1/32 scale
        ↓
    [Top-down pathway + lateral connections]
        ├─ P4 (1/16 scale)
        ├─ P3 (1/8 scale)
        ├─ P5 (1/32 scale)
        └─ P6 (1/64 scale, added)
        ↓
    Outputs: [P3, P4, P5, P6] - all with 256 channels

Key Benefits:
- Detects objects at multiple scales in one process
- Lateral connections leverage resolution + semantics
- Top-down pathway propagates semantic information
- Consistent channel dimensions (256) across all levels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class LateralConv(nn.Module):
    """
    1×1 convolution to align channel dimensions.
    
    Converts backbone features of different channels to target channels (256).
    Used in lateral connections of FPN.
    
    Args:
        in_channels: Input channels from backbone
        out_channels: Output channels (typically 256)
    """
    
    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        nn.init.kaiming_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 1×1 conv."""
        return self.conv(x)


class FeatureFusion(nn.Module):
    """
    Fuse lateral features with upsampled higher-level features.
    
    Formula:
        P_i = lateral(C_i) + bilinear_upsample(P_{i+1})
    
    Args:
        out_channels: Output channels (typically 256)
    """
    
    def __init__(self, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels
    
    def forward(
        self,
        lateral_feat: torch.Tensor,
        top_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse lateral and top-down features.
        
        Args:
            lateral_feat: Lateral connection from current level
            top_feat: Upsampled feature from higher level
        
        Returns:
            fused: (lateral_feat + upsampled(top_feat))
        """
        # Upsample top_feat to match lateral_feat spatial dimensions
        _, _, H, W = lateral_feat.shape
        top_feat_up = F.interpolate(
            top_feat,
            size=(H, W),
            mode='bilinear',
            align_corners=True
        )
        
        # Element-wise addition
        return lateral_feat + top_feat_up


class OutputConv(nn.Module):
    """
    3×3 convolution for output refinement.
    
    Reduces aliasing artifacts from upsampling.
    Adds non-linearity after fusion.
    
    Args:
        in_channels: Input channels (typically 256)
        out_channels: Output channels (typically 256)
    """
    
    def __init__(self, in_channels: int = 256, out_channels: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        nn.init.kaiming_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 3×3 conv."""
        return self.conv(x)


class FPN(nn.Module):
    """
    Feature Pyramid Network.
    
    Converts backbone multi-scale outputs into feature pyramid with:
    - Consistent channel dimensions (256)
    - Semantic information from coarse layers
    - Spatial detail from fine layers
    
    Used by both teacher (Swin-T) and student (MobileNetV2).
    
    Args:
        in_channels_list: List of input channels from backbone
                         [C1_ch, C2_ch, C3_ch, C4_ch]
                         For Swin-T: [96, 192, 384, 768]
                         For MobileNetV2: [16, 24, 32, 96]
        out_channels: Output channels for all pyramid levels (default 256)
        add_extra_levels: Whether to add P6 level (default True)
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        add_extra_levels: bool = True
    ):
        super().__init__()
        
        assert len(in_channels_list) == 4, "Expected 4 backbone levels"
        
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.add_extra_levels = add_extra_levels
        
        # Lateral convolutions (1×1) to unify channels
        self.lateral_convs = nn.ModuleList([
            LateralConv(ch, out_channels) for ch in in_channels_list
        ])
        
        # Feature fusion modules (upsample + add)
        self.feature_fusions = nn.ModuleList([
            FeatureFusion(out_channels) for _ in range(3)  # Fuse P4, P3, P5
        ])
        
        # Output convolutions (3×3) for refinement
        self.output_convs = nn.ModuleList([
            OutputConv(out_channels, out_channels) for _ in range(4)  # P3, P4, P5, P6
        ])
        
        # Extra level (P6) using max pooling
        if add_extra_levels:
            self.extra_level_pool = nn.MaxPool2d(
                kernel_size=1,
                stride=2,
                padding=0
            )
    
    def forward(
        self,
        backbone_features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Build feature pyramid from backbone outputs.
        
        Args:
            backbone_features: List of 4 backbone feature maps
                - backbone_features[0]: (B, C1, H1, W1) - 1/4 scale (coarse)
                - backbone_features[1]: (B, C2, H2, W2) - 1/8 scale
                - backbone_features[2]: (B, C3, H3, W3) - 1/16 scale
                - backbone_features[3]: (B, C4, H4, W4) - 1/32 scale (fine semantics)
        
        Returns:
            pyramid: List of 4 feature pyramid levels
                - pyramid[0]: P3 (1/8 scale, spatial detail)
                - pyramid[1]: P4 (1/16 scale)
                - pyramid[2]: P5 (1/32 scale)
                - pyramid[3]: P6 (1/64 scale, coarse)
        
        All with shape (B, out_channels, H_i, W_i)
        """
        
        # Sanity check
        if len(backbone_features) != 4:
            raise ValueError(f"Expected 4 backbone features, got {len(backbone_features)}")
        
        # Step 1: Apply lateral convolutions to all backbone features
        # This unifies channels to 256
        C1_lateral = self.lateral_convs[0](backbone_features[0])  # (B, 256, H1, W1)
        C2_lateral = self.lateral_convs[1](backbone_features[1])  # (B, 256, H2, W2)
        C3_lateral = self.lateral_convs[2](backbone_features[2])  # (B, 256, H3, W3)
        C4_lateral = self.lateral_convs[3](backbone_features[3])  # (B, 256, H4, W4)
        
        # Step 2: Top-down pathway with lateral connections
        # Start from coarsest level and work up
        
        # P5 = C4 (no fusion needed, already coarsest)
        P5 = C4_lateral
        
        # P4 = lateral(C3) + upsample(P5)
        P4 = self.feature_fusions[0](C3_lateral, P5)
        
        # P3 = lateral(C2) + upsample(P4)
        P3 = self.feature_fusions[1](C2_lateral, P4)
        
        # P2 = lateral(C1) + upsample(P3) - if we have 5 levels
        # But for 4 levels, we skip this and expand later with extra level
        
        # Step 3: Output convolutions (3×3) for refinement
        P3 = self.output_convs[0](P3)  # (B, 256, H2, W2)
        P4 = self.output_convs[1](P4)  # (B, 256, H3, W3)
        P5 = self.output_convs[2](P5)  # (B, 256, H4, W4)
        
        # Step 4: Add extra level P6 via max pooling
        if self.add_extra_levels:
            # P6 is obtained by using max pooling on P5
            # Reduces spatial dimensions by 2
            P6 = self.extra_level_pool(P5)  # (B, 256, H5, W5)
            P6 = self.output_convs[3](P6)
        else:
            P6 = None
        
        # Return in order from coarse to fine for detection
        if self.add_extra_levels:
            pyramid = [P3, P4, P5, P6]  # Spatial scales: 1/8, 1/16, 1/32, 1/64
        else:
            pyramid = [P3, P4, P5]
        
        return pyramid


def create_fpn_from_backbone(
    backbone_name: str,
    out_channels: int = 256
) -> Tuple[FPN, int]:
    """
    Factory function to create FPN for specific backbone.
    
    Args:
        backbone_name: Name of backbone ('swin_t' or 'mobilenetv2')
        out_channels: Output channels for all pyramid levels
    
    Returns:
        fpn: FPN instance
        fpn_out_channels: Output channels (256)
    """
    if backbone_name == 'swin_t':
        # Swin-T output channels per stage: [96, 192, 384, 768]
        in_channels_list = [96, 192, 384, 768]
    elif backbone_name == 'mobilenetv2':
        # MobileNetV2 output channels: [16, 24, 32, 96]
        in_channels_list = [16, 24, 32, 96]
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    fpn = FPN(in_channels_list, out_channels=out_channels)
    return fpn, out_channels


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("Testing Feature Pyramid Network")
    print("=" * 70)
    
    # Test with Swin-T backbone channels
    print("\n--- Testing with Swin-T backbone ---")
    fpn_swin = FPN(
        in_channels_list=[96, 192, 384, 768],
        out_channels=256,
        add_extra_levels=True
    )
    
    B = 2  # batch size
    # Simulate Swin-T backbone outputs
    backbone_features_swin = [
        torch.randn(B, 96, 256, 256),    # C1: 1/4 scale
        torch.randn(B, 192, 128, 128),   # C2: 1/8 scale
        torch.randn(B, 384, 64, 64),     # C3: 1/16 scale
        torch.randn(B, 768, 32, 32),     # C4: 1/32 scale (coarse)
    ]
    
    pyramid_swin = fpn_swin(backbone_features_swin)
    
    print(f"✓ FPN created for Swin-T")
    print(f"\nPyramid levels:")
    for i, p in enumerate(pyramid_swin):
        scale = 8 * (2 ** i)
        print(f"  P{i+3} (1/{scale} scale): {p.shape}")
    
    # Test with MobileNetV2 backbone channels
    print("\n--- Testing with MobileNetV2 backbone ---")
    fpn_mobile = FPN(
        in_channels_list=[16, 24, 32, 96],
        out_channels=256,
        add_extra_levels=True
    )
    
    # Simulate MobileNetV2 backbone outputs
    backbone_features_mobile = [
        torch.randn(B, 16, 64, 64),      # C1: 1/4 scale
        torch.randn(B, 24, 32, 32),      # C2: 1/8 scale
        torch.randn(B, 32, 16, 16),      # C3: 1/16 scale
        torch.randn(B, 96, 8, 8),        # C4: 1/32 scale (coarse)
    ]
    
    pyramid_mobile = fpn_mobile(backbone_features_mobile)
    
    print(f"✓ FPN created for MobileNetV2")
    print(f"\nPyramid levels:")
    for i, p in enumerate(pyramid_mobile):
        scale = 8 * (2 ** i)
        print(f"  P{i+3} (1/{scale} scale): {p.shape}")
    
    # Verify all outputs have same channels
    print(f"\n✓ Channel consistency check:")
    all_channels = [p.shape[1] for p in pyramid_swin]
    assert all(c == 256 for c in all_channels), "Not all channels are 256!"
    print(f"  All pyramid levels: {all_channels} ✓")
    
    # Factory function test
    print(f"\n--- Testing factory functions ---")
    fpn_factory_swin, ch = create_fpn_from_backbone('swin_t')
    fpn_factory_mobile, ch = create_fpn_from_backbone('mobilenetv2')
    print(f"✓ Factory functions created FPNs successfully")
    
    print("\n" + "=" * 70)
    print("✓ All FPN tests passed!")
    print("=" * 70)
