"""
Spatial Projection Adapters for Knowledge Distillation
========================================================

Bridges resolution gap (8×) between teacher and student feature maps using
deconvolution layers and channel projection.

Teacher Input:   1024×1024  → FPN features at 1/8, 1/16, 1/32, 1/64
Student Input:   128×128    → FPN features at 1/8, 1/16, 1/32, 1/64

Resolution mismatch: 8× downsampling
Adapter Strategy:
    - Student F_s at 64×64 (stride 16)
    - Upsample 2× → 128×128 (stride 8)
    - Project channels to match teacher
    - Output aligned with F_t at matching spatial scale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DeconvBlock(nn.Module):
    """
    Deconvolution block: Upsample + Conv + BatchNorm + ReLU
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if dropout > 0:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class ConvAdapt(nn.Module):
    """
    Channel adaptation layer: Project channels without spatial changes
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        
        if dropout > 0:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class SpatialAdapter(nn.Module):
    """
    Spatial Projection Adapter for single FPN level.
    
    Upsamples student features 2× and projects channels to match teacher.
    
    Design:
        - Input: Student features (C_student, H, W)
        - Deconv ×2: 2× upsampling → (C_student, 2H, 2W)
        - ConvAdapt: Channel projection → (C_teacher, 2H, 2W)
        - Output: Aligned features ready for KD loss
    
    Args:
        student_channels: Number of input channels (from student FPN)
        teacher_channels: Number of target channels (teacher FPN channels)
        num_upsample_blocks: Spatial upsampling factor (default: 2, for 8× total resolution gap)
    """
    
    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        num_upsample_blocks: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        self.student_channels = student_channels
        self.teacher_channels = teacher_channels
        self.num_upsample_blocks = num_upsample_blocks
        
        # Build deconvolution chain for upsampling
        self.upsample_blocks = nn.ModuleList()
        
        current_channels = student_channels
        for i in range(num_upsample_blocks):
            # Each deconv upsamples by 2×
            next_channels = teacher_channels if i == num_upsample_blocks - 1 else current_channels
            self.upsample_blocks.append(
                DeconvBlock(
                    in_channels=current_channels,
                    out_channels=next_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    dropout=dropout if i < num_upsample_blocks - 1 else 0
                )
            )
            current_channels = next_channels
        
        # Final channel adaptation (may be identity if already projected)
        if current_channels != teacher_channels:
            self.channel_adapt = ConvAdapt(
                current_channels,
                teacher_channels,
                kernel_size=1
            )
        else:
            self.channel_adapt = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Student features (B, C_student, H, W)
        
        Returns:
            Aligned features (B, C_teacher, 4H, 4W) or (B, C_teacher, (2**num_upsample_blocks) * H, (2**num_upsample_blocks) * W)
        """
        # Upsample
        for block in self.upsample_blocks:
            x = block(x)
        
        # Optional channel adaptation
        if self.channel_adapt is not None:
            x = self.channel_adapt(x)
        
        return x


class MultiLevelAdapters(nn.Module):
    """
    Collection of spatial adapters for multiple FPN levels.
    
    Adapts student FPN features to align with teacher FPN features.
    Handles different channel counts per level.
    
    Args:
        student_channels: List of student FPN channels per level
                         [C_P3, C_P4, C_P5, C_P6]
        teacher_channels: List of teacher FPN channels per level
                         [C_P3, C_P4, C_P5, C_P6]
        num_levels: Number of FPN levels (default: 4)
        num_upsample_blocks: Number of 2x upsampling blocks per adapter (default: 2)
    
    Example:
        # MobileNetV2 student → Swin-T teacher
        student_channels = [256, 256, 256, 256]  # FPN output channels
        teacher_channels = [256, 256, 256, 256]
        
        adapters = MultiLevelAdapters(
            student_channels=student_channels,
            teacher_channels=teacher_channels,
            num_levels=4,
            num_upsample_blocks=2  # 2× upsampling per level
        )
        
        adapted_features = adapters(student_fpn_features)
        # adapted_features[0] matches teacher spatial scale at P3
        # adapted_features[1] matches teacher spatial scale at P4
        # etc.
    """
    
    def __init__(
        self,
        student_channels: List[int],
        teacher_channels: List[int],
        num_levels: int = 4,
        num_upsample_blocks: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        
        assert len(student_channels) == num_levels, \
            f"student_channels length {len(student_channels)} != num_levels {num_levels}"
        assert len(teacher_channels) == num_levels, \
            f"teacher_channels length {len(teacher_channels)} != num_levels {num_levels}"
        
        self.num_levels = num_levels
        self.num_upsample_blocks = num_upsample_blocks
        
        self.adapters = nn.ModuleList()
        for i in range(num_levels):
            adapter = SpatialAdapter(
                student_channels=student_channels[i],
                teacher_channels=teacher_channels[i],
                num_upsample_blocks=num_upsample_blocks,
                dropout=dropout
            )
            self.adapters.append(adapter)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Adapt multiple student features to teacher space.
        
        Args:
            features: List of student FPN features
                     [F_P3, F_P4, F_P5, F_P6]
                     Each shape: (B, C, H, W)
        
        Returns:
            List of adapted features
            [F_adapted_P3, F_adapted_P4, F_adapted_P5, F_adapted_P6]
            Each shape: (B, C_teacher, (2**num_upsample_blocks)*H, (2**num_upsample_blocks)*W)
        """
        adapted = []
        for i, (feature, adapter) in enumerate(zip(features, self.adapters)):
            adapted_feat = adapter(feature)
            adapted.append(adapted_feat)
        
        return adapted


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import torch
    
    print("Testing Spatial Projection Adapters")
    print("=" * 60)
    
    # Test 1: Single level adapter
    print("\n1. Single Adapter (MobileNetV2 → Swin-T)")
    adapter = SpatialAdapter(
        student_channels=256,
        teacher_channels=256,
        num_upsample_blocks=2
    )
    
    # Student features: 128×128 input → FPN output at 1/16 (stride 16)
    # So P3 ~= 16×16
    student_feat = torch.randn(1, 256, 16, 16)
    adapted_feat = adapter(student_feat)
    
    print(f"  Input shape:  {tuple(student_feat.shape)}")
    print(f"  Output shape: {tuple(adapted_feat.shape)}")
    print(f"  Expected:     (1, 256, 64, 64)  [4× upsampling]")
    assert adapted_feat.shape == (1, 256, 64, 64), "Shape mismatch!"
    print("  ✓ PASS")
    
    # Test 2: Multi-level adapters
    print("\n2. Multi-Level Adapters (4 FPN levels)")
    adapters = MultiLevelAdapters(
        student_channels=[256, 256, 256, 256],
        teacher_channels=[256, 256, 256, 256],
        num_levels=4,
        num_upsample_blocks=2
    )
    
    # Create features for P3, P4, P5, P6
    # P3: stride 8 on 128×128 → 16×16
    # P4: stride 16 on 128×128 → 8×8
    # P5: stride 32 on 128×128 → 4×4
    # P6: stride 64 on 128×128 → 2×2
    features = [
        torch.randn(2, 256, 16, 16),  # P3
        torch.randn(2, 256, 8, 8),    # P4
        torch.randn(2, 256, 4, 4),    # P5
        torch.randn(2, 256, 2, 2),    # P6
    ]
    
    adapted = adapters(features)
    
    expected_shapes = [
        (2, 256, 64, 64),  # P3: 16×16 → 2×2 deconv (4× upsampling) → 64×64
        (2, 256, 32, 32),  # P4: 8×8 → 4× upsampling → 32×32
        (2, 256, 16, 16),  # P5: 4×4 → 4× upsampling → 16×16
        (2, 256, 8, 8),    # P6: 2×2 → 4× upsampling → 8×8
    ]
    
    for i, (feat, expected) in enumerate(zip(adapted, expected_shapes)):
        print(f"  Level {i}: {tuple(feat.shape)} (expected {expected})", end="")
        assert tuple(feat.shape) == expected, f"Shape mismatch at level {i}!"
        print(" ✓")
    
    print("  ✓ PASS")
    
    # Test 3: Different channel counts
    print("\n3. Channel Projection (256 → 512)")
    adapter = SpatialAdapter(
        student_channels=256,
        teacher_channels=512,
        num_upsample_blocks=2
    )
    
    student_feat = torch.randn(1, 256, 16, 16)
    adapted_feat = adapter(student_feat)
    
    print(f"  Input shape:  {tuple(student_feat.shape)}")
    print(f"  Output shape: {tuple(adapted_feat.shape)}")
    print(f"  Expected:     (1, 512, 64, 64)  [256 → 512 channels + 4× upsampling]")
    assert adapted_feat.shape == (1, 512, 64, 64), "Shape mismatch!"
    print("  ✓ PASS")
    
    # Test 4: Gradient flow
    print("\n4. Gradient Flow Test")
    adapter = SpatialAdapter(256, 256, num_upsample_blocks=2)
    student_feat = torch.randn(1, 256, 16, 16, requires_grad=True)
    adapted_feat = adapter(student_feat)
    loss = adapted_feat.sum()
    loss.backward()
    
    assert student_feat.grad is not None, "No gradient computed!"
    print(f"  Student gradient shape: {tuple(student_feat.grad.shape)}")
    print("  ✓ PASS")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
