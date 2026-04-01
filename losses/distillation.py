"""
Knowledge Distillation Loss Functions for Resolution-Agnostic KD
=================================================================

This module implements the core knowledge distillation (KD) losses that enable
the lightweight student model to learn from the high-capacity teacher model.

Two-Stream Knowledge Transfer:
==============================

1. LOGIT-BASED KD (Dark Knowledge Transfer)
   ├─ Transfers soft probability distributions from teacher to student
   ├─ Uses temperature-scaled softmax to reveal implicit knowledge
   ├─ Formula: KL_div(soft_teacher_logits, soft_student_logits)
   └─ Benefits: Transfers "hints" about confused classes

2. FEATURE-BASED KD (Representation Alignment)
   ├─ Aligns intermediate feature maps between teacher and student
   ├─ Student adapters project low-res features to match teacher spatial dims
   ├─ Formula: MSE(Teacher_FPN_features, Student_Adapter_features)
   └─ Benefits: Transfers fine-grained spatial reasoning

Why Two Forms of KD Work Better Than One:
==========================================

Imagine a student trying to learn to drive:

Without KD (Standard Training):
├─ Only sees final performance scores: "You crashed (loss=1.0)"
├─ Student guesses what went wrong: "Maybe steering angle?"
├─ Slow, inefficient learning

With Logit KD (Softmax temperature):
├─ Teacher reveals soft probabilities: "I'm 70% sure it's left turn, 20% right"
├─ Student learns not just THE answer, but the REASONING
├─ Learns that left/right are close competitors (shouldn't distinguish harshly)

With Feature KD (Intermediate supervision):
├─ Teacher shows intermediate decisions: "After first stage, I focused on road edges"
├─ Student learns to recognize SIMILAR patterns at each stage
├─ Even with low-res input, learns WHERE to look

Combined:
├─ Student gets both WHAT to predict AND HOW to think
├─ Convergence is much faster, performance much better
└─ Student gains ability to perform on low-res when trained on teacher's HR semantics

Mathematical Background:
========================

TEMPERATURE SCALING (Softmax smoothing):
┌─────────────────────────────────────────────────────────────────┐
│ Original softmax: σ(z_i) = exp(z_i) / Σ exp(z_j)              │
│                                                                   │
│ Temperature softmax: σ(z_i, T) = exp(z_i/T) / Σ exp(z_j/T)   │
│                                                                   │
│ Effect of temperature T:                                        │
│ ├─ T=1 (default):   Sharp probabilities [0.95, 0.03, 0.02]   │
│ ├─ T=4 (soft):      Distributed probabilities [0.5, 0.3, 0.2] │
│ └─ T→∞:             Uniform probabilities [0.33, 0.33, 0.33] │
│                                                                   │
│ Why T>1 helps KD:                                               │
│ └─ Large logits scale down, small logits scale up relative     │
│    to each other. Probabilities spread out, revealing which    │
│    "wrong" answers the teacher nearly picked.                  │
│    Student learns: "When I see X, teacher almost picks both    │
│    plane AND ship, so I should too."                           │
│    This is "dark knowledge" - implicit reasoning patterns.     │
└─────────────────────────────────────────────────────────────────┘

KL DIVERGENCE (distribution mismatch):
        KL(P || Q) = Σ P(i) * log(P(i) / Q(i))

For KD with temperature:
        KL(soft_teacher_T4 || soft_student_T4) 
        = Σ P_teacher(i, T=4) * log(P_teacher(i, T=4) / P_student(i, T=4))

Why temperature T in both:
├─ Ensures symmetric KL divergence (P and Q same temperature)
├─ Prevents vanishing gradients when distributions are too different
└─ Allows trade-off between hard targets (T→1) and dark knowledge (T→∞)

FEATURE MSE (spatial alignment):
        L_feature = ||F_teacher - F_student_adapted||^2 / N_pixels

Where:
├─ F_teacher: FPN output feature maps from teacher (e.g., shape [B,256,64,64])
├─ F_student_adapted: Student FPN + Adapter output (after upsampling, same shape)
└─ Ensures student learns SPATIAL patterns, not just class predictions


Cross-Resolution Challenge:
===========================

Teacher sees: 1024×1024 image
┌──────────────────────────────────────┐
│ High-res details:                    │
│ - Vehicle wheels, windows             │
│ - Building corners, edges             │
│ - Lake water ripples                  │
│ Features: [B, 256, 128, 128]          │
│           (at 1:8 stride)             │
└──────────────────────────────────────┘

Student sees: 128×128 image (8× downsampled, blurry)
┌──────────────────────────────────────┐
│ Low-res approximation:                │
│ - Vehicle blobs                       │
│ - Building silhouettes                │
│ - Water mass (no ripples)             │
│ Features: [B, 128, 16, 16]            │
│           (at 1:8 stride)             │
└──────────────────────────────────────┘

ADAPTERS SOLVE THIS:
Student features [B, 128, 16, 16] 
    → 2× deconv (4× total upsampling)
    → [B, 256, 64, 64] (matches teacher!)
    → MSE loss to teacher's [B, 256, 128, 128]?

NO! Spatial resolution mismatch remains. But:
├─ Student features [B, 128, 16, 16] captured LOW-RES patterns
├─ Adapter learns to EXPAND these into 64×64 "hallucinated" details
├─ Teacher's 128×128 features are too fine-grained for 128 pixel student
└─ Student realistically matches teacher's 64×64 representation level

The trick: Train student to match teacher's P4 level (stride 16), not P3 (stride 8).
This accounts for information loss due to 8× downsampling.


Typical Loss Weighting:
======================
Total Loss = α * L_detection + β * L_logit_kd + γ * L_feature_kd

├─ α = 1.0    (detection loss: essential, primary task)
├─ β = 0.5    (logit KD: helps but not paramount)
├─ γ = 1.0    (feature KD: crucial for spatial reasoning transfer)

Intuition:
├─ Logit KD alone: Student mimics confusions, but learns low-level shallow features
├─ Feature KD alone: Student aligns features, but loses implicit class knowledge
└─ Both combined: Student learns spatial patterns AND implicit reasoning = best!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# LOGIT-BASED KNOWLEDGE DISTILLATION (Dark Knowledge Transfer)
# ============================================================================

class LogitKDLoss(nn.Module):
    r"""
    Knowledge Distillation Loss for Detection Logits.
    
    Transfers class confusion patterns and implicit knowledge from teacher to student
    using temperature-scaled softmax targets.
    
    In object detection, after spatial filtering:
    ├─ Student classification logits shape: (num_pos_anchors, num_classes)
    ├─ Teacher classification logits shape: (num_pos_anchors, num_classes)
    └─ We don't use confidence scores (those are handled separately)
    
    Args:
        temperature (float): Softmax temperature for softening probability distributions.
                           Values > 1 reveal implicit knowledge by spreading probabilities.
                           Default: 4.0 (empirically chosen for object detection)
                           
                           Examples:
                           ├─ T=1.0:  Sharp [0.95, 0.03, 0.02] → standard CE
                           ├─ T=4.0:  Soft [0.35, 0.32, 0.33] → reveals confusion
                           └─ T=10.0: Very soft [0.34, 0.33, 0.33] → near uniform
        
        alpha (float): Weighting factor for KD loss in combined loss.
                      Typically 0.5. Default: 1.0 (can be tuned).
    
    Forward:
        inputs: Student classification logits (num_anchors, num_classes)
        targets: Teacher classification logits (num_anchors, num_classes)
        
    Returns:
        Scaled KL divergence loss
    
    Implementation Note:
        Uses PyTorch's KLDivLoss with log_target=False.
        This computes: Σ log(P_student) * (P_teacher - P_student)
        Where P is softmax(logits / T)
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 1.0):
        super(LogitKDLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=False)
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KD loss between student and teacher classification logits.
        
        Args:
            student_logits: Raw predictions from student (shape: [N, C])
                          N = number of positive anchors
                          C = number of classes
            teacher_logits: Raw predictions from teacher (shape: [N, C])
                          Same shape as student_logits
        
        Returns:
            Scalar KL divergence loss
        
        Step-by-step:
        1. Apply temperature scaling to logits
        2. Convert to soft probabilities via softmax
        3. Compute KL divergence between soft distributions
        4. Scale by temperature factor (T^2 maintains gradient magnitude)
        """
        
        # Step 1: Soft target generation (teacher perspective)
        # Apply temperature to smooth the distribution
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Step 2: Student soft predictions
        # We need log_softmax for KL divergence input
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # Step 3: KL divergence (teacher distribution vs student distribution)
        # KL(soft_targets || student_soft)
        # Even though we want to pull student toward teacher, KLDiv is asymmetric
        # and PyTorch convention is KL(target || prediction), so we pass them in that order
        kd_loss = self.kl_loss(student_soft, soft_targets)
        
        # Step 4: Scale by T^2 to maintain gradient magnitude
        # Higher T means lower gradient magnitude from softmax, so we compensate
        kd_loss = kd_loss * (self.temperature ** 2)
        
        # Step 5: Apply alpha weighting
        return self.alpha * kd_loss


# ============================================================================
# FEATURE-BASED KNOWLEDGE DISTILLATION (Representation Alignment)
# ============================================================================

class FeatureKDLoss(nn.Module):
    r"""
    Feature-Level Knowledge Distillation Loss.
    
    Enforces spatial alignment between teacher and student FPN feature maps.
    This transfers intermediate spatial representations and fine-grained patterns
    that cannot be captured by logit-only distillation.
    
    Critical for detecting small objects at low resolution:
    ├─ Logit KD alone: Student predicts "vehicle" but doesn't know WHICH pixels
    ├─ Feature KD: Student learns to focus on vehicle-shaped regions even at low-res
    └─ Combined: Student detects the right object in the right location
    
    Architecture:
        Teacher FPN outputs P3, P4, P5, P6 at multiple scales
        ├─ Student FPN outputs correspondingly at SAME semantic scale
        ├─ Student Adapters upsample to match teacher spatial dimensions
        └─ MSE loss between teacher and upsampled student features
    
    Why MSE and Not Other Metrics?
    ==============================
    Alternative: Cosine Similarity
    ├─ Pro: Scale-invariant (only direction matters)
    ├─ Con: Loses magnitude information (student might learn weak features)
    └─ For detection: magnitude (confidence) matters
    
    Alternative: Contrastive Learning
    ├─ Pro: More sophisticated, learns discriminative features
    ├─ Con: Expensive, many hyperparameters
    └─ For detection with adapters: overkill
    
    Our Choice: MSE
    ├─ Pro: Simple, stable, preserves magnitude, gradient flow is predictable
    ├─ Con: Sensitive to scale (but adapters handle projection)
    └─ Works well for intermediate feature alignment
    
    Args:
        gamma (float): Weighting factor for feature KD in combined loss.
                      Typically 1.0 (same as detection loss).
                      Can be tuned per pyramid level.
                      
        reduction (str): How to aggregate loss across spatial dimensions.
                        Options: 'mean' (default) or 'sum'
    
    Forward:
        teacher_features: Dict of teacher FPN features
                        {level: [B, C, H, W]}
        student_adapted_features: Dict of student adapted features
                        {level: [B, C, H, W]} (same shape as teacher)
        
    Returns:
        Scalar MSE loss averaged across all pyramid levels
    
    Implementation:
        For each pyramid level:
        ├─ Ensure spatial dimensions match (adapter responsibility)
        ├─ Compute L2 norm between feature maps
        └─ Average across spatial dimensions and batch
    """
    
    def __init__(self, gamma: float = 1.0, reduction: str = 'mean'):
        super(FeatureKDLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        teacher_features: Dict[str, torch.Tensor],
        student_adapted_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute feature-level KD loss.
        
        Args:
            teacher_features: Dict with keys like 'p3', 'p4', 'p5', 'p6'
                            Each value is [B, C, H, W] tensor
            student_adapted_features: Dict with same keys
                            Same shape as teacher_features (adapter ensures this)
        
        Returns:
            Scalar MSE loss
        
        Step-by-step:
        1. Iterate over pyramid levels
        2. For each level, verify spatial shape match
        3. Compute MSE between teacher and student
        4. Average across all levels
        """
        
        total_loss = 0.0
        num_levels = 0
        
        # Step 1: Process each pyramid level (P3, P4, P5, P6)
        for level_key in teacher_features.keys():
            if level_key not in student_adapted_features:
                logger.warning(f"Level {level_key} in teacher but not student_adapted")
                continue
            
            teacher_feat = teacher_features[level_key]
            student_feat = student_adapted_features[level_key]
            
            # Step 2: Verify shape compatibility
            if teacher_feat.shape != student_feat.shape:
                raise ValueError(
                    f"Shape mismatch at {level_key}: "
                    f"teacher {teacher_feat.shape} vs student {student_feat.shape}. "
                    f"Adapter should have upsampled student features to match teacher."
                )
            
            # Step 3: MSE loss for this level
            mse = self.mse_loss(teacher_feat, student_feat)
            total_loss += mse
            num_levels += 1
        
        # Step 4: Average across all levels
        if num_levels == 0:
            logger.warning("No matching levels between teacher and student features")
            return torch.tensor(0.0, device=teacher_features[list(teacher_features.keys())[0]].device)
        
        avg_loss = total_loss / num_levels
        
        # Step 5: Apply gamma weighting
        return self.gamma * avg_loss


# ============================================================================
# COMBINED DISTILLATION LOSS (Total Training Objective)
# ============================================================================

class DistillationLoss(nn.Module):
    r"""
    Total Knowledge Distillation Loss for Student Training (Phase 2B).
    
    Combines three objectives:
    1. Detection Loss: Ensures student detects objects (primary task)
    2. Logit KD Loss: Student learns class confusion patterns from teacher
    3. Feature KD Loss: Student's spatial features align with teacher
    
    Loss Combination:
    ==================
    Total = α * L_detection + β * L_logit_kd + γ * L_feature_kd
    
    where α, β, γ are configurable weights balancing the three signals.
    
    Typical values:
    ├─ α = 1.0 (detection loss: primary task, must always train)
    ├─ β = 0.5 (logit KD: important but soften with 0.5)
    └─ γ = 1.0 (feature KD: equally important as detection)
    
    Why Not Equal Weights for All?
    ==============================
    
    Detection Loss Characteristics:
    ├─ Sparse targets: Only positive anchors contribute gradient
    ├─ Hard targets: Binary correct/incorrect classification
    ├─ Range: Typically 0.1 - 2.0 depending on task difficulty
    └─ Must always drive learning (α = 1.0)
    
    Logit KD Loss Characteristics:
    ├─ Dense targets: All anchors (positive and negative) contribute
    ├─ Soft targets: Smooth probabilities reveal implicit knowledge
    ├─ Range: Typically 0.01 - 0.5 (depends on temperature and logit ranges)
    ├─ Usually smaller than detection loss
    └─ Can be de-emphasized with β < 1.0
    
    Feature KD Loss Characteristics:
    ├─ Dense targets: All spatial pixels in all pyramid levels
    ├─ Reconstruction: Pixel-wise MSE between feature maps
    ├─ Range: Larger absolute values (depends on feature magnitude)
    ├─ Critical for spatial reasoning transfer
    └─ Needs γ = 1.0 to maintain influence
    
    Args:
        detection_loss_fn: Loss function for detection task
                         Must implement forward(student_out, targets) → scalar
                         Examples: FocalLoss + SmoothL1Loss combined
        
        logit_kd_loss_fn: Loss function for logit KD
                         Must implement forward(student_logits, teacher_logits) → scalar
                         Example: LogitKDLoss()
        
        feature_kd_loss_fn: Loss function for feature KD
                          Must implement forward(teacher_feats, student_feats) → scalar
                          Example: FeatureKDLoss()
        
        alpha (float): Weight for detection loss. Default: 1.0
        beta (float): Weight for logit KD loss. Default: 0.5
        gamma (float): Weight for feature KD loss. Default: 1.0
    
    Forward:
        Returns a dict with all three losses for logging:
        {
            'total_loss': scalar,
            'detection_loss': scalar,
            'logit_kd_loss': scalar,
            'feature_kd_loss': scalar
        }
    """
    
    def __init__(
        self,
        detection_loss_fn: nn.Module,
        logit_kd_loss_fn: nn.Module,
        feature_kd_loss_fn: nn.Module,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 1.0
    ):
        super(DistillationLoss, self).__init__()
        
        self.detection_loss_fn = detection_loss_fn
        self.logit_kd_loss_fn = logit_kd_loss_fn
        self.feature_kd_loss_fn = feature_kd_loss_fn
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def forward(
        self,
        student_detection_out: Dict,
        student_logits: torch.Tensor,
        student_adapted_features: Dict[str, torch.Tensor],
        targets: Dict,
        teacher_logits: torch.Tensor,
        teacher_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined distillation loss.
        
        Args:
            student_detection_out: Detection output dict from student
                                 {
                                   'logits': [B*num_anchors, num_classes],
                                   'boxes': [B*num_anchors, 4],
                                   'features': {level: [B, C, H, W]}
                                 }
            student_logits: Classification logits [N, C] (for logit KD)
            student_adapted_features: Upsampled features for KD [Dict[str, Tensor]]
            targets: Ground truth targets dict
                   {
                     'classes': [B*num_anchors],
                     'boxes': [B*num_anchors, 4],
                     'positive_mask': [B*num_anchors]
                   }
            teacher_logits: Classification logits from teacher [N, C]
            teacher_features: FPN features from teacher [Dict[str, Tensor]]
        
        Returns:
            Dict with losses:
            {
                'total_loss': Combined loss (scalar),
                'detection_loss': Detection loss only (scalar),
                'logit_kd_loss': Logit KD loss only (scalar),
                'feature_kd_loss': Feature KD loss only (scalar)
            }
        """
        
        # Step 1: Detection Loss (primary task)
        # This is like the "baseline" training objective
        detection_loss = self.detection_loss_fn(student_detection_out, targets)
        
        # Step 2: Logit KD Loss (dark knowledge transfer)
        # Filter to only positive anchors for fair comparison
        positive_mask = targets.get('positive_mask', None)
        if positive_mask is not None:
            pos_student_logits = student_logits[positive_mask]
            pos_teacher_logits = teacher_logits[positive_mask]
        else:
            # If no mask, use all (assumes all anchors are supervised)
            pos_student_logits = student_logits
            pos_teacher_logits = teacher_logits
        
        logit_kd_loss = self.logit_kd_loss_fn(pos_student_logits, pos_teacher_logits)
        
        # Step 3: Feature KD Loss (spatial representation alignment)
        feature_kd_loss = self.feature_kd_loss_fn(
            teacher_features,
            student_adapted_features
        )
        
        # Step 4: Combine all losses with their weights
        total_loss = (
            self.alpha * detection_loss +
            self.beta * logit_kd_loss +
            self.gamma * feature_kd_loss
        )
        
        # Step 5: Return all losses for logging and monitoring
        return {
            'total_loss': total_loss,
            'detection_loss': detection_loss,
            'logit_kd_loss': logit_kd_loss,
            'feature_kd_loss': feature_kd_loss
        }


# ============================================================================
# UTILITY FUNCTION: Initialize All KD Losses from Config
# ============================================================================

def create_distillation_loss(config: dict, detection_loss_fn: nn.Module) -> DistillationLoss:
    """
    Factory function to create DistillationLoss from configuration dict.
    
    Args:
        config: Config dict (typically from config.yaml)
               Must contain:
               {
                 'training_student_kd': {
                   'kd': {
                     'temperature': float,
                     'alpha': float,
                     'beta': float,
                     'gamma': float
                   }
                 }
               }
        detection_loss_fn: Pre-initialized detection loss function
    
    Returns:
        DistillationLoss instance ready for training
    
    Example:
        >>> config = yaml.safe_load(open('config.yaml'))
        >>> det_loss = create_detection_loss(config)
        >>> kd_loss = create_distillation_loss(config, det_loss)
    """
    
    kd_config = config.get('training_student_kd', {}).get('kd', {})
    
    temperature = kd_config.get('temperature', 4.0)
    alpha = kd_config.get('alpha', 1.0)
    beta = kd_config.get('beta', 0.5)
    gamma = kd_config.get('gamma', 1.0)
    
    logit_kd = LogitKDLoss(temperature=temperature, alpha=1.0)
    feature_kd = FeatureKDLoss(gamma=1.0)
    
    return DistillationLoss(
        detection_loss_fn=detection_loss_fn,
        logit_kd_loss_fn=logit_kd,
        feature_kd_loss_fn=feature_kd,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )
