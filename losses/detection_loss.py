"""
Detection Loss Functions for Object Detection
==============================================

This module implements the core loss functions used in the object detection pipeline.
Since this is a remote sensing object detection task (DOTA dataset with oriented bounding
boxes), we need TWO separate loss functions that work together:

1. FOCAL LOSS (for classification)
   ├─ Purpose: Predict which class each object belongs to (plane, ship, vehicle, etc.)
   ├─ Challenge: Most anchor boxes contain NO objects (background)
   │  so the loss gets dominated by "easy negatives"
   └─ Solution: Down-weight easy examples, focus on hard ones

2. SMOOTH L1 LOSS (for bounding box regression)
   ├─ Purpose: Fine-tune the exact coordinates of detected objects
   ├─ Challenge: Large coordinate errors cause gradient explosion
   └─ Solution: Use L1 loss for large errors (linear), L2 for small errors (quadratic)

Both losses are combined into a single total detection loss used during training.

Mathematical Background:
========================
In object detection, for each image we generate thousands of "anchor boxes" at
different scales and positions. The network must:

1. Classify each anchor: "Is this anchor background or an object?" (and which class?)
2. Regress each anchor: "If this is an object, where exactly is it?"

Without Focal Loss, the overwhelming number of background anchors would dominate
training, drowning out the signal from actual objects. Focal Loss solves this by
automatically down-weighting easy negatives.

Teacher Model Behavior (High Resolution):
==========================================
- Input: 1024×1024 image with fine details
- Output: Accurate predictions at multiple scales via FPN
- Loss: Will be LOW (model is accurate)
- Training: Converges quickly

Student Model Behavior (Low Resolution):
=========================================
- Input: 128×128 image (8× downsampled, blurry)
- Output: Less accurate predictions
- Loss: Will be HIGHER (model struggles)
- Training: Need to help with knowledge distillation
- In THIS phase: Only detection loss (no KD yet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# FOCAL LOSS - Classification Loss with Hard Negative Mining
# ============================================================================

class FocalLoss(nn.Module):
    r"""
    Focal Loss for addressing class imbalance in object detection.
    
    The standard cross-entropy loss treats all misclassifications equally. In object
    detection, most anchors are background (negative class), so the loss gets
    dominated by easy-to-classify negatives. Focal loss down-weights easy negatives
    and focuses training on hard positives and hard negatives.
    
    Focal Loss Formula:
    ==================
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    where:
        p_t = model probability of the ground truth class
        γ (gamma) = focusing parameter (default 2.0)
        α_t = weight for class t (default 0.25)
    
    The key insight is the (1 - p_t)^γ term:
    
    Example with γ=2:
    ├─ If p_t = 0.99 (correct and confident):
    │  └─ (1 - 0.99)^2 = 0.0001 → Loss scaled DOWN by 10,000×
    │     (easy example, don't waste training time)
    │
    ├─ If p_t = 0.5 (correct but uncertain):
    │  └─ (1 - 0.5)^2 = 0.25 → Loss scaled DOWN by 4×
    │     (moderate example, still easy)
    │
    └─ If p_t = 0.1 (confident but WRONG):
       └─ (1 - 0.1)^2 = 0.81 → Loss scaled UP slightly
          (hard example, model is confused - focus here!)
    
    Why Original Cross-Entropy Fails in Detection:
    ================================================
    In a typical image, you might have:
    - ~50,000 anchor boxes total
    - ~100 boxes with actual objects (positive)
    - ~49,900 background boxes (negative)
    
    Standard Cross-Entropy:
    ├─ Loss from 100 objects: High contribution (but small count)
    └─ Loss from 49,900 background: Dominates training (easy negatives)
       Result: Network overfits to background classification, ignores objects!
    
    Focal Loss:
    ├─ Loss from easy negatives: Auto down-weighted (almost 0)
    ├─ Loss from hard negatives: Full weight (network learns from mistakes)
    └─ Loss from objects: Always full weight (focus here!)
       Result: Balanced training, better object detection!
    
    Args:
        alpha (float): Weight for positive class. Default: 0.25
                      Balances between positive and negative classes.
        gamma (float): Focusing parameter. Default: 2.0
                      Higher γ = more focus on hard examples.
                      γ=0 → regular cross-entropy
                      γ=2 → strong focusing on hard examples
    
    Input Shape:
        inputs: (N, C) where N=number of anchors, C=number of classes
                Raw logits from the classification subnet (before softmax)
        targets: (N,) where each element is class index (0 to C-1)
                0 is usually background for detection tasks
    
    Output:
        loss: scalar tensor, averaged over all anchors
    
    Example:
        >>> focal_loss = FocalLoss()
        >>> logits = torch.randn(1000, 15)  # 1000 anchors, 15 DOTA classes
        >>> targets = torch.randint(0, 15, (1000,))
        >>> loss = focal_loss(logits, targets)
        >>> print(loss.item())  # Single scalar value
        1.234
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Raw classification logits of shape (N, C)
                   N = total number of anchors in batch
                   C = number of classes (e.g., 15 for DOTA)
            targets: Ground truth class indices of shape (N,)
                    Each element is an integer in range [0, C-1]
                    0 = background (no object)
                    1..14 = actual object classes
        
        Returns:
            loss: Scalar focal loss value
        
        Step-by-step computation:
        =========================
        1. Convert logits to probabilities using softmax
        2. Extract probability of ground truth class for each anchor
        3. Compute cross-entropy for each anchor
        4. Apply focal term (1 - p_t)^γ to down-weight easy examples
        5. Average over all anchors
        """
        
        # Step 1: Get softmax probabilities from raw logits
        # This converts raw model outputs to probabilities that sum to 1
        p = F.softmax(inputs, dim=-1)
        
        # Step 2: Extract probability of ground truth class
        # For each anchor, get the probability corresponding to its true class
        # Use 'gather' to select the right probability for each example
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Clamp to prevent log(0) which would give -inf
        # This ensures numerical stability in the log computation
        p_t = p_t.clamp(min=1e-7, max=1.0)
        
        # Step 3: Compute standard cross-entropy
        # This is -log(p_t), measuring how wrong the prediction is
        # If p_t is close to 1 (correct): log(p_t) ≈ 0 → CE ≈ 0 (small loss)
        # If p_t is close to 0 (wrong): log(p_t) ≈ large negative → CE ≈ large (big loss)
        ce_loss = -torch.log(p_t)
        
        # Step 4: Apply focal term - the key innovation!
        # (1 - p_t)^γ down-weights easy examples
        focal_weight = (1 - p_t) ** self.gamma
        
        # Step 5: Combine everything
        # Multiply cross-entropy by focal weight and alpha
        focal_loss = self.alpha * focal_weight * ce_loss
        
        # Return average loss across all anchors
        return focal_loss.mean()


# ============================================================================
# SMOOTH L1 LOSS - Bounding Box Regression Loss
# ============================================================================

class SmoothL1Loss(nn.Module):
    r"""
    Smooth L1 Loss for bounding box regression in object detection.
    
    After classifying which anchors contain objects, we need to fine-tune the
    COORDINATES of those anchor boxes to match the actual ground truth boxes.
    This is the "regression" part of detection.
    
    Why Not Simple L2 (MSE) or L1?
    ===============================
    
    L2 Loss (Mean Squared Error):
    ├─ Formula: loss = (y - y_hat)^2
    ├─ Problem with LARGE errors:
    │  - Error of 10 → loss = 100 (quadratic explosion!)
    │  - Error of 100 → loss = 10,000
    │  - Gradient is 2 * error = 200 for error=100
    │  - Training becomes unstable (gradient explosion)
    └─ Problem with SMALL errors:
       - Error of 0.1 → loss = 0.01 (very small, little gradient)
    
    L1 Loss (Mean Absolute Error):
    ├─ Formula: loss = |y - y_hat|
    ├─ Pro: Gradient is constant (stable even for large errors)
    └─ Con: Non-differentiable at x=0 (kink in the loss curve)
    
    Smooth L1 Loss (BEST):
    ├─ Formula: 
    │  ├─ If |error| < 1:   loss = 0.5 * error^2        (quadratic, smooth)
    │  └─ If |error| ≥ 1:   loss = |error| - 0.5        (linear, stable)
    │
    ├─ Behavior:
    │  ├─ Small errors: Quadratic penalty (precise gradient)
    │  ├─ Large errors: Linear penalty (gradient doesn't explode)
    │  └─ Smooth transition: Differentiable everywhere
    │
    └─ Intuition: Quadratic when you're close (fine-tuning), linear when far away
    
    Visualization of Loss Curves:
    =============================
    
    Vertical axis = loss value
    Horizontal axis = prediction error
    
                   MSE (L2)
                  /
                /
              /
            /_____ Smooth L1 (BEST CHOICE)
                     /
                    / L1
                   /
    
    Position Example:
    =================
    Suppose ground truth box has x-coordinate = 100 pixels.
    
    Scenario 1: Model predicts x = 105 (error = 5 pixels)
    ├─ MSE:       loss = 5^2 = 25,           gradient = 2*5 = 10
    ├─ L1:        loss = |5| = 5,            gradient = 1
    └─ Smooth L1: loss = 0.5*25 = 12.5,      gradient = 5
                  (uses quadratic since |5| > 1 is FALSE, |error| < 1 is TRUE)
    
    Wait, let me recalculate:
    For error=5, since |5| > 1:
    └─ Smooth L1: loss = |5| - 0.5 = 4.5,    gradient ≈ 1 (stable!)
    
    Scenario 2: Model predicts x = 150 (error = 50 pixels - BIG mistake!)
    ├─ MSE:       loss = 50^2 = 2500,        gradient = 2*50 = 100 (HUGE!)
    │             Training goes haywire with such big gradients
    ├─ L1:        loss = |50| = 50,          gradient = 1 (stable)
    └─ Smooth L1: loss = |50| - 0.5 = 49.5,  gradient ≈ 1 (stable, GOOD!)
    
    Why This Matters for Training:
    ==============================
    
    Without Smooth L1 (with L2):
    ├─ Early training: Large coordinate errors produce huge gradients
    ├─ Gradient explosion: Weights explode, training crashes
    ├─ Loss plot: Zigzags erratically, never converges
    └─ Result: Model doesn't learn
    
    With Smooth L1:
    ├─ Early training: Gradients bounded (max ≈ 1)
    ├─ Stable learning: Weights update smoothly
    ├─ Loss plot: Smooth, monotonic decrease
    └─ Result: Model converges well!
    
    Args:
        beta (float): The threshold where Smooth L1 switches from quadratic to linear.
                     Default: 1.0
                     Larger beta = more quadratic (softer), smaller = more linear (harder)
    
    Input Shape:
        pred: (N, M, 4) where N=batch, M=anchors per image, 4=coordinates (x, y, w, h)
              Raw regression outputs from the regression subnet
        target: (N, M, 4) with same shape as pred
               Ground truth coordinates
    
    Output:
        loss: scalar tensor, averaged over all valid boxes
    
    Real-World Example:
    ===================
    In DOTA dataset detecting airplanes:
    
    Ground truth box: center at (512, 512), width=100, height=80
    After detection, we have ~50,000 anchor boxes at various positions.
    Most anchors are far from any airplane.
    
    For anchors FAR from objects (error > 1):
    ├─ Smooth L1 uses linear loss
    ├─ Gradient is small and stable
    ├─ Network learns: "this anchor is far from any object"
    
    For anchors NEAR the airplane (error < 1):
    ├─ Smooth L1 uses quadratic loss
    ├─ Larger gradient pulls network toward precise coordinates
    ├─ Network learns: "fine-tune this anchor to match exactly"
    """
    
    def __init__(self, beta: float = 1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute Smooth L1 loss for bounding box regression.
        
        Args:
            pred: Predicted box coordinates, shape (N, M, 4) or (N*M, 4)
                 Where 4 coordinates are typically (dx, dy, dw, dh)
                 - dx, dy: offsets from anchor center
                 - dw, dh: log scale adjustments
            target: Ground truth coordinates, same shape as pred
            weights: Optional weights for each box, shape matching pred.
                    Anchors with weight=0 are ignored (e.g., background boxes)
                    Anchors with weight=1 are fully counted
        
        Returns:
            loss: Scalar regression loss
        
        The Smooth L1 formula is:
        =========================
        For each coordinate c in each box b:
            error = |pred_b_c - target_b_c|
            
            if error < β:
                loss_b_c = 0.5 * (error^2) / β
            else:
                loss_b_c = error - 0.5 * β
        
        This version uses β to normalize, making the loss scale-dependent.
        """
        
        # Compute absolute difference between prediction and ground truth
        diff = torch.abs(pred - target)
        
        # Create a mask for errors less than beta
        is_small_error = diff < self.beta
        
        # Calculate loss for small errors (quadratic part)
        # Formula: 0.5 * (error^2) / beta
        # This is smooth and has a nice gradient for fine-tuning
        loss_small = 0.5 * (diff ** 2) / self.beta
        
        # Calculate loss for large errors (linear part)
        # Formula: error - 0.5 * beta
        # This is linear with stable gradient for gross errors
        loss_large = diff - 0.5 * self.beta
        
        # Combine: use quadratic for small errors, linear for large
        loss = torch.where(is_small_error, loss_small, loss_large)
        
        # Apply weights if provided (for filtering background boxes)
        if weights is not None:
            # weights should be shape (N*M, 1) or broadcastable
            loss = loss * weights
        
        # Return mean loss
        return loss.mean()


# ============================================================================
# COMBINED DETECTION LOSS
# ============================================================================

class DetectionLoss(nn.Module):
    """
    Combined Detection Loss = Focal Loss + Smooth L1 Loss
    
    This is the main loss function used during Phase 1 (teacher training) and
    Phase 2a (student baseline training). It combines two components:
    
    1. Classification Loss (Focal Loss)
       ├─ Predicts object class or background
       ├─ Uses focal loss to handle class imbalance
       └─ Down-weights easy negatives, focuses on hard positives
    
    2. Regression Loss (Smooth L1)
       ├─ Predicts exact bounding box coordinates
       ├─ Uses smooth L1 to handle coordinate outliers
       └─ Stable gradients for training convergence
    
    The total detection loss is a weighted combination:
    =====================================
    L_detection = λ_cls * L_focal + λ_reg * L_smooth_l1
    
    Typical weights:
    ├─ λ_cls = 1.0 (classification is primary concern)
    └─ λ_reg = 50.0 (regression needs amplification)
    
    Why the 50x amplification for regression?
    ==========================================
    Focal loss and smooth L1 loss have different scales:
    
    Example values:
    ├─ Focal loss: 0.1 to 3.0 (well-behaved, normalized)
    └─ Smooth L1: 0.001 to 0.1 (very small, can be ignored)
    
    If we don't weight regression higher, the network would optimize:
    ├─ 80% for classification (makes correct predictions)
    ├─ 20% for regression (forgets to adjust coordinates)
    └─ Result: Good class predictions, bad bounding boxes!
    
    With 50x weighting:
    ├─ 50 parts regression vs 1 part classification
    ├─ Network balances both concerns
    └─ Result: Both good class AND good boxes!
    
    Training Curve Example:
    ======================
    Epoch | L_cls  | L_reg  | L_det (total) | mAP
    ------|--------|--------|---------------|------
    1     | 1.50   | 0.010  | 1.50          | 0.15
    5     | 0.80   | 0.008  | 0.80          | 0.40
    10    | 0.50   | 0.004  | 0.50          | 0.55
    20    | 0.30   | 0.002  | 0.30          | 0.63
    50    | 0.15   | 0.001  | 0.15          | 0.70
    
    Notice:
    - Both losses decrease together (dual objective)
    - mAP improves as losses decrease (validation metric)
    - Regression loss is naturally small (due to smooth L1 / β normalization)
    
    Args:
        num_classes (int): Number of object classes. For DOTA: 15
        alpha (float): Alpha parameter for focal loss. Default: 0.25
        gamma (float): Gamma (focusing) parameter for focal loss. Default: 2.0
        beta (float): Beta parameter for smooth L1 loss. Default: 1.0
        cls_weight (float): Weight for classification loss. Default: 1.0
        reg_weight (float): Weight for regression loss. Default: 50.0
    
    Example Usage:
    ==============
    >>> loss_fn = DetectionLoss(num_classes=15)
    >>>
    >>> # During training, for each batch:
    >>> cls_logits = model.cls_head(features)      # (B*num_anchors, 15)
    >>> bbox_preds = model.bbox_head(features)     # (B*num_anchors, 4)
    >>> cls_targets = grab_targets_from_batch()    # (B*num_anchors,)
    >>> bbox_targets = grab_bbox_targets()         # (B*num_anchors, 4)
    >>>
    >>> total_loss = loss_fn(
    ...     cls_logits, cls_targets,
    ...     bbox_preds, bbox_targets
    ... )
    >>> print(f"Loss: {total_loss.item():.4f}")  # Single scalar
    Loss: 0.2345
    """
    
    def __init__(
        self,
        num_classes: int = 15,
        alpha: float = 0.25,
        gamma: float = 2.0,
        beta: float = 1.0,
        cls_weight: float = 1.0,
        reg_weight: float = 50.0
    ):
        super(DetectionLoss, self).__init__()
        
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        
        # Initialize individual loss components
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.smooth_l1_loss = SmoothL1Loss(beta=beta)
    
    def forward(
        self,
        cls_logits: torch.Tensor,
        cls_targets: torch.Tensor,
        bbox_preds: torch.Tensor,
        bbox_targets: torch.Tensor,
        bbox_weights: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total detection loss.
        
        Args:
            cls_logits: Classification logits from the network.
                       Shape: (N, num_classes) where N = total anchors in batch
                       These are raw scores, will be passed to softmax in focal loss
            cls_targets: Ground truth class indices.
                        Shape: (N,) with values in [0, num_classes-1]
                        0 = background (no object)
                        1..14 = actual classes
            bbox_preds: Predicted bounding box offsets/scales.
                       Shape: (N, 4) with values [dx, dy, dw, dh]
                       These are offsets relative to anchor boxes
            bbox_targets: Ground truth bounding box offsets/scales.
                         Shape: (N, 4) with same meaning as bbox_preds
            bbox_weights: Optional weights for each bounding box.
                         Shape: (N, 1) or (N,)
                         Use 1.0 for boxes to train on (objects)
                         Use 0.0 for boxes to ignore (background)
        
        Returns:
            losses_dict: Dictionary with keys:
                - 'loss_cls': Classification (focal) loss
                - 'loss_reg': Regression (smooth L1) loss
                - 'loss_det': Total combined loss
        
        Step-by-step Computation:
        ==========================
        1. Compute focal loss on classification logits
           - Penalizes misclassified anchors
           - Ignores easy negatives (background far from objects)
        
        2. Compute smooth L1 loss on bounding boxes
           - Penalizes coordinate mismatch
           - Uses quadratic for small errors, linear for large
          
        3. Combine with weights
           - Ensure both components contribute fairly
           - Scaled so both converge at similar rates
        
        4. Return components separately and combined
           - Allows monitoring training dynamics
        """
        
        # Step 1: Compute classification loss (focal loss)
        loss_cls = self.focal_loss(cls_logits, cls_targets)
        
        # Step 2: Compute regression loss (smooth L1)
        loss_reg = self.smooth_l1_loss(bbox_preds, bbox_targets, bbox_weights)
        
        # Step 3: Combine with weights
        # This ensures both components contribute meaningfully
        loss_det = self.cls_weight * loss_cls + self.reg_weight * loss_reg
        
        # Step 4: Return dict with all components
        # Useful for monitoring during training
        return {
            'loss_cls': loss_cls,
            'loss_reg': loss_reg,
            'loss_det': loss_det
        }


# ============================================================================
# SIMPLE TEST / DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    """
    Simple test to verify the loss functions work correctly.
    
    This code can be run standalone to understand how the losses work.
    """
    
    print("=" * 80)
    print("DETECTION LOSS COMPONENTS TEST")
    print("=" * 80)
    
    # Test parameters
    batch_size = 2
    num_anchors_per_image = 9
    num_classes = 15
    total_anchors = batch_size * num_anchors_per_image
    
    # Create dummy data
    print("\n1. Creating dummy data...")
    print(f"   - Batch size: {batch_size} images")
    print(f"   - Anchors per image: {num_anchors_per_image}")
    print(f"   - Total anchors: {total_anchors}")
    print(f"   - Number of classes: {num_classes}")
    
    # Random classification logits (would come from network's classification branch)
    cls_logits = torch.randn(total_anchors, num_classes)
    print(f"\n   - Classification logits shape: {cls_logits.shape}")
    print(f"     Range: [{cls_logits.min():.2f}, {cls_logits.max():.2f}]")
    
    # Random classification targets (ground truth)
    cls_targets = torch.randint(0, num_classes, (total_anchors,))
    print(f"   - Classification targets shape: {cls_targets.shape}")
    print(f"     Values: {torch.unique(cls_targets).tolist()}")
    
    # Random bbox predictions (would come from network's regression branch)
    bbox_preds = torch.randn(total_anchors, 4)
    print(f"\n   - BBox predictions shape: {bbox_preds.shape}")
    print(f"     Range: [{bbox_preds.min():.2f}, {bbox_preds.max():.2f}]")
    
    # Random bbox targets
    bbox_targets = torch.randn(total_anchors, 4)
    print(f"   - BBox targets shape: {bbox_targets.shape}")
    print(f"     Range: [{bbox_targets.min():.2f}, {bbox_targets.max():.2f}]")
    
    # Test individual losses
    print("\n" + "=" * 80)
    print("2. TESTING FOCAL LOSS")
    print("=" * 80)
    
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    loss_focal = focal_loss_fn(cls_logits, cls_targets)
    print(f"\nFocal Loss: {loss_focal.item():.6f}")
    print("This measures how well the network classifies objects.")
    print("Lower is better (model is confident and correct).")
    
    print("\n" + "=" * 80)
    print("3. TESTING SMOOTH L1 LOSS")
    print("=" * 80)
    
    smooth_l1_fn = SmoothL1Loss(beta=1.0)
    loss_smooth_l1 = smooth_l1_fn.forward(bbox_preds, bbox_targets)
    print(f"\nSmooth L1 Loss: {loss_smooth_l1.item():.6f}")
    print("This measures how accurate the bounding box coordinates are.")
    print("Lower is better (coordinates match ground truth).")
    
    print("\n" + "=" * 80)
    print("4. TESTING COMBINED DETECTION LOSS")
    print("=" * 80)
    
    detection_loss_fn = DetectionLoss(
        num_classes=num_classes,
        alpha=0.25,
        gamma=2.0,
        beta=1.0,
        cls_weight=1.0,
        reg_weight=50.0
    )
    
    losses_dict = detection_loss_fn(
        cls_logits, cls_targets,
        bbox_preds, bbox_targets
    )
    
    print(f"\nClassification Loss (L_cls): {losses_dict['loss_cls'].item():.6f}")
    print(f"Regression Loss (L_reg):      {losses_dict['loss_reg'].item():.6f}")
    print(f"Total Detection Loss (L_det): {losses_dict['loss_det'].item():.6f}")
    
    print("\n" + "=" * 80)
    print("5. INTERPRETATION")
    print("=" * 80)
    print("""
During training, we want to minimize the total detection loss.
    
This total loss is used with an optimizer (like Adam or SGD) to update the
network weights via backpropagation. In Phase 1 (teacher training), this is
the ONLY loss we use.

The network learns:
├─ Through L_cls: Which anchors have objects, what class they are
└─ Through L_reg: Where to position anchor boxes around objects

The 50× weighting ensures both concerns are balanced. If we didn't weight
regression, the network would focus only on classification and lose precision
on coordinate prediction.

Next Step:
==========
In Phase 2b (student with knowledge distillation), we'll ADD two more losses:
├─ L_logit_kd: Student matches teacher's "soft" class predictions
└─ L_feature_kd: Student's adapted features match teacher's features

But those are beyond the scope of this file. This file only handles the
fundamental detection loss needed for all three training phases.
    """)
    print("=" * 80)
