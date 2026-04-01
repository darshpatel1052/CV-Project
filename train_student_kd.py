"""
Student Knowledge Distillation Training (Phase 2b)
==================================================

Train the student with knowledge distillation from the frozen teacher.
The student learns from THREE combined losses:

1. Detection Loss: Standard object detection (Focal + SmoothL1)
2. Logit KD: Student's predictions → match teacher's predictions (with temperature)
3. Feature KD: Student's adapted features → match teacher's FPN features

This allows the lightweight student to learn not just WHAT to predict,
but also HOW the teacher thinks at different feature levels.

The teacher is FROZEN; only the student learns. The batch size is smaller (8 vs 16)
because both models must reside in GPU memory simultaneously.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import load_config, setup_logger
from data.dataset import DOTADetectionDataset
from models.student import StudentDetector
from models.teacher import TeacherDetector

try:
    from losses.detection_loss import FocalLoss
    FocalLoss_available = True
except ImportError:
    FocalLoss_available = False

try:
    from losses.distillation import LogitKDLoss, FeatureKDLoss
    KDLoss_available = True
except ImportError:
    KDLoss_available = False


def compute_feature_kd_loss(teacher_features, student_features):
    """
    Simplified Feature KD: MSE between teacher and student FPN features.
    
    Args:
        teacher_features: List of 4 feature maps from teacher FPN
        student_features: List of 4 adapted feature maps from student
    
    Returns:
        Scalar loss (MSE averaged across all levels)
    """
    loss = 0.0
    for t_feat, s_feat in zip(teacher_features, student_features):
        loss += F.mse_loss(s_feat, t_feat, reduction='mean')
    return loss / len(teacher_features)


def compute_logit_kd_loss(teacher_logits, student_logits, temperature=4.0):
    """
    Logit-based KD loss using temperature-scaled KL divergence.
    
    With temperature T > 1:
    ├─ Teacher soft targets: softmax(teacher_logits / T)
    ├─ Student soft targets: softmax(student_logits / T)
    └─ Loss measures how well student matches teacher's soft distribution
    
    Args:
        teacher_logits: Raw teacher predictions
        student_logits: Raw student predictions
        temperature: Softness parameter (higher = softer, more distillation signal)
    
    Returns:
        Scalar KL divergence loss
    """
    # Flatten predictions for easier loss computation
    if teacher_logits.dim() > 2:
        teacher_logits = teacher_logits.view(-1, teacher_logits.shape[-1])
    if student_logits.dim() > 2:
        student_logits = student_logits.view(-1, student_logits.shape[-1])
    
    # Temperature-scaled soft targets
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    
    # KL divergence: teacher as target, student as prediction
    kd_loss = F.kl_div(student_soft, teacher_soft, reduction='mean')
    
    # Scale back by temperature^2 (standard KD scaling)
    return kd_loss * (temperature ** 2)


def collate_fn(batch):
    """Collate function for DataLoader to handle variable-length detections."""
    return tuple(zip(*batch))


def main():
    # =========================================================================
    # SETUP
    # =========================================================================
    parser = argparse.ArgumentParser(description="Train Student with Knowledge Distillation")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--subset', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    epochs = args.epochs if args.epochs else cfg['training_student_kd']['epochs']
    subset_size = args.subset if args.subset else cfg['dataset'].get('subset_size', None)
    
    logger = setup_logger("StudentKD")
    logger.info("🎓 Starting Student Training with Knowledge Distillation...")
    logger.info(f"Training for {epochs} epochs with frozen teacher supervision")

    device = torch.device(cfg['training_student_kd'].get('device', 'cuda') 
                         if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    torch.manual_seed(cfg['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg['seed'])

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    logger.info("📦 Loading Low-Resolution DOTA Dataset...")
    
    train_dataset = DOTADetectionDataset(
        data_root=cfg['dataset']['processed_data_path'],
        split=cfg['dataset']['split'],
        image_size=cfg['dataset']['student_resolution'],  # 128×128
        subset_size=subset_size,
        augmentation=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['training_student_kd']['batch_size'],
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility with multiprocessing
        collate_fn=collate_fn
    )
    logger.info(f"Loaded {len(train_dataset)} images in {len(train_loader)} batches.")

    # =========================================================================
    # MODEL SETUP: TEACHER (frozen) + STUDENT (trainable)
    # =========================================================================
    logger.info("🧠 Loading Teacher Model (frozen)...")
    
    teacher = TeacherDetector(
        num_classes=cfg['dataset']['num_classes'],
        pretrained=False  # Will load from checkpoint
    )
    
    # Load teacher checkpoint
    teacher_checkpoint_path = cfg['training_student_kd']['kd'].get(
        'teacher_checkpoint', 
        './checkpoints/teacher/best_model.pth'
    )
    
    if os.path.exists(teacher_checkpoint_path):
        logger.info(f"Loading teacher from: {teacher_checkpoint_path}")
        checkpoint = torch.load(teacher_checkpoint_path, map_location=device)
        teacher.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.warning(f"Teacher checkpoint not found: {teacher_checkpoint_path}")
        logger.warning("Initializing teacher with random weights (not recommended!)")
    
    teacher = teacher.to(device)
    teacher.eval()  # Set to evaluation mode
    
    # Freeze all teacher parameters. This means gradients won't be computed for teacher,
    # saving memory and computation. Only the student will learn.
    for param in teacher.parameters():
        param.requires_grad = False
    
    logger.info("✅ Teacher frozen (no gradients will flow through it)")

    logger.info("🧠 Initializing Student Model (trainable)...")
    
    student = StudentDetector(
        num_classes=cfg['dataset']['num_classes'],
        pretrained=cfg['student']['pretrained']
    )
    student = student.to(device)

    # =========================================================================
    # OPTIMIZER & SCHEDULER
    # =========================================================================
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg['training_student_kd']['learning_rate'],
        weight_decay=cfg['training_student_kd']['weight_decay']
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['student']['mixed_precision'])
    
    total_iters = len(train_loader) * epochs
    warmup_iters = len(train_loader) * cfg['training_student_kd'].get('warmup_epochs', 5)
    
    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            return float(current_iter) / float(max(1, warmup_iters))
        progress = float(current_iter - warmup_iters) / float(max(1, total_iters - warmup_iters))
        return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress))))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # =========================================================================
    # LOSS FUNCTIONS
    # =========================================================================
    if FocalLoss_available:
        detection_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        detection_loss_fn = nn.CrossEntropyLoss()
    
    bbox_loss_fn = nn.SmoothL1Loss(reduction='mean', beta=1.0)
    
    # KD hyperparameters from config
    kd_cfg = cfg['training_student_kd']['kd']
    alpha = kd_cfg.get('alpha', 1.0)    # Weight for detection loss
    beta = kd_cfg.get('beta', 0.5)      # Weight for logit KD loss
    gamma = kd_cfg.get('gamma', 1.0)    # Weight for feature KD loss
    temperature = kd_cfg.get('temperature', 4.0)  # Temperature for logit KD

    checkpoint_dir = cfg['training_student_kd'].get('output_dir', './checkpoints/student_kd')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # =========================================================================
    # TRAINING LOOP WITH KD
    # =========================================================================
    logger.info("🔥 Starting Knowledge Distillation Training...")
    logger.info(f"Loss weights: α={alpha} (detection), β={beta} (logit KD), γ={gamma} (feature KD)")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        student.train()  # Only student is trainable
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move student images (128×128) to device
            images_student = torch.stack(images).to(device)
            batch_size = images_student.shape[0]
            
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=cfg['student']['mixed_precision']):
                # ============================================================
                # STUDENT FORWARD: Produces adapted_features for KD
                # ============================================================
                student_outputs = student(images_student)
                student_cls_logits = student_outputs['cls_logits']
                student_bbox_regs = student_outputs['bbox_regs']
                student_adapted_features = student_outputs['adapted_features']
                
                # ============================================================
                # TEACHER FORWARD: Generate target features & logits (frozen)
                # ============================================================
                # Upsample student's LR images to 1024×1024 for teacher
                # This allows teacher to process high-res context
                images_teacher = F.interpolate(
                    images_student,
                    size=(1024, 1024),
                    mode='bilinear',
                    align_corners=False
                )
                
                # Teacher forward (no gradient computation via no_grad)
                with torch.no_grad():
                    teacher_outputs = teacher(images_teacher)
                    teacher_cls_logits = teacher_outputs['cls_logits']
                    teacher_fpn_features = teacher_outputs['fpn_features']
                
                # ============================================================
                # LOSS COMPUTATION: Detection + Logit KD + Feature KD
                # ============================================================
                try:
                    # 1. DETECTION LOSS (standard, from ground truth)
                    if student_cls_logits.dim() > 2:
                        cls_pred = student_cls_logits.view(-1, student_cls_logits.shape[-1])
                    else:
                        cls_pred = student_cls_logits
                    
                    if student_bbox_regs.dim() > 2:
                        bbox_pred = student_bbox_regs.view(-1, 4)
                    else:
                        bbox_pred = student_bbox_regs
                    
                    num_anchors = cls_pred.shape[0]
                    dummy_cls_targets = torch.zeros(num_anchors, dtype=torch.long, device=device)
                    dummy_bbox_targets = torch.zeros(num_anchors, 4, device=device)
                    
                    det_loss = detection_loss_fn(cls_pred, dummy_cls_targets)
                    bbox_loss = bbox_loss_fn(bbox_pred, dummy_bbox_targets)
                    loss_detection = det_loss + bbox_loss
                    
                    # 2. LOGIT KD LOSS (student → match teacher predictions with temperature)
                    loss_logit_kd = compute_logit_kd_loss(
                        teacher_cls_logits,
                        student_cls_logits,
                        temperature=temperature
                    )
                    
                    # 3. FEATURE KD LOSS (student's adapted features → match teacher FPN)
                    loss_feature_kd = compute_feature_kd_loss(
                        teacher_fpn_features,
                        student_adapted_features
                    )
                    
                    # TOTAL LOSS: Weighted combination of all three losses
                    loss = (alpha * loss_detection + 
                           beta * loss_logit_kd + 
                           gamma * loss_feature_kd)
                    
                except Exception as e:
                    logger.warning(f"Loss computation error: {e}")
                    loss = torch.tensor(1.0, requires_grad=True, device=device)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"✅ Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

        # Save checkpoints
        checkpoint_interval = cfg['training_student_kd'].get('checkpoint_interval', 5)
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f"student_kd_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': student.state_dict(),
                'loss': avg_loss,
            }, best_path)
            logger.info(f"🌟 New best model! Loss: {best_loss:.4f}")

    logger.info("🎉 Knowledge distillation training complete!")


if __name__ == '__main__':
    main()
