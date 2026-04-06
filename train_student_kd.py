"""
Student Knowledge Distillation Training (Phase 2b)
==================================================

Train the student with knowledge distillation from the frozen teacher.
Three combined losses:
  1. Detection Loss: Standard (Focal + SmoothL1) with proper anchor matching
  2. Logit KD: Student predictions → match teacher predictions (temperature scaled)
  3. Feature KD: Student adapted features → match teacher FPN features (MSE)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import (
    load_config, setup_logger, generate_all_anchors,
    match_anchors_to_targets, postprocess_detections, compute_map
)
from data.dataset import DOTADetectionDataset
from models.student import StudentDetector
from models.teacher import TeacherDetector
from losses.detection_loss import FocalLoss, SmoothL1Loss
from train_teacher import compute_detection_loss, validate


def compute_feature_kd_loss(teacher_features, student_features):
    """MSE between teacher and student FPN features (averaged across levels)."""
    loss = 0.0
    count = 0
    for t_feat, s_feat in zip(teacher_features, student_features):
        # Ensure spatial dimensions match (they should from adapters)
        if t_feat.shape != s_feat.shape:
            s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=False)
        loss += F.mse_loss(s_feat, t_feat, reduction='mean')
        count += 1
    return loss / max(count, 1)


def compute_logit_kd_loss(teacher_logits, student_logits, temperature=4.0):
    """KL divergence between temperature-scaled soft targets."""
    if teacher_logits.dim() > 2:
        teacher_logits = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    if student_logits.dim() > 2:
        student_logits = student_logits.reshape(-1, student_logits.shape[-1])

    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)

    kd_loss = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean')
    return kd_loss * (temperature ** 2)


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    parser = argparse.ArgumentParser(description="Train Student with Knowledge Distillation")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--subset', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    epochs = args.epochs if args.epochs else cfg['training_student_kd']['epochs']
    subset_size = args.subset if args.subset else cfg['dataset'].get('subset_size', None)

    logger = setup_logger("StudentKD")
    logger.info("Starting Student KD Training")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg['seed'])

    num_classes = cfg['dataset']['num_classes']
    student_image_size = cfg['dataset']['student_resolution']
    teacher_image_size = cfg['dataset']['teacher_resolution']

    # =========================================================================
    # DATA
    # =========================================================================
    train_dataset = DOTADetectionDataset(
        data_root=cfg['dataset']['processed_data_path'],
        split='train', image_size=student_image_size,
        subset_size=subset_size, augmentation=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['training_student_kd']['batch_size'],
        shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True
    )

    val_subset = min(200, subset_size) if subset_size else 200
    val_dataset = DOTADetectionDataset(
        data_root=cfg['dataset']['processed_data_path'],
        split='val', image_size=student_image_size,
        subset_size=val_subset, augmentation=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=max(1, cfg['training_student_kd']['batch_size'] // 2),
        shuffle=False, num_workers=2, collate_fn=collate_fn, pin_memory=True
    )
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # =========================================================================
    # MODELS: Teacher (frozen) + Student (trainable)
    # =========================================================================
    logger.info("Loading frozen teacher...")
    teacher = TeacherDetector(num_classes=num_classes, pretrained=False)

    teacher_ckpt = cfg['training_student_kd'].get('teacher_checkpoint',
                                                    './checkpoints/teacher/best_model.pth')
    if os.path.exists(teacher_ckpt):
        ckpt = torch.load(teacher_ckpt, map_location=device)
        # Filter out mismatching keys
        model_dict = teacher.state_dict()
        filtered_dict = {
            k: v for k, v in ckpt['model_state_dict'].items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(filtered_dict)
        teacher.load_state_dict(model_dict)
        logger.info(f"Loaded teacher from {teacher_ckpt} (matched keys)")
    else:
        logger.warning(f"Teacher checkpoint not found: {teacher_ckpt}. Using random weights!")

    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    logger.info("Initializing student...")
    student = StudentDetector(num_classes=num_classes, pretrained=cfg['student']['pretrained'])
    student = student.to(device)

    # =========================================================================
    # ANCHORS for student detection loss
    # =========================================================================
    with torch.no_grad():
        dummy = torch.randn(1, 3, student_image_size, student_image_size, device=device)
        dummy_out = student(dummy)
        adapted_sizes = [f.shape[-1] for f in dummy_out['adapted_features']]
        total_anchors = dummy_out['cls_logits'].shape[1]

    fpn_strides = [student_image_size // s for s in adapted_sizes]
    anchors = generate_all_anchors(student_image_size, fpn_strides, base_sizes=[s * 4 for s in fpn_strides])
    assert anchors.shape[0] == total_anchors
    logger.info(f"Student anchors: {anchors.shape[0]}, strides: {fpn_strides}")

    # =========================================================================
    # OPTIMIZER & LOSS
    # =========================================================================
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg['training_student_kd']['learning_rate'],
        weight_decay=cfg['training_student_kd']['weight_decay']
    )

    use_amp = cfg['student'].get('mixed_precision', True) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    total_iters = len(train_loader) * epochs
    warmup_iters = len(train_loader) * cfg['training_student_kd'].get('warmup_epochs', 5)

    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            return float(current_iter) / float(max(1, warmup_iters))
        import math
        progress = float(current_iter - warmup_iters) / float(max(1, total_iters - warmup_iters))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    bbox_loss_fn = SmoothL1Loss(beta=1.0 / 9.0)

    kd_cfg = cfg['training_student_kd']['kd']
    alpha = kd_cfg.get('alpha', 1.0)
    beta = kd_cfg.get('beta', 0.5)
    gamma = kd_cfg.get('gamma', 1.0)
    temperature = kd_cfg.get('temperature', 4.0)

    checkpoint_dir = cfg['training_student_kd'].get('output_dir', './checkpoints/student_kd')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    logger.info(f"Training {epochs} epochs | α={alpha} β={beta} γ={gamma} T={temperature}")
    best_map = 0.0

    for epoch in range(epochs):
        student.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, targets in pbar:
            images_student = torch.stack(images).to(device)
            targets_list = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                # Student forward
                student_out = student(images_student)

                # Teacher forward (upsample student images to teacher resolution)
                images_teacher = F.interpolate(
                    images_student, size=(teacher_image_size, teacher_image_size),
                    mode='bilinear', align_corners=False
                )
                with torch.no_grad():
                    teacher_out = teacher(images_teacher)

                # 1. Detection loss (with proper anchor matching)
                loss_cls, loss_reg = compute_detection_loss(
                    student_out['cls_logits'], student_out['bbox_regs'],
                    anchors, targets_list, device,
                    focal_loss_fn, bbox_loss_fn, num_classes
                )
                loss_det = loss_cls + loss_reg

                # 2. Logit KD loss
                loss_logit_kd = compute_logit_kd_loss(
                    teacher_out['cls_logits'], student_out['cls_logits'],
                    temperature=temperature
                )

                # 3. Feature KD loss
                loss_feature_kd = compute_feature_kd_loss(
                    teacher_out['fpn_features'], student_out['adapted_features']
                )

                # Combined loss
                loss = alpha * loss_det + beta * loss_logit_kd + gamma * loss_feature_kd

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({
                'det': f"{loss_det.item():.3f}",
                'kd_l': f"{loss_logit_kd.item():.3f}",
                'kd_f': f"{loss_feature_kd.item():.3f}",
            })

        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        # Validation
        checkpoint_interval = cfg['training_student_kd'].get('checkpoint_interval', 5)
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == epochs:
            logger.info("Validating...")
            mAP, _ = validate(student, val_loader, anchors, device, num_classes)
            logger.info(f"Val mAP@0.5: {mAP:.4f}")

            ckpt_path = os.path.join(checkpoint_dir, f"student_kd_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss, 'mAP': mAP,
            }, ckpt_path)

            if mAP > best_map:
                best_map = mAP
                best_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': student.state_dict(),
                    'loss': avg_loss, 'mAP': mAP,
                }, best_path)
                logger.info(f"New best! mAP: {best_map:.4f}")

    logger.info(f"KD training complete! Best mAP: {best_map:.4f}")


if __name__ == '__main__':
    main()
