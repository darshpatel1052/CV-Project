"""
Teacher Model Training Pipeline (Phase 1)
==========================================

Trains the high-capacity Swin-T + FPN + RetinaNet teacher on HR (1024×1024) DOTA images
using proper anchor-based detection: IoU matching, Focal Loss for classification,
Smooth L1 for regression. Includes validation loop with real mAP tracking.
"""

import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import (
    load_config, setup_logger, generate_all_anchors,
    match_anchors_to_targets, postprocess_detections, compute_map
)
from data.dataset import DOTADetectionDataset
from models.teacher import TeacherDetector
from losses.detection_loss import FocalLoss, SmoothL1Loss


def collate_fn(batch):
    """Collate function for variable-length detections."""
    return tuple(zip(*batch))


def compute_detection_loss(
    cls_logits, bbox_regs, anchors, targets, device,
    focal_loss_fn, bbox_loss_fn, num_classes=15
):
    """
    Compute detection loss with proper anchor-GT matching.

    Args:
        cls_logits: (B, total_anchors, num_classes+1) including background
        bbox_regs: (B, total_anchors, 4)
        anchors: (total_anchors, 4) pre-generated anchors
        targets: list of target dicts with 'boxes' and 'labels'
        device: torch device
        focal_loss_fn: FocalLoss instance
        bbox_loss_fn: SmoothL1Loss instance
        num_classes: number of object classes

    Returns:
        loss_cls, loss_reg: scalar losses
    """
    batch_size = cls_logits.shape[0]
    all_cls_losses = []
    all_reg_losses = []

    for b in range(batch_size):
        gt_boxes = targets[b]['boxes'].to(device)
        gt_labels = targets[b]['labels'].to(device)

        # Match anchors to GT
        cls_targets, reg_targets, pos_mask, neg_mask = match_anchors_to_targets(
            anchors.to(device), gt_boxes, gt_labels,
            pos_iou_thresh=0.5, neg_iou_thresh=0.4,
            num_classes=num_classes
        )

        # Classification loss: use positive + negative anchors (skip ignored)
        valid_mask = pos_mask | neg_mask
        if valid_mask.sum() > 0:
            cls_loss = focal_loss_fn(
                cls_logits[b][valid_mask],
                cls_targets[valid_mask]
            )
            all_cls_losses.append(cls_loss)

        # Regression loss: only positive anchors
        if pos_mask.sum() > 0:
            reg_loss = bbox_loss_fn(
                bbox_regs[b][pos_mask],
                reg_targets[pos_mask].to(device)
            )
            all_reg_losses.append(reg_loss)

    # Average across batch
    if all_cls_losses:
        loss_cls = torch.stack(all_cls_losses).mean()
    else:
        loss_cls = torch.tensor(0.0, device=device, requires_grad=True)

    if all_reg_losses:
        loss_reg = torch.stack(all_reg_losses).mean()
    else:
        loss_reg = torch.tensor(0.0, device=device, requires_grad=True)

    return loss_cls, loss_reg


@torch.no_grad()
def validate(model, val_loader, anchors, device, num_classes=15):
    """Run validation and compute mAP."""
    model.eval()
    all_predictions = []
    all_targets = []

    for images, targets in val_loader:
        images = torch.stack(images).to(device)
        targets_dev = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in t.items()} for t in targets]

        outputs = model(images)
        preds = postprocess_detections(
            outputs['cls_logits'], outputs['bbox_regs'],
            anchors.to(device), image_size=images.shape[-1],
            num_classes=num_classes,
            conf_threshold=0.05, nms_threshold=0.5,
            use_background=False  # sigmoid focal loss, no background class
        )

        all_predictions.extend(preds)
        all_targets.extend(targets_dev)

    mAP, per_class_ap = compute_map(all_predictions, all_targets,
                                     num_classes=num_classes, iou_threshold=0.5)
    model.train()
    return mAP, per_class_ap


def main():
    parser = argparse.ArgumentParser(description="Train the Teacher Object Detector")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--subset', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    epochs = args.epochs if args.epochs else cfg['training_teacher']['epochs']
    subset_size = args.subset if args.subset else cfg['dataset'].get('subset_size', None)

    logger = setup_logger("TrainTeacher")
    logger.info("Starting Teacher Training Pipeline")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    torch.manual_seed(cfg['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg['seed'])

    num_classes = cfg['dataset']['num_classes']
    image_size = cfg['dataset']['teacher_resolution']

    # -------------------------------------------------------------------------
    # DATA
    # -------------------------------------------------------------------------
    logger.info("Loading HR DOTA Dataset...")
    train_dataset = DOTADetectionDataset(
        data_root=cfg['dataset']['processed_data_path'],
        split='train', image_size=image_size,
        subset_size=subset_size, augmentation=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['training_teacher']['batch_size'],
        shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True
    )

    # Validation set
    val_subset = min(200, subset_size) if subset_size else 200
    val_dataset = DOTADetectionDataset(
        data_root=cfg['dataset']['processed_data_path'],
        split='val', image_size=image_size,
        subset_size=val_subset, augmentation=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=max(1, cfg['training_teacher']['batch_size'] // 2),
        shuffle=False, num_workers=2, collate_fn=collate_fn, pin_memory=True
    )
    logger.info(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------
    model = TeacherDetector(num_classes=num_classes, pretrained=cfg['teacher']['pretrained'])
    model = model.to(device)

    # -------------------------------------------------------------------------
    # ANCHORS (pre-generate once, reuse every batch)
    # -------------------------------------------------------------------------
    # The teacher's detection head processes adapted features with strides [8,16,32,64]
    # w.r.t. the 1024×1024 input. But cls_logits has num_classes+1 outputs because
    # FocalLoss uses class 0 as background.
    # However, the RetinaNetHead outputs num_classes (15) directly (no +1).
    # We need to handle this: the head outputs (B, total_anchors, 15) for cls
    # and matching returns targets in [0=bg, 1..15=classes].
    # So we need num_classes+1 in the focal loss, or we shift matching.
    
    # Generate anchors matching the FPN feature map sizes
    # For 1024 input with strides [8,16,32,64]:
    # P3: 128x128, P4: 64x64, P5: 32x32, P6: 16x16
    # But our FPN uses backbone at strides [4,8,16,32] producing C1..C4
    # Then FPN maps: P3 from C2 (1/8), P4 from C3 (1/16), P5 from C4 (1/32), P6 via pool (1/64)
    # So effective strides for the teacher are [8, 16, 32, 64] relative to input.
    
    # Let's verify by running a dummy forward pass
    with torch.no_grad():
        dummy = torch.randn(1, 3, image_size, image_size, device=device)
        dummy_out = model(dummy)
        fpn_sizes = [f.shape[-1] for f in dummy_out['fpn_features']]
        total_anchors_from_model = dummy_out['cls_logits'].shape[1]
        num_cls_output = dummy_out['cls_logits'].shape[2]
        logger.info(f"FPN feature map sizes: {fpn_sizes}")
        logger.info(f"Total anchors from model: {total_anchors_from_model}")
        logger.info(f"Cls output channels: {num_cls_output}")

    # Compute actual strides from FPN sizes
    fpn_strides = [image_size // s for s in fpn_sizes]
    logger.info(f"Computed FPN strides: {fpn_strides}")

    anchors = generate_all_anchors(image_size, fpn_strides, base_sizes=[s * 4 for s in fpn_strides])
    logger.info(f"Generated {anchors.shape[0]} anchors (should match {total_anchors_from_model})")
    assert anchors.shape[0] == total_anchors_from_model, \
        f"Anchor count mismatch: {anchors.shape[0]} vs {total_anchors_from_model}"

    # -------------------------------------------------------------------------
    # OPTIMIZER & LOSS
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training_teacher']['learning_rate'],
        weight_decay=cfg['training_teacher']['weight_decay']
    )

    use_amp = cfg['teacher'].get('mixed_precision', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    total_iters = len(train_loader) * epochs
    warmup_iters = len(train_loader) * cfg['training_teacher'].get('warmup_epochs', 5)

    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            return float(current_iter) / float(max(1, warmup_iters))
        progress = float(current_iter - warmup_iters) / float(max(1, total_iters - warmup_iters))
        import math
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Detection head outputs num_classes channels (no explicit background).
    # Our matching produces targets: 0=bg, 1..num_classes=objects.
    # So focal loss needs num_classes+1 classes.
    # BUT: the RetinaNetHead currently outputs num_classes (15), not 16.
    # So we need focal loss with 15 classes and shift targets back.
    # Alternative: use sigmoid focal loss per-class (standard for RetinaNet).
    # Let's use sigmoid focal loss for proper RetinaNet behavior.
    
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    bbox_loss_fn = SmoothL1Loss(beta=1.0 / 9.0)

    checkpoint_dir = cfg['training_teacher'].get('output_dir', './checkpoints/teacher')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------------------
    logger.info(f"Starting training for {epochs} epochs...")
    best_map = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_cls_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (images, targets) in enumerate(pbar):
            images = torch.stack(images).to(device)
            targets_list = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images)
                cls_logits = outputs['cls_logits']  # (B, N, num_classes)
                bbox_regs = outputs['bbox_regs']    # (B, N, 4)

                loss_cls, loss_reg = compute_detection_loss(
                    cls_logits, bbox_regs, anchors, targets_list,
                    device, focal_loss_fn, bbox_loss_fn, num_classes
                )

                loss = loss_cls + loss_reg

            scaler.scale(loss).backward()

            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_cls_loss += loss_cls.item()
            epoch_reg_loss += loss_reg.item()
            epoch_total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({
                'cls': f"{loss_cls.item():.4f}",
                'reg': f"{loss_reg.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        avg_cls = epoch_cls_loss / num_batches
        avg_reg = epoch_reg_loss / num_batches
        avg_total = epoch_total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_total:.4f} | Cls: {avg_cls:.4f} | Reg: {avg_reg:.4f}")

        # Validation
        checkpoint_interval = cfg['training_teacher'].get('checkpoint_interval', 5)
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == epochs:
            logger.info("Running validation...")
            mAP, per_class_ap = validate(model, val_loader, anchors, device, num_classes)
            logger.info(f"Val mAP@0.5: {mAP:.4f}")

            # Save checkpoint
            ckpt_path = os.path.join(checkpoint_dir, f"teacher_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_total, 'mAP': mAP,
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

            if mAP > best_map:
                best_map = mAP
                best_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_total, 'mAP': mAP,
                }, best_path)
                logger.info(f"New best model! mAP: {best_map:.4f}")

    logger.info(f"Training complete! Best mAP: {best_map:.4f}")


if __name__ == '__main__':
    main()
