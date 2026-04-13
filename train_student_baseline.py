"""
Student Baseline Training (Phase 2a)
====================================

Train a lightweight MobileNetV2-based student detector on LR (128×128) images
WITHOUT knowledge distillation. Uses proper anchor matching and real detection loss.
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
from models.student import StudentDetector
from losses.detection_loss import FocalLoss, SmoothL1Loss
from train_teacher import compute_detection_loss, validate


def collate_fn(batch):
    """Collate function for variable-length detections."""
    return tuple(zip(*batch))


def main():
    parser = argparse.ArgumentParser(description="Train Student Detector (Baseline, no KD)")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--subset', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    epochs = args.epochs if args.epochs else cfg['training_student_baseline']['epochs']
    subset_size = args.subset if args.subset else cfg['dataset'].get('subset_size', None)

    logger = setup_logger("StudentBaseline")
    logger.info("Starting Student Baseline Training (no KD)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    torch.manual_seed(cfg['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg['seed'])

    num_classes = cfg['dataset']['num_classes']
    image_size = cfg['dataset']['student_resolution']  # 128

    # =========================================================================
    # DATA
    # =========================================================================
    train_dataset = DOTADetectionDataset(
        data_root=cfg['dataset']['processed_data_path'],
        split='train', image_size=image_size,
        subset_size=subset_size, augmentation=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['training_student_baseline']['batch_size'],
        shuffle=True, num_workers=4, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=True
    )

    val_subset = min(200, subset_size) if subset_size else 200
    val_dataset = DOTADetectionDataset(
        data_root=cfg['dataset']['processed_data_path'],
        split='val', image_size=image_size,
        subset_size=val_subset, augmentation=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=max(1, cfg['training_student_baseline']['batch_size'] // 2),
        shuffle=False, num_workers=4, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=True
    )
    logger.info(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    # =========================================================================
    # MODEL & ANCHORS
    # =========================================================================
    model = StudentDetector(num_classes=num_classes, pretrained=cfg['student']['pretrained'])
    model = model.to(device)
    # torch.compile() fuses operations and reduces Python overhead (~15-30% speedup).
    # The first epoch is slower due to one-time compilation; subsequent epochs are faster.
    if device.type == 'cuda':
        model = torch.compile(model)

    # The student's detection head runs on adapted_features (upsampled 4×).
    # So the effective spatial sizes are same as if the input were 4× larger.
    # For 128 input: adapted P3..P6 sizes are [128, 64, 32, 16] (same shape as teacher on 1024).
    # But anchors need to be relative to the ORIGINAL 128×128 image, not the adapted feature space.
    # Let's determine the actual FPN strides by running a dummy forward pass.

    with torch.no_grad():
        dummy = torch.randn(1, 3, image_size, image_size, device=device)
        dummy_out = model(dummy)
        # The student head runs on adapted_features, which are upsampled versions.
        # But cls_logits is produced from these adapted features.
        # adapted_features sizes correspond to teacher's FPN sizes (for 1024 input).
        # The number of anchors from adapted features = same as teacher.
        # But our image is only 128×128, so anchors at teacher scale don't make sense.
        # The RIGHT approach: generate anchors matching the ADAPTED feature map sizes,
        # but scale them to the 128×128 image space.
        adapted_sizes = [f.shape[-1] for f in dummy_out['adapted_features']]
        total_anchors_model = dummy_out['cls_logits'].shape[1]
        num_cls_output = dummy_out['cls_logits'].shape[2]
        logger.info(f"Adapted feature map sizes: {adapted_sizes}")
        logger.info(f"Total anchors: {total_anchors_model}, Cls channels: {num_cls_output}")

    # Compute strides relative to the student's image size
    # adapted_sizes come from upsampling the native FPN features 4×.
    # For the student: native FPN produces P3..P6 at sizes [16,8,4,2] for 128 input.
    # After 4× upsampling: [64,32,16,8]. But the anchors should tile the 128×128 image.
    # So stride = 128 / adapted_size for each level.
    fpn_strides = [image_size // s for s in adapted_sizes]
    logger.info(f"Student FPN strides (on adapted features): {fpn_strides}")

    anchors = generate_all_anchors(image_size, fpn_strides, base_sizes=[s * 4 for s in fpn_strides])
    logger.info(f"Generated {anchors.shape[0]} anchors (model expects {total_anchors_model})")
    assert anchors.shape[0] == total_anchors_model, \
        f"Anchor count mismatch: {anchors.shape[0]} vs {total_anchors_model}"

    # =========================================================================
    # OPTIMIZER & LOSS
    # =========================================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training_student_baseline']['learning_rate'],
        weight_decay=cfg['training_student_baseline']['weight_decay']
    )

    use_amp = cfg['student'].get('mixed_precision', True) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    total_iters = len(train_loader) * epochs
    warmup_iters = len(train_loader) * cfg['training_student_baseline'].get('warmup_epochs', 5)

    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            return float(current_iter) / float(max(1, warmup_iters))
        import math
        progress = float(current_iter - warmup_iters) / float(max(1, total_iters - warmup_iters))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    bbox_loss_fn = SmoothL1Loss(beta=1.0 / 9.0)

    checkpoint_dir = cfg['training_student_baseline'].get('output_dir', './checkpoints/student_baseline')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    logger.info(f"Starting training for {epochs} epochs...")
    best_map = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, targets in pbar:
            images = torch.stack(images).to(device)
            targets_list = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images)
                loss_cls, loss_reg = compute_detection_loss(
                    outputs['cls_logits'], outputs['bbox_regs'],
                    anchors, targets_list, device,
                    focal_loss_fn, bbox_loss_fn, num_classes
                )
                loss = loss_cls + loss_reg

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({
                'cls': f"{loss_cls.item():.4f}",
                'reg': f"{loss_reg.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

        # Validation and checkpointing
        checkpoint_interval = cfg['training_student_baseline'].get('checkpoint_interval', 5)
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == epochs:
            logger.info("Running validation...")
            mAP, _ = validate(model, val_loader, anchors, device, num_classes)
            logger.info(f"Val mAP@0.5: {mAP:.4f}")

            ckpt_path = os.path.join(checkpoint_dir, f"student_baseline_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss, 'mAP': mAP,
            }, ckpt_path)

            if mAP > best_map:
                best_map = mAP
                best_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'loss': avg_loss, 'mAP': mAP,
                }, best_path)
                logger.info(f"New best model! mAP: {best_map:.4f}")

    logger.info(f"Student baseline training complete! Best mAP: {best_map:.4f}")


if __name__ == '__main__':
    main()
