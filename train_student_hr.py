"""
Student on HR Training (Phase 2 — Additional Baseline)
=======================================================

Train the SAME lightweight MobileNetV2 student architecture on
HIGH-RESOLUTION (1024×1024) images WITHOUT knowledge distillation.

Purpose: Isolate the RESOLUTION EFFECT from the DISTILLATION EFFECT.
- If this outperforms Student Baseline (128px), resolution matters.
- If KD Student (128px) approaches this, KD successfully recovers
  resolution information without needing HR images at inference time.

This is a critical baseline that most KD papers omit.
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
    return tuple(zip(*batch))


def main():
    parser = argparse.ArgumentParser(description="Train Student on HR images (no KD, 1024×1024)")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--subset', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    hr_cfg = cfg.get('training_student_hr', cfg['training_student_baseline'])
    epochs = args.epochs if args.epochs else hr_cfg.get('epochs', 50)
    subset_size = args.subset if args.subset else cfg['dataset'].get('subset_size', None)

    logger = setup_logger("StudentHR")
    logger.info("=" * 60)
    logger.info("Student on HR Training (Resolution Isolation Baseline)")
    logger.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    torch.manual_seed(cfg['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg['seed'])

    num_classes = cfg['dataset']['num_classes']
    image_size = cfg['dataset']['teacher_resolution']  # 1024 — HR!

    # =========================================================================
    # DATA — HR images for student (key difference from baseline)
    # =========================================================================
    logger.info(f"Loading HR DOTA at {image_size}×{image_size}...")
    train_dataset = DOTADetectionDataset(
        data_root=cfg['dataset']['processed_data_path'],
        split='train', image_size=image_size,
        subset_size=subset_size, augmentation=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=hr_cfg.get('batch_size', 2),
        shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True
    )

    val_subset = min(200, subset_size) if subset_size else 200
    val_dataset = DOTADetectionDataset(
        data_root=cfg['dataset']['processed_data_path'],
        split='val', image_size=image_size,
        subset_size=val_subset, augmentation=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1,
        shuffle=False, num_workers=2, collate_fn=collate_fn, pin_memory=True
    )
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # =========================================================================
    # MODEL — Same student architecture, but processing HR inputs
    # =========================================================================
    model = StudentDetector(num_classes=num_classes, pretrained=cfg['student']['pretrained'])
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Student params: {total_params / 1e6:.2f}M (same as LR baseline)")

    # =========================================================================
    # ANCHORS — Computed for student at HR resolution
    # =========================================================================
    with torch.no_grad():
        dummy = torch.randn(1, 3, image_size, image_size, device=device)
        dummy_out = model(dummy)
        adapted_sizes = [f.shape[-1] for f in dummy_out['adapted_features']]
        total_anchors_model = dummy_out['cls_logits'].shape[1]

    fpn_strides = [image_size // s for s in adapted_sizes]
    anchors = generate_all_anchors(image_size, fpn_strides,
                                    base_sizes=[s * 4 for s in fpn_strides])
    assert anchors.shape[0] == total_anchors_model
    logger.info(f"Anchors: {anchors.shape[0]}, FPN strides: {fpn_strides}")

    # =========================================================================
    # OPTIMIZER & LOSS
    # =========================================================================
    lr = hr_cfg.get('learning_rate', 0.0005)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=hr_cfg.get('weight_decay', 0.0001)
    )

    use_amp = cfg['student'].get('mixed_precision', True) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    total_iters = len(train_loader) * epochs
    warmup_iters = len(train_loader) * hr_cfg.get('warmup_epochs', 5)

    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            return float(current_iter) / float(max(1, warmup_iters))
        import math
        progress = float(current_iter - warmup_iters) / float(max(1, total_iters - warmup_iters))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    bbox_loss_fn = SmoothL1Loss(beta=1.0 / 9.0)

    checkpoint_dir = hr_cfg.get('output_dir', './checkpoints/student_hr')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    logger.info(f"Training {epochs} epochs with HR images...")
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
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        checkpoint_interval = hr_cfg.get('checkpoint_interval', 5)
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == epochs:
            logger.info("Validating...")
            mAP, _ = validate(model, val_loader, anchors, device, num_classes)
            logger.info(f"Val mAP@0.5: {mAP:.4f}")

            ckpt_path = os.path.join(checkpoint_dir, f"student_hr_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss, 'mAP': mAP,
                'image_size': image_size,
            }, ckpt_path)

            if mAP > best_map:
                best_map = mAP
                best_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'loss': avg_loss, 'mAP': mAP,
                    'image_size': image_size,
                }, best_path)
                logger.info(f"New best! mAP: {best_map:.4f}")

    logger.info(f"Student HR training complete! Best mAP: {best_map:.4f}")


if __name__ == '__main__':
    main()
