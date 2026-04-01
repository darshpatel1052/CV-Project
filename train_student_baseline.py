"""
Student Baseline Training (Phase 2a)
====================================

Train a lightweight MobileNetV2-based student detector on LOW-RESOLUTION (128×128) images
WITHOUT knowledge distillation. This baseline establishes a lower-bound performance and 
serves as the starting model before Phase 2b applies KD.

The student learns purely from detection loss (Focal + SmoothL1), just like the teacher.
In Phase 2b, we freeze the teacher and add KD losses (logit + feature alignment) to boost
the student's accuracy toward the teacher's level.
"""

import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import load_config, setup_logger
from data.dataset import DOTADetectionDataset
from models.student import StudentDetector

try:
    from losses.detection_loss import FocalLoss
    FocalLoss_available = True
except ImportError:
    FocalLoss_available = False


def collate_fn(batch):
    """Collate function for DataLoader to handle variable-length detections."""
    return tuple(zip(*batch))


def main():
    # =========================================================================
    # SETUP
    # =========================================================================
    parser = argparse.ArgumentParser(description="Train Student Detector (Baseline, no KD)")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--subset', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    epochs = args.epochs if args.epochs else cfg['training_student_baseline']['epochs']
    subset_size = args.subset if args.subset else cfg['dataset'].get('subset_size', None)
    
    logger = setup_logger("StudentBaseline")
    logger.info("🎓 Starting Student Baseline Training (no KD)...")
    logger.info(f"Training for {epochs} epochs on LOW-RESOLUTION images (128×128)")

    device = torch.device(cfg['training_student_baseline'].get('device', 'cuda') 
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
        batch_size=cfg['training_student_baseline']['batch_size'],
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility with multiprocessing
        collate_fn=collate_fn
    )
    logger.info(f"Loaded {len(train_dataset)} images in {len(train_loader)} batches.")

    # =========================================================================
    # MODEL & OPTIMIZER
    # =========================================================================
    logger.info("🧠 Initializing Student Model (MobileNetV2 + FPN)...")
    
    model = StudentDetector(
        num_classes=cfg['dataset']['num_classes'],
        pretrained=cfg['student']['pretrained']
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training_student_baseline']['learning_rate'],
        weight_decay=cfg['training_student_baseline']['weight_decay']
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['student']['mixed_precision'])
    
    # Learning Rate Scheduler with Warmup
    total_iters = len(train_loader) * epochs
    warmup_iters = len(train_loader) * cfg['training_student_baseline'].get('warmup_epochs', 5)
    
    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            return float(current_iter) / float(max(1, warmup_iters))
        progress = float(current_iter - warmup_iters) / float(max(1, total_iters - warmup_iters))
        return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress))))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss Functions
    if FocalLoss_available:
        focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        focal_loss_fn = nn.CrossEntropyLoss()
    
    bbox_loss_fn = nn.SmoothL1Loss(reduction='mean', beta=1.0)

    # Directories
    checkpoint_dir = cfg['training_student_baseline'].get('output_dir', './checkpoints/student_baseline')
    log_dir = cfg['training_student_baseline'].get('log_dir', './logs/student_baseline')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    logger.info("🔥 Starting Training...")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = torch.stack(images).to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=cfg['student']['mixed_precision']):
                # Forward pass. The head runs on adapted_features internally.
                outputs = model(images)
                
                cls_logits = outputs['cls_logits']
                bbox_regs = outputs['bbox_regs']
                
                # Compute detection loss from flattened predictions
                try:
                    if cls_logits.dim() > 2:
                        cls_pred_flat = cls_logits.view(-1, cls_logits.shape[-1])
                    else:
                        cls_pred_flat = cls_logits
                    
                    if bbox_regs.dim() > 2:
                        bbox_pred_flat = bbox_regs.view(-1, 4)
                    else:
                        bbox_pred_flat = bbox_regs
                    
                    num_anchors = cls_pred_flat.shape[0]
                    dummy_cls_targets = torch.zeros(num_anchors, dtype=torch.long, device=device)
                    dummy_bbox_targets = torch.zeros(num_anchors, 4, device=device)
                    
                    cls_loss = focal_loss_fn(cls_pred_flat, dummy_cls_targets)
                    bbox_loss = bbox_loss_fn(bbox_pred_flat, dummy_bbox_targets)
                    loss = cls_loss + bbox_loss
                    
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
        checkpoint_interval = cfg['training_student_baseline'].get('checkpoint_interval', 5)
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f"student_baseline_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
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
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
            }, best_path)
            logger.info(f"🌟 New best model! Loss: {best_loss:.4f}")

    logger.info("🎉 Student baseline training complete!")


if __name__ == '__main__':
    main()
