"""
Teacher on Upsampled LR (Naive Upsampling Baseline)
=====================================================

Evaluates the TRAINED teacher model on naively upsampled LR images.
Takes 128×128 images, bilinearly upsamples to 1024×1024, feeds to teacher.

Purpose: Establish whether simple upsampling + powerful model beats
         resolution-agnostic KD. If KD student outperforms this,
         the adapter approach adds value beyond naive interpolation.

This does NOT train anything — it evaluates the existing teacher checkpoint.
"""

import os
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import (
    load_config, setup_logger, generate_all_anchors,
    postprocess_detections, compute_map
)
from data.dataset import DOTADetectionDataset
from models.teacher import TeacherDetector


DOTA_CLASSES = [
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'tennis-court', 'basketball-court', 'ground-track-field',
    'harbor', 'bridge', 'large-vehicle', 'small-vehicle',
    'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool',
    'container-crane'
]


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    parser = argparse.ArgumentParser(description="Evaluate teacher on upsampled LR images")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--subset', type=int, default=None)
    parser.add_argument('--upsample_mode', type=str, default='bilinear',
                        choices=['bilinear', 'bicubic', 'nearest'],
                        help='Upsampling interpolation method')
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("TeacherUpsampled")
    logger.info("=" * 60)
    logger.info("Teacher on Upsampled LR — Naive Baseline Evaluation")
    logger.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = cfg['dataset']['num_classes']
    teacher_resolution = cfg['dataset']['teacher_resolution']  # 1024
    student_resolution = cfg['dataset']['student_resolution']  # 128

    # =========================================================================
    # DATA — Load LR images (128×128) — same as student sees
    # =========================================================================
    subset_size = args.subset or cfg['dataset'].get('subset_size', None)
    eval_dataset = DOTADetectionDataset(
        data_root=cfg['dataset']['processed_data_path'],
        split='val', image_size=student_resolution,
        subset_size=subset_size, augmentation=False
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )
    logger.info(f"Loaded {len(eval_dataset)} LR evaluation images")

    # =========================================================================
    # MODEL — Load trained teacher
    # =========================================================================
    teacher = TeacherDetector(num_classes=num_classes, pretrained=False)

    teacher_ckpt = cfg['training_student_kd'].get('teacher_checkpoint',
                                                    './checkpoints/teacher/best_model.pth')
    if not os.path.exists(teacher_ckpt):
        logger.error(f"Teacher checkpoint not found: {teacher_ckpt}")
        logger.error("Train the teacher first with: python train_teacher.py")
        return

    ckpt = torch.load(teacher_ckpt, map_location=device)
    model_dict = teacher.state_dict()
    filtered_dict = {
        k: v for k, v in ckpt['model_state_dict'].items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model_dict.update(filtered_dict)
    teacher.load_state_dict(model_dict)
    logger.info(f"Loaded teacher from {teacher_ckpt}")

    teacher = teacher.to(device)
    teacher.eval()

    # =========================================================================
    # ANCHORS — For teacher at teacher resolution (1024)
    # =========================================================================
    with torch.no_grad():
        dummy = torch.randn(1, 3, teacher_resolution, teacher_resolution, device=device)
        dummy_out = teacher(dummy)
        fpn_sizes = [f.shape[-1] for f in dummy_out['fpn_features']]
        total_anchors = dummy_out['cls_logits'].shape[1]

    fpn_strides = [teacher_resolution // s for s in fpn_sizes]
    anchors = generate_all_anchors(teacher_resolution, fpn_strides,
                                    base_sizes=[s * 4 for s in fpn_strides])
    assert anchors.shape[0] == total_anchors
    logger.info(f"Teacher anchors: {anchors.shape[0]}")

    # =========================================================================
    # EVALUATE — Upsample LR → HR, run teacher, scale predictions back
    # =========================================================================
    logger.info(f"Evaluating with {args.upsample_mode} upsampling: "
                f"{student_resolution}→{teacher_resolution}...")

    all_predictions = []
    all_targets = []
    scale_factor = student_resolution / teacher_resolution  # 128/1024 = 0.125

    with torch.no_grad():
        for images, targets in tqdm(eval_loader, desc="Evaluating"):
            images_lr = torch.stack(images).to(device)

            # Upsample LR to HR
            images_hr = F.interpolate(
                images_lr,
                size=(teacher_resolution, teacher_resolution),
                mode=args.upsample_mode,
                align_corners=False if args.upsample_mode != 'nearest' else None
            )

            # Teacher inference at HR resolution
            outputs = teacher(images_hr)

            # Post-process detections (at teacher resolution)
            preds = postprocess_detections(
                outputs['cls_logits'], outputs['bbox_regs'],
                anchors.to(device), image_size=teacher_resolution,
                num_classes=num_classes,
                conf_threshold=0.05, nms_threshold=0.5,
                max_detections=100, use_background=False
            )

            # Scale prediction boxes back to LR resolution for fair evaluation
            for pred in preds:
                pred['boxes'] = pred['boxes'] * scale_factor

            all_predictions.extend(preds)

            # Targets are already at LR resolution (128×128)
            targets_dev = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in t.items()} for t in targets]
            all_targets.extend(targets_dev)

    # =========================================================================
    # COMPUTE METRICS
    # =========================================================================
    mAP, per_class_ap = compute_map(
        all_predictions, all_targets,
        num_classes=num_classes, iou_threshold=0.5
    )

    # Overall precision/recall
    from utils import _compute_iou_np
    total_tp, total_fp, total_gt = 0, 0, 0
    for preds, gts in zip(all_predictions, all_targets):
        pred_boxes = preds['boxes'].cpu().numpy()
        pred_labels = preds['labels'].cpu().numpy()
        gt_boxes = gts['boxes'].cpu().numpy()
        gt_labels = gts['labels'].cpu().numpy()

        total_gt += len(gt_boxes)
        if len(pred_boxes) == 0:
            continue
        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue

        iou_matrix = _compute_iou_np(pred_boxes, gt_boxes)
        matched = set()
        for i in range(len(pred_boxes)):
            best_j = np.argmax(iou_matrix[i])
            if iou_matrix[i, best_j] >= 0.5 and best_j not in matched and pred_labels[i] == gt_labels[best_j]:
                total_tp += 1
                matched.add(best_j)
            else:
                total_fp += 1

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    # =========================================================================
    # REPORT
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: Teacher on Upsampled LR ({args.upsample_mode})")
    logger.info(f"{'='*60}")
    logger.info(f"  Upsampling: {student_resolution}→{teacher_resolution} ({args.upsample_mode})")
    logger.info(f"  mAP@0.5:   {mAP:.4f}")
    logger.info(f"  Precision:  {precision:.4f}")
    logger.info(f"  Recall:     {recall:.4f}")
    logger.info(f"  F1:         {f1:.4f}")

    logger.info("\n  Per-class AP:")
    for cls_id, ap in sorted(per_class_ap.items()):
        logger.info(f"    {DOTA_CLASSES[cls_id]:25s}: {ap:.4f}")

    # Save results
    results_dir = './outputs/metrics'
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f'teacher_upsampled_{args.upsample_mode}.txt')
    with open(results_path, 'w') as f:
        f.write(f"upsample_mode: {args.upsample_mode}\n")
        f.write(f"mAP@0.5: {mAP:.4f}\n")
        f.write(f"precision: {precision:.4f}\n")
        f.write(f"recall: {recall:.4f}\n")
        f.write(f"f1: {f1:.4f}\n")
        for cls_id, ap in sorted(per_class_ap.items()):
            f.write(f"{DOTA_CLASSES[cls_id]}: {ap:.4f}\n")
    logger.info(f"\nResults saved to {results_path}")

    logger.info("=" * 60)
    logger.info("KEY COMPARISON:")
    logger.info("  If Student KD (128px) > this result:")
    logger.info("    → KD adapters provide value beyond naive upsampling")
    logger.info("  If this result > Student KD:")
    logger.info("    → More work needed on adapter design")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
