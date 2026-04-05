"""
Evaluation Script
=================

Proper evaluation with:
  - Anchor-based detection decoding
  - Per-class NMS
  - mAP@0.5 (PASCAL VOC 11-point interpolation)
  - Per-class AP breakdown
  - FPS benchmarking
  - Model parameter / size comparison
"""

import os
import torch
import argparse
import time
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import (
    load_config, setup_logger, generate_all_anchors,
    postprocess_detections, compute_map
)
from data.dataset import DOTADetectionDataset
from models.teacher import TeacherDetector
from models.student import StudentDetector


DOTA_CLASSES = [
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'tennis-court', 'basketball-court', 'ground-track-field',
    'harbor', 'bridge', 'large-vehicle', 'small-vehicle',
    'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool'
]


def get_anchors_for_model(model, image_size, device):
    """Run a dummy forward pass to determine FPN sizes and generate matching anchors."""
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, image_size, image_size, device=device)
        out = model(dummy)

        # Detect if model has adapted_features (student) or just fpn_features (teacher)
        if 'adapted_features' in out:
            feat_sizes = [f.shape[-1] for f in out['adapted_features']]
        else:
            feat_sizes = [f.shape[-1] for f in out['fpn_features']]

        total_anchors = out['cls_logits'].shape[1]

    strides = [image_size // s for s in feat_sizes]
    anchors = generate_all_anchors(image_size, strides, base_sizes=[s * 4 for s in strides])
    assert anchors.shape[0] == total_anchors, \
        f"Anchor mismatch: {anchors.shape[0]} vs {total_anchors}"
    return anchors


def evaluate_model(model, data_loader, anchors, device, num_classes=15, image_size=128):
    """
    Evaluate a model with proper detection decoding and mAP computation.

    Returns:
        mAP, per_class_ap, precision, recall, f1
    """
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = torch.stack(images).to(device)
            targets_dev = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in t.items()} for t in targets]

            outputs = model(images)

            preds = postprocess_detections(
                outputs['cls_logits'], outputs['bbox_regs'],
                anchors.to(device), image_size=image_size,
                num_classes=num_classes,
                conf_threshold=0.05, nms_threshold=0.5,
                max_detections=100,
                use_background=False  # sigmoid focal loss, no background class
            )

            all_predictions.extend(preds)
            all_targets.extend(targets_dev)

    # Compute mAP
    mAP, per_class_ap = compute_map(
        all_predictions, all_targets,
        num_classes=num_classes, iou_threshold=0.5
    )

    # Compute overall precision, recall, F1
    total_tp = 0
    total_fp = 0
    total_num_gt = 0

    for preds, gts in zip(all_predictions, all_targets):
        pred_boxes = preds['boxes'].cpu().numpy()
        pred_labels = preds['labels'].cpu().numpy()
        gt_boxes = gts['boxes'].cpu().numpy()
        gt_labels = gts['labels'].cpu().numpy()

        total_num_gt += len(gt_boxes)

        if len(pred_boxes) == 0:
            continue
        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue

        # Simple matching for aggregate precision/recall
        from utils import _compute_iou_np
        iou_matrix = _compute_iou_np(pred_boxes, gt_boxes)
        matched_gt = set()
        for i in range(len(pred_boxes)):
            if len(gt_boxes) == 0:
                total_fp += 1
                continue
            best_j = np.argmax(iou_matrix[i])
            if iou_matrix[i, best_j] >= 0.5 and best_j not in matched_gt and pred_labels[i] == gt_labels[best_j]:
                total_tp += 1
                matched_gt.add(best_j)
            else:
                total_fp += 1

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_num_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return mAP, per_class_ap, precision, recall, f1


def benchmark_fps(model, device, image_size, num_iters=100):
    """Benchmark inference speed."""
    model.eval()

    with torch.no_grad():
        dummy = torch.randn(1, 3, image_size, image_size, device=device)
        for _ in range(10):
            _ = model(dummy)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in range(num_iters):
            dummy = torch.randn(1, 3, image_size, image_size, device=device)
            _ = model(dummy)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start

    return num_iters / elapsed


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--subset', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("Evaluate")
    logger.info("Starting Model Evaluation")

    device = torch.device(args.device)
    num_classes = cfg['dataset']['num_classes']

    # =========================================================================
    # LOAD TEST DATA (use val split, LR resolution for fair comparison)
    # =========================================================================
    subset_size = args.subset
    test_dataset = DOTADetectionDataset(
        data_root=cfg['dataset']['processed_data_path'],
        split='val', image_size=128,
        subset_size=subset_size, augmentation=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )
    logger.info(f"Loaded {len(test_dataset)} test samples")

    # =========================================================================
    # EVALUATE EACH MODEL
    # =========================================================================
    models_to_eval = [
        ('Teacher (1024)', TeacherDetector, 1024, './checkpoints/teacher/best_model.pth'),
        ('Student Baseline (128)', StudentDetector, 128, './checkpoints/student_baseline/best_model.pth'),
        ('Student KD (128)', StudentDetector, 128, './checkpoints/student_kd/best_model.pth'),
    ]

    results = []

    for model_name, model_class, img_size, ckpt_path in models_to_eval:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"{'='*60}")

        model = model_class(num_classes=num_classes, pretrained=False)

        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint not found: {ckpt_path} — skipping")
            continue

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)

        # Generate anchors for this model
        # For the teacher, we need to evaluate at the EVAL resolution (128 for fair comparison, 
        # or at its native 1024 with HR data). Let's evaluate each at its native resolution.
        # But the test_loader loads 128×128 images. For the teacher, we'd need HR test data.
        # For a fair comparison, let's evaluate all on 128×128 (LR) data.
        # The teacher processes 128→1024 via interpolation internally? No, we pass the raw image.
        # So for teacher eval on LR data, it receives 128×128 input (not ideal but fair comparison).

        eval_size = img_size
        if model_name.startswith('Teacher'):
            # For teacher, we need HR test data
            hr_test_dataset = DOTADetectionDataset(
                data_root=cfg['dataset']['processed_data_path'],
                split='val', image_size=img_size,
                subset_size=subset_size, augmentation=False
            )
            hr_test_loader = DataLoader(
                hr_test_dataset, batch_size=1, shuffle=False,
                num_workers=2, collate_fn=collate_fn
            )
            eval_loader = hr_test_loader
        else:
            eval_loader = test_loader
            eval_size = 128

        anchors = get_anchors_for_model(model, eval_size, device)

        # mAP evaluation
        mAP, per_class_ap, precision, recall, f1 = evaluate_model(
            model, eval_loader, anchors, device,
            num_classes=num_classes, image_size=eval_size
        )

        logger.info(f"  mAP@0.5:    {mAP:.4f}")
        logger.info(f"  Precision:  {precision:.4f}")
        logger.info(f"  Recall:     {recall:.4f}")
        logger.info(f"  F1:         {f1:.4f}")

        # Per-class AP
        logger.info("  Per-class AP:")
        for cls_id, ap in sorted(per_class_ap.items()):
            logger.info(f"    {DOTA_CLASSES[cls_id]:25s}: {ap:.4f}")

        # FPS
        fps = benchmark_fps(model, device, eval_size, num_iters=50)
        logger.info(f"  FPS:        {fps:.1f}")

        # Parameters & Size
        num_params = count_parameters(model)
        model_size = get_model_size_mb(ckpt_path)
        logger.info(f"  Params:     {num_params / 1e6:.2f}M")
        logger.info(f"  Size:       {model_size:.1f}MB")

        results.append({
            'Model': model_name, 'mAP@0.5': mAP, 'F1': f1,
            'FPS': fps, 'Params (M)': num_params / 1e6, 'Size (MB)': model_size,
        })

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"{'Model':<25} {'mAP@0.5':<10} {'F1':<10} {'FPS':<10} {'Params':<10} {'Size':<10}")
    logger.info("-" * 80)
    for r in results:
        logger.info(
            f"{r['Model']:<25} {r['mAP@0.5']:<10.4f} {r['F1']:<10.4f} "
            f"{r['FPS']:<10.1f} {r['Params (M)']:<10.2f} {r['Size (MB)']:<10.1f}"
        )
    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
