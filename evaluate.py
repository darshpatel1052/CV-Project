"""
Comprehensive Evaluation Script — Enhanced
============================================

Evaluates all trained models with:
  - AABB-based mAP@0.5 (primary metric)
  - Optional OBB-based mAP@0.5 (for comparison with published results)
  - Per-class AP breakdown
  - Precision, Recall, F1
  - FPS benchmarking
  - Model parameter / size comparison
  - Side-by-side comparison table

Models evaluated:
  1. Teacher (Swin-T, 1024×1024) — upper bound
  2. Student Baseline (MobileNetV2, 128×128) — lower bound
  3. Student HR (MobileNetV2, 1024×1024) — resolution isolation
  4. Student KD (MobileNetV2, 128×128 + KD) — main result
  5. Teacher on Upsampled LR — naive baseline (evaluated separately)

Usage:
    python evaluate.py                    # Evaluate all available models
    python evaluate.py --subset 200       # Quick evaluation on subset
    python evaluate.py --obb              # Include OBB-aware evaluation
"""

import os
import torch
import argparse
import time
import json
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from utils import (
    load_config, setup_logger, generate_all_anchors,
    postprocess_detections, compute_map, _compute_iou_np
)
from data.dataset import DOTADetectionDataset
from models.teacher import TeacherDetector
from models.student import StudentDetector


DOTA_CLASSES = [
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'tennis-court', 'basketball-court', 'ground-track-field',
    'harbor', 'bridge', 'large-vehicle', 'small-vehicle',
    'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool',
    'container-crane'
]


def get_anchors_for_model(model, image_size, device):
    """Run a dummy forward pass to determine FPN sizes and generate matching anchors."""
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, image_size, image_size, device=device)
        out = model(dummy)

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
    """Evaluate with proper detection decoding and mAP computation."""
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
                max_detections=100, use_background=False
            )

            all_predictions.extend(preds)
            all_targets.extend(targets_dev)

    # mAP
    mAP, per_class_ap = compute_map(
        all_predictions, all_targets,
        num_classes=num_classes, iou_threshold=0.5
    )

    # Precision / Recall / F1
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
            if iou_matrix[i, best_j] >= 0.5 and best_j not in matched \
               and pred_labels[i] == gt_labels[best_j]:
                total_tp += 1
                matched.add(best_j)
            else:
                total_fp += 1

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_gt, 1)
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
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0


def load_model_checkpoint(model, ckpt_path, device, logger):
    """Load checkpoint with key filtering for robustness."""
    if not os.path.exists(ckpt_path):
        logger.warning(f"Checkpoint not found: {ckpt_path}")
        return False

    ckpt = torch.load(ckpt_path, map_location=device)
    model_dict = model.state_dict()
    filtered_dict = {
        k: v for k, v in ckpt['model_state_dict'].items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    matched = len(filtered_dict)
    total = len(ckpt['model_state_dict'])
    logger.info(f"Loaded {matched}/{total} keys from {ckpt_path}")

    if 'mAP' in ckpt:
        logger.info(f"  Training mAP: {ckpt['mAP']:.4f} (epoch {ckpt.get('epoch', '?')})")

    return True


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    parser = argparse.ArgumentParser(description="Evaluate all trained models")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--subset', type=int, default=None)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--obb', action='store_true',
                        help='Include OBB-aware evaluation')
    parser.add_argument('--save_results', action='store_true', default=True,
                        help='Save evaluation results to JSON')
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("Evaluate")
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE MODEL EVALUATION")
    logger.info("=" * 80)

    device = torch.device(args.device)
    num_classes = cfg['dataset']['num_classes']

    # =========================================================================
    # DEFINE ALL MODELS TO EVALUATE
    # =========================================================================
    models_to_eval = [
        {
            'name': 'Teacher (Swin-T, 1024)',
            'class': TeacherDetector,
            'image_size': 1024,
            'checkpoint': './checkpoints/teacher/best_model.pth',
            'description': 'Upper bound: high-capacity model on HR images',
        },
        {
            'name': 'Student Baseline (128)',
            'class': StudentDetector,
            'image_size': 128,
            'checkpoint': './checkpoints/student_baseline/best_model.pth',
            'description': 'Lower bound: student on LR without KD',
        },
        {
            'name': 'Student HR (1024)',
            'class': StudentDetector,
            'image_size': 1024,
            'checkpoint': './checkpoints/student_hr/best_model.pth',
            'description': 'Resolution isolation: student architecture on HR',
        },
        {
            'name': 'Student KD (128)',
            'class': StudentDetector,
            'image_size': 128,
            'checkpoint': './checkpoints/student_kd/best_model.pth',
            'description': 'Main result: student with KD on LR',
        },
    ]

    # Also check for experiment checkpoints
    exp_dir = './checkpoints/experiments'
    if os.path.exists(exp_dir):
        for exp_name in sorted(os.listdir(exp_dir)):
            best_path = os.path.join(exp_dir, exp_name, 'best_model.pth')
            if os.path.exists(best_path):
                models_to_eval.append({
                    'name': f'Exp: {exp_name}',
                    'class': StudentDetector,
                    'image_size': 128,
                    'checkpoint': best_path,
                    'description': f'Ablation experiment: {exp_name}',
                })

    results = []
    skipped = []

    for model_info in models_to_eval:
        model_name = model_info['name']
        ckpt_path = model_info['checkpoint']
        img_size = model_info['image_size']

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"  {model_info['description']}")
        logger.info(f"{'='*60}")

        if not os.path.exists(ckpt_path):
            logger.warning(f"  ⚠ Checkpoint not found: {ckpt_path} — skipping")
            skipped.append({'name': model_name, 'checkpoint': ckpt_path, 'reason': 'checkpoint_not_found'})
            continue

        # Initialize model
        model_class = model_info['class']
        model = model_class(num_classes=num_classes, pretrained=False)

        if not load_model_checkpoint(model, ckpt_path, device, logger):
            continue

        model = model.to(device)

        # Load test data at appropriate resolution
        subset_size = args.subset
        test_dataset = DOTADetectionDataset(
            data_root=cfg['dataset']['processed_data_path'],
            split='val', image_size=img_size,
            subset_size=subset_size, augmentation=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1 if img_size >= 512 else 4,
            shuffle=False, num_workers=2, collate_fn=collate_fn
        )

        # Generate anchors
        anchors = get_anchors_for_model(model, img_size, device)

        # Run evaluation
        mAP, per_class_ap, precision, recall, f1 = evaluate_model(
            model, test_loader, anchors, device,
            num_classes=num_classes, image_size=img_size
        )

        # FPS benchmark
        fps = benchmark_fps(model, device, img_size, num_iters=50)

        # Model stats
        total_params, trainable_params = count_parameters(model)
        model_size = get_model_size_mb(ckpt_path)

        # Log results
        logger.info(f"\n  AABB-mAP@0.5:  {mAP:.4f}")
        logger.info(f"  Precision:     {precision:.4f}")
        logger.info(f"  Recall:        {recall:.4f}")
        logger.info(f"  F1:            {f1:.4f}")
        logger.info(f"  FPS:           {fps:.1f}")
        logger.info(f"  Parameters:    {total_params / 1e6:.2f}M ({trainable_params / 1e6:.2f}M trainable)")
        logger.info(f"  Model Size:    {model_size:.1f}MB")

        logger.info("\n  Per-class AP:")
        for cls_id in range(num_classes):
            ap = per_class_ap.get(cls_id, 0.0)
            logger.info(f"    {DOTA_CLASSES[cls_id]:25s}: {ap:.4f}")

        result = {
            'name': model_name,
            'image_size': img_size,
            'mAP': mAP,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fps': fps,
            'params_M': total_params / 1e6,
            'size_MB': model_size,
            'per_class_ap': {DOTA_CLASSES[k]: v for k, v in per_class_ap.items()},
            'checkpoint': ckpt_path,
        }
        results.append(result)

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    if results:
        logger.info(f"\n{'='*90}")
        logger.info("EVALUATION SUMMARY")
        logger.info(f"{'='*90}")
        logger.info(f"{'Model':<30} {'AABB-mAP':<10} {'F1':<8} {'FPS':<8} {'Params':<10} {'Size':<8}")
        logger.info("-" * 90)
        for r in results:
            logger.info(
                f"{r['name']:<30} {r['mAP']:<10.4f} {r['f1']:<8.4f} "
                f"{r['fps']:<8.1f} {r['params_M']:<10.2f}M {r['size_MB']:<8.1f}MB"
            )

        # Key comparisons
        teacher = next((r for r in results if 'Teacher' in r['name']), None)
        baseline = next((r for r in results if 'Baseline' in r['name']), None)
        kd = next((r for r in results if 'KD' in r['name']), None)
        student_hr = next((r for r in results if 'HR' in r['name'] and 'Student' in r['name']), None)

        logger.info(f"\n{'='*50}")
        logger.info("KEY COMPARISONS")
        logger.info(f"{'='*50}")

        if baseline and kd:
            gain = kd['mAP'] - baseline['mAP']
            logger.info(f"KD gain over baseline:     +{gain:.4f} mAP")
            logger.info(f"  → KD {'improves' if gain > 0 else 'does NOT improve'} over baseline")

        if teacher and kd:
            gap = teacher['mAP'] - kd['mAP']
            logger.info(f"Teacher-Student gap:       {gap:.4f} mAP")

        if student_hr and kd:
            gap = student_hr['mAP'] - kd['mAP']
            logger.info(f"HR-KD gap:                 {gap:.4f} mAP")
            if gap < 0.05:
                logger.info("  → KD nearly matches HR student! Adapter approach works.")

        if teacher and kd:
            speedup = kd['fps'] / max(teacher['fps'], 1)
            compression = teacher['params_M'] / max(kd['params_M'], 0.1)
            logger.info(f"Speed improvement:         {speedup:.1f}× faster")
            logger.info(f"Model compression:         {compression:.1f}× smaller")

    # =========================================================================
    # SAVE RESULTS  (always write, even if some/all models were skipped)
    # =========================================================================
    if args.save_results:
        results_dir = cfg.get('evaluation', {}).get('output_dir', './outputs/metrics')
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = os.path.join(results_dir, f'evaluation_{timestamp}.json')

        output = {
            'timestamp': timestamp,
            'models_evaluated': len(results),
            'models_skipped': len(skipped),
            'results': results,
            'skipped': skipped,
        }

        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"\nResults saved to: {results_path}")

        if not results:
            logger.warning(
                "⚠  No models were successfully evaluated. "
                "Check that checkpoints exist in ./checkpoints/. "
                f"Skipped: {[s['name'] for s in skipped]}"
            )

    logger.info("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
