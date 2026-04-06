"""
Data Verification & Ground Truth Visualization
================================================

CRITICAL: Run this BEFORE any training to verify:
1. HR patches have correctly positioned bounding boxes
2. LR patches have correctly scaled bounding boxes
3. No off-by-one errors in coordinate transformation
4. OBB → AABB conversion preserves object coverage

Generates visual plots with GT boxes overlaid on patches for manual inspection.

Usage:
    python verify_data.py --num_samples 20
    python verify_data.py --num_samples 10 --resolution both
    python verify_data.py --stats  # Print dataset statistics only
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from PIL import Image
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


DOTA_CLASSES = [
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'tennis-court', 'basketball-court', 'ground-track-field',
    'harbor', 'bridge', 'large-vehicle', 'small-vehicle',
    'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool',
    'container-crane'
]

# Colors for each class (deterministic)
CLASS_COLORS = plt.cm.tab20(np.linspace(0, 1, len(DOTA_CLASSES)))


def load_annotations(label_path):
    """
    Load DOTA-format annotations from a label file.

    Returns list of dicts with 'obb', 'aabb', 'class', 'difficulty'.
    """
    objects = []
    if not os.path.exists(label_path):
        return objects

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('imagesource') or line.startswith('gsd:'):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                coords = [float(p) for p in parts[:8]]
                class_name = parts[8]
                difficulty = int(parts[9]) if len(parts) > 9 else 0

                xs = [coords[0], coords[2], coords[4], coords[6]]
                ys = [coords[1], coords[3], coords[5], coords[7]]

                objects.append({
                    'obb': coords,
                    'aabb': [min(xs), min(ys), max(xs), max(ys)],
                    'class': class_name,
                    'difficulty': difficulty
                })
            except (ValueError, IndexError):
                continue

    return objects


def draw_verification_plot(image_path, objects, save_path, title="", show_obb=True):
    """
    Draw both OBB and AABB on an image for visual verification.

    Args:
        image_path: Path to the image file
        objects: List of annotation dicts
        save_path: Where to save the visualization
        title: Plot title
        show_obb: Whether to draw oriented boxes (green) in addition to AABB (red)
    """
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)

    fig, axes = plt.subplots(1, 2 if show_obb else 1, figsize=(20, 10) if show_obb else (10, 10))
    if not show_obb:
        axes = [axes]

    # --- Left panel: AABB boxes ---
    ax = axes[0]
    ax.imshow(img_np)
    ax.set_title(f"AABB Boxes — {title}", fontsize=12)

    for obj in objects:
        x1, y1, x2, y2 = obj['aabb']
        cls_name = obj['class']
        cls_idx = DOTA_CLASSES.index(cls_name) if cls_name in DOTA_CLASSES else 0
        color = CLASS_COLORS[cls_idx]

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 3), cls_name, fontsize=7,
                color='white', bbox=dict(boxstyle='round,pad=0.2',
                facecolor=color, alpha=0.8))

    ax.set_xlim(0, img_np.shape[1])
    ax.set_ylim(img_np.shape[0], 0)
    ax.axis('off')

    # --- Right panel: OBB boxes ---
    if show_obb:
        ax = axes[1]
        ax.imshow(img_np)
        ax.set_title(f"OBB Boxes — {title}", fontsize=12)

        for obj in objects:
            obb = obj['obb']
            cls_name = obj['class']
            cls_idx = DOTA_CLASSES.index(cls_name) if cls_name in DOTA_CLASSES else 0
            color = CLASS_COLORS[cls_idx]

            # Draw oriented polygon
            polygon_pts = [(obb[i], obb[i+1]) for i in range(0, 8, 2)]
            poly = Polygon(polygon_pts, closed=True,
                          linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(poly)
            ax.text(obb[0], max(0, obb[1] - 3), cls_name, fontsize=7,
                    color='white', bbox=dict(boxstyle='round,pad=0.2',
                    facecolor=color, alpha=0.8))

        ax.set_xlim(0, img_np.shape[1])
        ax.set_ylim(img_np.shape[0], 0)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def draw_hr_lr_comparison(hr_image_path, lr_image_path, hr_objects, lr_objects,
                           save_path, patch_name):
    """
    Side-by-side HR vs LR with boxes to verify coordinate scaling.
    """
    hr_img = np.array(Image.open(hr_image_path).convert('RGB'))
    lr_img = np.array(Image.open(lr_image_path).convert('RGB'))

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # HR
    ax = axes[0]
    ax.imshow(hr_img)
    ax.set_title(f"HR (1024×1024) — {patch_name}", fontsize=11)
    for obj in hr_objects:
        x1, y1, x2, y2 = obj['aabb']
        cls_name = obj['class']
        cls_idx = DOTA_CLASSES.index(cls_name) if cls_name in DOTA_CLASSES else 0
        color = CLASS_COLORS[cls_idx]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 3), f"{cls_name}", fontsize=7,
                color='white', bbox=dict(facecolor=color, alpha=0.8))
    ax.axis('off')

    # LR
    ax = axes[1]
    ax.imshow(lr_img)
    ax.set_title(f"LR (128×128) — {patch_name}", fontsize=11)
    for obj in lr_objects:
        x1, y1, x2, y2 = obj['aabb']
        cls_name = obj['class']
        cls_idx = DOTA_CLASSES.index(cls_name) if cls_name in DOTA_CLASSES else 0
        color = CLASS_COLORS[cls_idx]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 3), f"{cls_name}", fontsize=7,
                color='white', bbox=dict(facecolor=color, alpha=0.8))
    ax.axis('off')

    plt.suptitle(f"HR ↔ LR Coordinate Verification: {patch_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_dataset_statistics(data_root, split='train'):
    """
    Compute and print comprehensive dataset statistics.
    """
    stats = {
        'num_images_hr': 0, 'num_images_lr': 0,
        'total_objects_hr': 0, 'total_objects_lr': 0,
        'class_counts': Counter(),
        'object_sizes': [],  # (width, height) in HR
        'aspect_ratios': [],
        'objects_per_image': [],
        'obb_vs_aabb_area_ratios': [],
    }

    for quality in ['hr', 'lr']:
        images_dir = os.path.join(data_root, f'{split}_{quality}', 'images')
        labels_dir = os.path.join(data_root, f'{split}_{quality}', 'labels')

        if not os.path.exists(images_dir):
            logger.warning(f"Directory not found: {images_dir}")
            continue

        image_files = sorted([f for f in os.listdir(images_dir)
                              if f.lower().endswith(('.jpg', '.png', '.tif'))])

        key = f'num_images_{quality}'
        stats[key] = len(image_files)

        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            label_path = os.path.join(labels_dir, f"{base_name}.txt")
            objects = load_annotations(label_path)

            stats[f'total_objects_{quality}'] += len(objects)

            if quality == 'hr':
                stats['objects_per_image'].append(len(objects))
                for obj in objects:
                    cls_name = obj['class']
                    stats['class_counts'][cls_name] += 1

                    x1, y1, x2, y2 = obj['aabb']
                    w = x2 - x1
                    h = y2 - y1
                    if w > 0 and h > 0:
                        stats['object_sizes'].append((w, h))
                        stats['aspect_ratios'].append(w / h)

                        # OBB area vs AABB area
                        obb = obj['obb']
                        obb_pts = np.array(obb).reshape(4, 2)
                        obb_area = 0.5 * abs(
                            np.sum(obb_pts[:, 0] * np.roll(obb_pts[:, 1], -1)) -
                            np.sum(obb_pts[:, 1] * np.roll(obb_pts[:, 0], -1))
                        )
                        aabb_area = w * h
                        if aabb_area > 0:
                            stats['obb_vs_aabb_area_ratios'].append(obb_area / aabb_area)

    return stats


def print_statistics(stats):
    """Pretty-print dataset statistics."""
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    print(f"\n📊 Image Counts:")
    print(f"  HR images: {stats['num_images_hr']}")
    print(f"  LR images: {stats['num_images_lr']}")

    print(f"\n📦 Object Counts:")
    print(f"  Total objects (HR): {stats['total_objects_hr']}")
    print(f"  Total objects (LR): {stats['total_objects_lr']}")
    if stats['objects_per_image']:
        print(f"  Objects per image: mean={np.mean(stats['objects_per_image']):.1f}, "
              f"median={np.median(stats['objects_per_image']):.1f}, "
              f"max={np.max(stats['objects_per_image'])}")

    print(f"\n🏷️  Per-Class Distribution:")
    for cls_name in DOTA_CLASSES:
        count = stats['class_counts'].get(cls_name, 0)
        bar = '█' * (count // max(1, max(stats['class_counts'].values()) // 40))
        print(f"  {cls_name:25s}: {count:6d} {bar}")

    if stats['object_sizes']:
        sizes = np.array(stats['object_sizes'])
        print(f"\n📐 Object Size Distribution (HR, pixels):")
        print(f"  Width:  mean={sizes[:, 0].mean():.1f}, std={sizes[:, 0].std():.1f}, "
              f"min={sizes[:, 0].min():.1f}, max={sizes[:, 0].max():.1f}")
        print(f"  Height: mean={sizes[:, 1].mean():.1f}, std={sizes[:, 1].std():.1f}, "
              f"min={sizes[:, 1].min():.1f}, max={sizes[:, 1].max():.1f}")

        # Size categories (COCO-style)
        areas = sizes[:, 0] * sizes[:, 1]
        small = (areas < 32 * 32).sum()
        medium = ((areas >= 32 * 32) & (areas < 96 * 96)).sum()
        large = (areas >= 96 * 96).sum()
        print(f"\n  Size breakdown:")
        print(f"    Small  (<32²):  {small:5d} ({100 * small / len(areas):.1f}%)")
        print(f"    Medium (32²-96²): {medium:5d} ({100 * medium / len(areas):.1f}%)")
        print(f"    Large  (>96²):  {large:5d} ({100 * large / len(areas):.1f}%)")

    if stats['aspect_ratios']:
        ar = np.array(stats['aspect_ratios'])
        print(f"\n📏 Aspect Ratio Distribution:")
        print(f"  Mean: {ar.mean():.2f}, Std: {ar.std():.2f}")
        print(f"  Min: {ar.min():.2f}, Max: {ar.max():.2f}")

    if stats['obb_vs_aabb_area_ratios']:
        ratios = np.array(stats['obb_vs_aabb_area_ratios'])
        print(f"\n🔄 OBB/AABB Area Ratio (1.0 = perfectly axis-aligned):")
        print(f"  Mean: {ratios.mean():.3f}, Std: {ratios.std():.3f}")
        print(f"  Min: {ratios.min():.3f}, Max: {ratios.max():.3f}")
        tight = (ratios > 0.9).sum()
        loose = (ratios < 0.7).sum()
        print(f"  Tight (>0.9): {tight} ({100 * tight / len(ratios):.1f}%)")
        print(f"  Loose (<0.7): {loose} ({100 * loose / len(ratios):.1f}%)")
        if ratios.mean() < 0.75:
            print("  ⚠️  WARNING: Many objects are significantly rotated!")
            print("     AABB approximation introduces substantial area inflation.")
            print("     Consider OBB-aware evaluation for accurate mAP.")

    print("\n" + "=" * 70)


def verify_coordinate_scaling(data_root, split='train', num_check=50):
    """
    Verify that HR and LR annotations are consistently scaled.
    Reports any mismatches.
    """
    hr_labels_dir = os.path.join(data_root, f'{split}_hr', 'labels')
    lr_labels_dir = os.path.join(data_root, f'{split}_lr', 'labels')

    if not os.path.exists(hr_labels_dir) or not os.path.exists(lr_labels_dir):
        logger.warning("Both HR and LR label directories must exist for scaling check")
        return

    hr_files = sorted(os.listdir(hr_labels_dir))[:num_check]
    errors = []
    scale_factor = 128.0 / 1024.0  # Expected scale

    for label_file in hr_files:
        hr_path = os.path.join(hr_labels_dir, label_file)
        lr_path = os.path.join(lr_labels_dir, label_file)

        hr_objects = load_annotations(hr_path)
        lr_objects = load_annotations(lr_path)

        if len(hr_objects) != len(lr_objects):
            errors.append(f"{label_file}: object count mismatch HR={len(hr_objects)} LR={len(lr_objects)}")
            continue

        for i, (hr_obj, lr_obj) in enumerate(zip(hr_objects, lr_objects)):
            hr_aabb = np.array(hr_obj['aabb'])
            lr_aabb = np.array(lr_obj['aabb'])
            expected_lr = hr_aabb * scale_factor

            if not np.allclose(expected_lr, lr_aabb, atol=1.0):
                errors.append(
                    f"{label_file} obj {i}: HR {hr_aabb} * {scale_factor} = "
                    f"{expected_lr} ≠ LR {lr_aabb}"
                )

    if errors:
        logger.error(f"❌ Found {len(errors)} coordinate scaling errors:")
        for err in errors[:10]:
            logger.error(f"   {err}")
        if len(errors) > 10:
            logger.error(f"   ... and {len(errors) - 10} more")
    else:
        logger.info(f"✅ All {len(hr_files)} checked files have correct HR↔LR scaling")


def main():
    parser = argparse.ArgumentParser(description="Verify dataset quality before training")
    parser.add_argument('--data_root', type=str, default='./datasets/processed',
                        help='Path to processed dataset')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='./outputs/data_verification',
                        help='Where to save verification plots')
    parser.add_argument('--stats', action='store_true',
                        help='Print dataset statistics only (no visualization)')
    parser.add_argument('--resolution', type=str, default='both',
                        choices=['hr', 'lr', 'both'],
                        help='Which resolution to verify')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================================
    # 1. Dataset Statistics
    # =========================================================================
    logger.info("Computing dataset statistics...")
    stats = compute_dataset_statistics(args.data_root, args.split)
    print_statistics(stats)

    if args.stats:
        return

    # =========================================================================
    # 2. Coordinate Scaling Verification
    # =========================================================================
    logger.info("\nVerifying HR ↔ LR coordinate scaling...")
    verify_coordinate_scaling(args.data_root, args.split)

    # =========================================================================
    # 3. Visual Verification
    # =========================================================================
    logger.info(f"\nGenerating {args.num_samples} verification plots...")

    for quality in (['hr', 'lr'] if args.resolution == 'both' else [args.resolution]):
        images_dir = os.path.join(args.data_root, f'{args.split}_{quality}', 'images')
        labels_dir = os.path.join(args.data_root, f'{args.split}_{quality}', 'labels')

        if not os.path.exists(images_dir):
            logger.warning(f"Directory not found: {images_dir}")
            continue

        image_files = sorted([f for f in os.listdir(images_dir)
                              if f.lower().endswith(('.jpg', '.png', '.tif'))])

        # Sample evenly across dataset
        indices = np.linspace(0, len(image_files) - 1, min(args.num_samples, len(image_files)))
        indices = [int(i) for i in indices]

        for idx in indices:
            img_file = image_files[idx]
            base_name = os.path.splitext(img_file)[0]

            img_path = os.path.join(images_dir, img_file)
            label_path = os.path.join(labels_dir, f"{base_name}.txt")

            objects = load_annotations(label_path)

            save_path = os.path.join(args.output_dir, f"{quality}_{base_name}_verify.png")
            draw_verification_plot(
                img_path, objects, save_path,
                title=f"{quality.upper()} {base_name} ({len(objects)} objects)"
            )
            logger.info(f"  Saved: {save_path}")

    # =========================================================================
    # 4. HR ↔ LR Side-by-Side Comparison
    # =========================================================================
    if args.resolution == 'both':
        logger.info("\nGenerating HR ↔ LR comparison plots...")
        hr_images_dir = os.path.join(args.data_root, f'{args.split}_hr', 'images')
        lr_images_dir = os.path.join(args.data_root, f'{args.split}_lr', 'images')
        hr_labels_dir = os.path.join(args.data_root, f'{args.split}_hr', 'labels')
        lr_labels_dir = os.path.join(args.data_root, f'{args.split}_lr', 'labels')

        if os.path.exists(hr_images_dir) and os.path.exists(lr_images_dir):
            hr_files = sorted([f for f in os.listdir(hr_images_dir)
                               if f.lower().endswith(('.jpg', '.png', '.tif'))])

            indices = np.linspace(0, len(hr_files) - 1, min(args.num_samples, len(hr_files)))
            indices = [int(i) for i in indices]

            for idx in indices:
                img_file = hr_files[idx]
                base_name = os.path.splitext(img_file)[0]

                hr_path = os.path.join(hr_images_dir, img_file)
                lr_path = os.path.join(lr_images_dir, img_file)

                if not os.path.exists(lr_path):
                    continue

                hr_objects = load_annotations(os.path.join(hr_labels_dir, f"{base_name}.txt"))
                lr_objects = load_annotations(os.path.join(lr_labels_dir, f"{base_name}.txt"))

                save_path = os.path.join(args.output_dir, f"compare_{base_name}.png")
                draw_hr_lr_comparison(
                    hr_path, lr_path, hr_objects, lr_objects,
                    save_path, base_name
                )
                logger.info(f"  Saved: {save_path}")

    logger.info(f"\n✅ Verification complete! Check outputs in: {args.output_dir}")
    logger.info("Review the generated plots to confirm GT boxes align with objects.")


if __name__ == '__main__':
    main()
