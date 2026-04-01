"""
Visualization Script
====================

Visualize model predictions and internal representations:

1. DETECTION BOXES: Draw detected objects (bounding boxes + class names)
   └─ Shows what the model actually detects on test images

2. FEATURE MAPS: Visualize intermediate FPN features
   └─ Shows how the model sees different scales

3. HEATMAPS: Attention/activation maps
   └─ Shows which image regions the model focuses on

This helps understand model behavior and debug issues.
"""

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader

from utils import load_config, setup_logger
from data.dataset import DOTADetectionDataset
from models.teacher import TeacherDetector
from models.student import StudentDetector


def denormalize(img_tensor):
    """
    Convert a normalized image tensor back to displayable format.
    
    Args:
        img_tensor: Tensor of shape (3, H, W), values in [0, 1]
    
    Returns:
        numpy array of shape (H, W, 3), values in [0, 255]
    """
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    return img


def visualize_detections(image, boxes, labels, class_names, save_path=None):
    """
    Draw bounding boxes and class labels on an image.
    
    Args:
        image: Tensor of shape (3, H, W) or numpy array (H, W, 3)
        boxes: Array of shape (N, 4) in format [x1, y1, x2, y2]
        labels: Array of class IDs
        class_names: List of class name strings
        save_path: Path to save the visualization
    
    Returns:
        fig: Matplotlib figure object
    """
    # Convert to displayable format
    if isinstance(image, torch.Tensor):
        img_display = denormalize(image)
    else:
        img_display = image.astype(np.uint8)
    
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(img_display)
    ax.set_title("Object Detections")
    
    # Draw each box
    colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))
    
    for box, label_id in zip(boxes, labels):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor=colors[int(label_id)],
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label text
        class_name = class_names[int(label_id)]
        ax.text(
            x1, y1 - 5,
            class_name,
            fontsize=10,
            color='white',
            bbox=dict(facecolor=colors[int(label_id)], alpha=0.7)
        )
    
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    return fig


def visualize_feature_maps(feature_maps, save_path=None, num_channels=4):
    """
    Visualize FPN feature maps across scales.
    
    Args:
        feature_maps: List of 4 feature tensors [P3, P4, P5, P6]
        save_path: Path to save visualization
        num_channels: Number of channels to display per level
    
    Returns:
        fig: Matplotlib figure object
    """
    fig, axes = plt.subplots(4, num_channels, figsize=(16, 12))
    fig.suptitle("FPN Feature Maps (4 Levels × First 4 Channels)")
    
    for level_idx, feat in enumerate(feature_maps):
        # feat shape: (B, C, H, W) - take first image in batch
        feat_data = feat[0].cpu().detach()  # (C, H, W)
        
        for ch_idx in range(min(num_channels, feat_data.shape[0])):
            ax = axes[level_idx, ch_idx]
            
            # Normalize channel to [0, 1] for visualization
            channel = feat_data[ch_idx]
            channel_min = channel.min()
            channel_max = channel.max()
            if channel_max > channel_min:
                channel_norm = (channel - channel_min) / (channel_max - channel_min)
            else:
                channel_norm = channel
            
            ax.imshow(channel_norm.numpy(), cmap='viridis')
            ax.set_title(f"P{level_idx+3}_ch{ch_idx}")
            ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
    
    return fig


def visualize_heatmap(image, feature_map, save_path=None):
    """
    Overlay activation heatmap on original image.
    
    Args:
        image: Tensor (3, H, W) or array (H, W, 3)
        feature_map: Tensor (B, C, H_feat, W_feat)
        save_path: Path to save visualization
    
    Returns:
        fig: Matplotlib figure object
    """
    # Convert image to displayable format
    if isinstance(image, torch.Tensor):
        img_display = denormalize(image)
    else:
        img_display = image.astype(np.uint8)
    
    # Get heatmap from first image and average across channels
    feat_avg = feature_map[0].mean(dim=0).cpu().detach().numpy()
    
    # Normalize to [0, 1]
    feat_min = feat_avg.min()
    feat_max = feat_avg.max()
    if feat_max > feat_min:
        feat_norm = (feat_avg - feat_min) / (feat_max - feat_min)
    else:
        feat_norm = feat_avg
    
    # Upscale feature map to match image size
    from scipy import ndimage
    h, w = img_display.shape[:2]
    feat_big = ndimage.zoom(feat_norm, [h / feat_norm.shape[0], w / feat_norm.shape[1]])
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Original image
    axes[0].imshow(img_display)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(feat_big, cmap='hot')
    axes[1].set_title("Activation Heatmap")
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(img_display)
    axes[2].imshow(feat_big, cmap='hot', alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize model predictions and features")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--model', type=str, choices=['teacher', 'student_baseline', 'student_kd'],
                       default='student_kd', help='Which model to visualize')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("Visualize")
    logger.info(f"🎨 Starting visualization for {args.model}...")

    device = torch.device(args.device)
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    logger.info("📦 Loading test dataset...")
    
    if args.model == 'teacher':
        image_size = 1024
    else:
        image_size = 128
    
    test_dataset = DOTADetectionDataset(
        data_root=cfg['dataset']['processed_data_path'],
        split='val',
        image_size=image_size,
        subset_size=args.num_samples,
        augmentation=False
    )
    
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # =========================================================================
    # LOAD MODEL
    # =========================================================================
    logger.info(f"🧠 Loading {args.model} model...")
    
    model_configs = {
        'teacher': (TeacherDetector, './checkpoints/teacher/best_model.pth'),
        'student_baseline': (StudentDetector, './checkpoints/student_baseline/best_model.pth'),
        'student_kd': (StudentDetector, './checkpoints/student_kd/best_model.pth'),
    }
    
    model_class, checkpoint_path = model_configs[args.model]
    model = model_class(num_classes=cfg['dataset']['num_classes'], pretrained=False)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✅ Loaded checkpoint from {checkpoint_path}")
    else:
        logger.warning(f"⚠ Checkpoint not found: {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    
    # Create output directory
    output_dir = cfg.get('visualization', {}).get('output_dir', './outputs/visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # VISUALIZE SAMPLES
    # =========================================================================
    class_names = cfg['dataset']['classes']
    
    with torch.no_grad():
        for sample_idx, (images, targets) in enumerate(test_loader):
            if sample_idx >= args.num_samples:
                break
            
            logger.info(f"\nVisualizing sample {sample_idx + 1}/{args.num_samples}")
            
            image = images[0]
            target = targets[0]
            images_batch = torch.stack(images).to(device)
            
            # Forward pass
            outputs = model(images_batch)
            fpn_features = outputs['fpn_features']
            
            # =========================================================
            # 1. DETECTION BOXES VISUALIZATION
            # =========================================================
            logger.info(f"  Drawing detection boxes...")
            
            if 'boxes' in target and len(target['boxes']) > 0:
                gt_boxes = target['boxes'].numpy()
                gt_labels = target['labels'].numpy()
            else:
                gt_boxes = np.array([])
                gt_labels = np.array([])
            
            fig = visualize_detections(
                image, gt_boxes, gt_labels, class_names,
                save_path=os.path.join(output_dir, f'{args.model}_sample{sample_idx}_boxes.png')
            )
            plt.close(fig)
            
            # =========================================================
            # 2. FEATURE MAPS VISUALIZATION
            # =========================================================
            logger.info(f"  Visualizing feature maps...")
            
            fig = visualize_feature_maps(
                fpn_features,
                save_path=os.path.join(output_dir, f'{args.model}_sample{sample_idx}_features.png'),
                num_channels=4
            )
            plt.close(fig)
            
            # =========================================================
            # 3. HEATMAP VISUALIZATION
            # =========================================================
            logger.info(f"  Visualizing activation heatmap...")
            
            fig = visualize_heatmap(
                image, fpn_features[0],  # Use P3 (highest resolution)
                save_path=os.path.join(output_dir, f'{args.model}_sample{sample_idx}_heatmap.png')
            )
            plt.close(fig)
    
    logger.info(f"\n✅ Visualization complete! Saved to {output_dir}")


if __name__ == '__main__':
    main()
