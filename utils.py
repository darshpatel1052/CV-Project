"""
Utility functions for Resolution-Agnostic Knowledge Distillation project.
Includes: anchor generation, NMS, metrics, data loading helpers, etc.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict
import yaml
import logging

# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    Set up logger with console and optional file output.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
    
    Returns:
        logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# CONFIGURATION UTILITIES
# ============================================================================

def load_config(config_path: str) -> Dict:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to config.yaml
    
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, output_path: str):
    """Save configuration to YAML file."""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


# ============================================================================
# ANCHOR GENERATION (for RetinaNet)
# ============================================================================

def generate_anchors(
    base_size: int = 16,
    ratios: List[float] = None,
    scales: List[float] = None
) -> np.ndarray:
    """
    Generate anchor boxes for a given base size.
    
    Args:
        base_size: Base anchor size (default 16 pixels)
        ratios: Aspect ratios (default [0.5, 1.0, 2.0])
        scales: Scales (default [1, 2^(1/3), 2^(2/3)])
    
    Returns:
        anchors: Array of shape (num_anchors, 4) with [x1, y1, x2, y2]
    
    Example:
        >>> anchors = generate_anchors(base_size=16)
        >>> print(anchors.shape)  # (9, 4)
    """
    if ratios is None:
        ratios = [0.5, 1.0, 2.0]
    if scales is None:
        scales = np.array([1.0, 2**(1/3), 2**(2/3)])
    
    ratios = np.array(ratios)
    scales = np.array(scales)
    
    num_anchors = len(ratios) * len(scales)
    anchors = np.zeros((num_anchors, 4))
    
    idx = 0
    for ratio in ratios:
        for scale in scales:
            w = (base_size * scale) * np.sqrt(ratio)
            h = (base_size * scale) / np.sqrt(ratio)
            
            # Create anchor centered at origin
            anchors[idx] = [-w/2, -h/2, w/2, h/2]
            idx += 1
    
    return anchors


def generate_feature_anchors(
    feature_map_size: int,
    stride: int,
    anchors_base: np.ndarray
) -> np.ndarray:
    """
    Generate anchors for entire feature map.
    
    Args:
        feature_map_size: Size of feature map (e.g., 64 for 64x64)
        stride: Stride from image to feature map (e.g., 4 for P3)
        anchors_base: Base anchors from generate_anchors()
    
    Returns:
        anchors: Array of shape (feature_map_size**2, num_base_anchors, 4)
    """
    num_base_anchors = anchors_base.shape[0]
    all_anchors = []
    
    for y in range(feature_map_size):
        for x in range(feature_map_size):
            # Feature map coordinates to image coordinates
            cx = (x + 0.5) * stride
            cy = (y + 0.5) * stride
            
            # Shift anchors to this position
            anchors = anchors_base.copy()
            anchors[:, 0] += cx  # x1
            anchors[:, 2] += cx  # x2
            anchors[:, 1] += cy  # y1
            anchors[:, 3] += cy  # y2
            
            all_anchors.append(anchors)
    
    all_anchors = np.stack(all_anchors)  # (feature_map_size**2, num_base_anchors, 4)
    return all_anchors


# ============================================================================
# NMS (Non-Maximum Suppression)
# ============================================================================

def compute_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU (Intersection over Union) between two sets of boxes.
    
    Args:
        boxes1: Array of shape (N, 4) with [x1, y1, x2, y2]
        boxes2: Array of shape (M, 4)
    
    Returns:
        iou: Array of shape (N, M)
    """
    # Compute areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Compute intersections
    x1_inter = np.maximum(boxes1[:, 0:1], boxes2[:, 0])
    y1_inter = np.maximum(boxes1[:, 1:2], boxes2[:, 1])
    x2_inter = np.minimum(boxes1[:, 2:3], boxes2[:, 2])
    y2_inter = np.minimum(boxes1[:, 3:4], boxes2[:, 3])
    
    inter_w = np.maximum(0, x2_inter - x1_inter)
    inter_h = np.maximum(0, y2_inter - y1_inter)
    inter_area = inter_w * inter_h
    
    # Compute union
    union_area = area1[:, np.newaxis] + area2 - inter_area
    
    # Compute IoU
    iou = inter_area / (union_area + 1e-6)
    return iou


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """
    Non-Maximum Suppression.
    
    Args:
        boxes: Array of shape (N, 4) with [x1, y1, x2, y2]
        scores: Array of shape (N,)
        iou_threshold: IoU threshold for suppression
    
    Returns:
        indices: Indices of boxes to keep
    """
    # Sort by score
    sorted_indices = np.argsort(-scores)
    
    keep = []
    while len(sorted_indices) > 0:
        # Keep highest score box
        current_idx = sorted_indices[0]
        keep.append(current_idx)
        
        if len(sorted_indices) == 1:
            break
        
        # Compute IoU with remaining boxes
        current_box = boxes[current_idx:current_idx+1]
        remaining_boxes = boxes[sorted_indices[1:]]
        
        ious = compute_iou(current_box, remaining_boxes)
        
        # Keep only boxes with IoU < threshold
        mask = ious[0] < iou_threshold
        sorted_indices = sorted_indices[1:][mask]
    
    return np.array(keep)


# ============================================================================
# METRICS
# ============================================================================

def compute_precision_recall(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 15
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute precision-recall curves for object detection.
    
    Args:
        predictions: List of dicts with 'boxes', 'scores', 'class_ids'
        ground_truths: List of dicts with 'boxes', 'class_ids'
        iou_threshold: IoU threshold for matching
        num_classes: Number of classes
    
    Returns:
        precisions: Array of shape (num_classes,)
        recalls: Array of shape (num_classes,)
    """
    precisions = np.zeros(num_classes)
    recalls = np.zeros(num_classes)
    
    for class_id in range(num_classes):
        # Filter predictions and ground truths for this class
        preds_class = [p for p in predictions if p['class_id'] == class_id]
        gts_class = [g for g in ground_truths if g['class_id'] == class_id]
        
        if len(gts_class) == 0:
            continue
        
        # Sort predictions by score
        preds_class = sorted(preds_class, key=lambda x: x['score'], reverse=True)
        
        tp = np.zeros(len(preds_class))
        fp = np.zeros(len(preds_class))
        
        gt_matched = set()
        
        for pred_idx, pred in enumerate(preds_class):
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gts_class):
                if gt_idx in gt_matched:
                    continue
                
                iou = compute_iou(
                    np.array([pred['box']]),
                    np.array([gt['box']])
                )[0, 0]
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[pred_idx] = 1
                gt_matched.add(best_gt_idx)
            else:
                fp[pred_idx] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recall = tp_cumsum / len(gts_class)
        
        precisions[class_id] = np.mean(precision)
        recalls[class_id] = np.mean(recall)
    
    return precisions, recalls


def compute_map(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """
    Compute Mean Average Precision (mAP).
    
    Args:
        precisions: Precision array
        recalls: Recall array
    
    Returns:
        mAP: Mean average precision
    """
    return np.mean(precisions)


# ============================================================================
# DEVICE UTILITIES
# ============================================================================

def get_device() -> torch.device:
    """Get GPU device if available, else CPU."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def get_device_info() -> str:
    """Get information about available compute devices."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"GPU: {device_name} ({device_memory:.1f}GB)"
    else:
        return "CPU only (no GPU detected)"


# ============================================================================
# CHECKPOINT UTILITIES
# ============================================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    checkpoint_path: str
):
    """Save model checkpoint with optimizer state and metrics."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, checkpoint_path)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    load_optimizer: bool = False
) -> Tuple[int, Dict]:
    """
    Load model checkpoint.
    
    Returns:
        epoch: Epoch number
        metrics: Metrics dict
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return checkpoint['epoch'], checkpoint['metrics']


if __name__ == "__main__":
    # Test functions
    print("Testing anchor generation...")
    anchors = generate_anchors(base_size=16)
    print(f"Generated {anchors.shape[0]} base anchors")
    
    print("\nTesting NMS...")
    test_boxes = np.array([[10, 10, 50, 50], [15, 15, 55, 55], [100, 100, 150, 150]])
    test_scores = np.array([0.9, 0.8, 0.7])
    keep_indices = nms(test_boxes, test_scores, iou_threshold=0.3)
    print(f"NMS kept indices: {keep_indices}")
    
    print("\nDevice info:", get_device_info())
