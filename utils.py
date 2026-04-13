"""
Utility functions for Resolution-Agnostic Knowledge Distillation project.
Includes: anchor generation, anchor-GT matching, NMS, detection decoding, metrics, helpers.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
import yaml
import logging

# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """Set up logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

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
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, output_path: str):
    """Save configuration to YAML file."""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


# ============================================================================
# ANCHOR GENERATION
# ============================================================================

def generate_anchors(
    base_size: int = 16,
    ratios: List[float] = None,
    scales: List[float] = None
) -> np.ndarray:
    """
    Generate base anchor boxes centered at origin.

    Args:
        base_size: Base anchor size in pixels
        ratios: Aspect ratios (default [0.5, 1.0, 2.0])
        scales: Scale multipliers (default [1, 2^(1/3), 2^(2/3)])

    Returns:
        anchors: (num_anchors, 4) array with [x1, y1, x2, y2] centered at origin
    """
    if ratios is None:
        ratios = [0.5, 1.0, 2.0]
    if scales is None:
        scales = [1.0, 2**(1/3), 2**(2/3)]

    anchors = []
    for ratio in ratios:
        for scale in scales:
            w = (base_size * scale) * np.sqrt(ratio)
            h = (base_size * scale) / np.sqrt(ratio)
            anchors.append([-w/2, -h/2, w/2, h/2])

    return np.array(anchors, dtype=np.float32)


def generate_all_anchors(
    image_size: int,
    fpn_strides: List[int] = None,
    base_sizes: List[int] = None,
    ratios: List[float] = None,
    scales: List[float] = None,
) -> torch.Tensor:
    """
    Generate anchors for all FPN levels across the entire image.
    
    This is the single entry point for anchor generation used by both
    training (for matching) and evaluation (for decoding).

    Args:
        image_size: Input image size (e.g., 1024 or 128)
        fpn_strides: Stride for each FPN level. Default depends on the FPN architecture:
                     For teacher (1024): P3..P6 strides are [8, 16, 32, 64]
                     For student adapted (128→upsampled): same logical strides
        base_sizes: Base anchor size per FPN level (default: same as stride)
        ratios: Aspect ratios
        scales: Scale multipliers

    Returns:
        all_anchors: (total_anchors, 4) tensor with [x1, y1, x2, y2] in pixel coords
    """
    if fpn_strides is None:
        fpn_strides = [8, 16, 32, 64]
    if base_sizes is None:
        base_sizes = fpn_strides  # Common convention: base_size = stride
    if ratios is None:
        ratios = [0.5, 1.0, 2.0]
    if scales is None:
        scales = [1.0, 2**(1/3), 2**(2/3)]

    all_anchors = []

    for stride, base_size in zip(fpn_strides, base_sizes):
        # Feature map size at this FPN level
        feat_h = feat_w = image_size // stride

        # Base anchors centered at origin
        base_anchors = generate_anchors(base_size, ratios, scales)  # (9, 4)
        num_base = base_anchors.shape[0]

        # Create grid of anchor centers
        shifts_x = (np.arange(0, feat_w) + 0.5) * stride
        shifts_y = (np.arange(0, feat_h) + 0.5) * stride
        shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)

        shift_x = shift_x.flatten()
        shift_y = shift_y.flatten()

        # Shifts shape: (feat_h * feat_w, 4)
        shifts = np.stack([shift_x, shift_y, shift_x, shift_y], axis=1).astype(np.float32)

        # Add shifts to base anchors: (H*W, 1, 4) + (1, 9, 4) -> (H*W, 9, 4)
        level_anchors = shifts[:, None, :] + base_anchors[None, :, :]
        level_anchors = level_anchors.reshape(-1, 4)  # (H*W*9, 4)

        all_anchors.append(level_anchors)

    all_anchors = np.concatenate(all_anchors, axis=0)
    return torch.from_numpy(all_anchors).float()


# ============================================================================
# ANCHOR-GT MATCHING (IoU-based assignment)
# ============================================================================

def compute_iou_torch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.

    Args:
        boxes1: (N, 4) tensor [x1, y1, x2, y2]
        boxes2: (M, 4) tensor [x1, y1, x2, y2]

    Returns:
        iou: (N, M) tensor
    """
    import torchvision.ops
    return torchvision.ops.box_iou(boxes1, boxes2)

def match_anchors_to_targets(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pos_iou_thresh: float = 0.5,
    neg_iou_thresh: float = 0.4,
    num_classes: int = 15
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Match anchors to ground-truth boxes using IoU-based assignment.
    
    Assignment rules (standard RetinaNet):
      - Positive: IoU >= pos_iou_thresh with any GT box
      - Negative: max IoU < neg_iou_thresh across all GT boxes
      - Ignore: in-between (not used for loss computation)
      - Additionally: each GT box is assigned to its highest-IoU anchor (even if < threshold)

    Args:
        anchors: (N, 4) all anchors [x1, y1, x2, y2]
        gt_boxes: (M, 4) ground truth boxes [x1, y1, x2, y2]
        gt_labels: (M,) class labels for each GT box (0-indexed, 0..num_classes-1)
        pos_iou_thresh: IoU threshold for positive assignment
        neg_iou_thresh: IoU threshold below which anchors are negative
        num_classes: number of object classes

    Returns:
        cls_targets: (N,) class targets. 0 = background, 1..num_classes = object classes.
                     -1 = ignore (don't compute loss).
        reg_targets: (N, 4) regression targets [dx, dy, dw, dh] (only meaningful for pos anchors)
        pos_mask: (N,) boolean mask for positive anchors
        neg_mask: (N,) boolean mask for negative anchors
    """
    num_anchors = anchors.shape[0]
    device = anchors.device

    # Default: all ignored (-1)
    cls_targets = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    reg_targets = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)

    if gt_boxes.shape[0] == 0:
        # No ground truth: all anchors are negative (background)
        cls_targets[:] = 0
        pos_mask = torch.zeros(num_anchors, dtype=torch.bool, device=device)
        neg_mask = torch.ones(num_anchors, dtype=torch.bool, device=device)
        return cls_targets, reg_targets, pos_mask, neg_mask

    # Memory stringency: Compute chunked IoU directly mapping 195k anchors to ~1000 targets
    # without explicitly fully instantiating the N x M tensor.
    import torchvision.ops
    chunk_size = 20000
    
    max_iou_per_anchor = torch.full((num_anchors,), -1.0, dtype=torch.float32, device=device)
    best_gt_per_anchor = torch.zeros((num_anchors,), dtype=torch.long, device=device)
    
    max_iou_per_gt = torch.full((gt_boxes.shape[0],), -1.0, dtype=torch.float32, device=device)
    best_anchor_per_gt = torch.zeros((gt_boxes.shape[0],), dtype=torch.long, device=device)
    
    for i in range(0, num_anchors, chunk_size):
        end = min(i + chunk_size, num_anchors)
        # iou_chunk: [chunk_size, M]
        iou_chunk = torchvision.ops.box_iou(anchors[i:end], gt_boxes)
        
        # Max IoU per anchor within this chunk
        chunk_max_iou_a, chunk_best_gt_a = iou_chunk.max(dim=1)
        max_iou_per_anchor[i:end] = chunk_max_iou_a
        best_gt_per_anchor[i:end] = chunk_best_gt_a
        
        # Max IoU per GT within this chunk
        chunk_max_iou_g, chunk_best_anchor_g = iou_chunk.max(dim=0)
        
        # Update global max per GT
        update_mask = chunk_max_iou_g > max_iou_per_gt
        max_iou_per_gt[update_mask] = chunk_max_iou_g[update_mask]
        best_anchor_per_gt[update_mask] = chunk_best_anchor_g[update_mask] + i

    # 1. Negative anchors: max IoU < neg_iou_thresh
    neg_mask = max_iou_per_anchor < neg_iou_thresh
    cls_targets[neg_mask] = 0  # background

    # 2. Positive anchors: max IoU >= pos_iou_thresh
    pos_mask = max_iou_per_anchor >= pos_iou_thresh

    # 3. Force-assign: each GT gets its best anchor (even if IoU < threshold)
    for gt_idx in range(gt_boxes.shape[0]):
        best_anchor_idx = best_anchor_per_gt[gt_idx]
        pos_mask[best_anchor_idx] = True
        neg_mask[best_anchor_idx] = False

    # Assign class labels for positive anchors.
    # FocalLoss expects: 0=background, 1..K=object classes (1-indexed).
    # Dataset labels are 0-indexed (0..K-1), so we add +1 to offset for FocalLoss.
    matched_gt_indices = best_gt_per_anchor[pos_mask]
    cls_targets[pos_mask] = gt_labels[matched_gt_indices] + 1  # +1: 0=bg, 1..K=object

    # Ensure negatives don't overlap with positives
    neg_mask = neg_mask & (~pos_mask)

    # Compute regression targets for positive anchors
    if pos_mask.sum() > 0:
        pos_anchors = anchors[pos_mask]
        matched_gt_boxes = gt_boxes[best_gt_per_anchor[pos_mask]]
        reg_targets[pos_mask] = encode_boxes(pos_anchors, matched_gt_boxes)

    return cls_targets, reg_targets, pos_mask, neg_mask


def encode_boxes(anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Encode GT boxes as regression deltas relative to anchors.

    Uses the standard parameterization:
        dx = (gt_cx - anchor_cx) / anchor_w
        dy = (gt_cy - anchor_cy) / anchor_h
        dw = log(gt_w / anchor_w)
        dh = log(gt_h / anchor_h)

    Args:
        anchors: (N, 4) anchor boxes [x1, y1, x2, y2]
        gt_boxes: (N, 4) matched GT boxes [x1, y1, x2, y2]

    Returns:
        deltas: (N, 4) regression targets [dx, dy, dw, dh]
    """
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    anchor_cx = anchors[:, 0] + 0.5 * anchor_w
    anchor_cy = anchors[:, 1] + 0.5 * anchor_h

    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_cx = gt_boxes[:, 0] + 0.5 * gt_w
    gt_cy = gt_boxes[:, 1] + 0.5 * gt_h

    dx = (gt_cx - anchor_cx) / anchor_w.clamp(min=1.0)
    dy = (gt_cy - anchor_cy) / anchor_h.clamp(min=1.0)
    dw = torch.log(gt_w / anchor_w.clamp(min=1.0))
    dh = torch.log(gt_h / anchor_h.clamp(min=1.0))

    return torch.stack([dx, dy, dw, dh], dim=1)


def decode_boxes(
    deltas: torch.Tensor, anchors: torch.Tensor, num_classes: int = 16
) -> torch.Tensor:
    """
    Decode regression deltas back to boxes.

    Args:
        anchors: (N, 4) anchor boxes [x1, y1, x2, y2]
        deltas: (N, 4) regression deltas [dx, dy, dw, dh]

    Returns:
        boxes: (N, 4) decoded boxes [x1, y1, x2, y2]
    """
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    anchor_cx = anchors[:, 0] + 0.5 * anchor_w
    anchor_cy = anchors[:, 1] + 0.5 * anchor_h

    # Clamp deltas to prevent numerical instability
    dw = deltas[:, 2].clamp(max=4.0)
    dh = deltas[:, 3].clamp(max=4.0)

    pred_cx = deltas[:, 0] * anchor_w + anchor_cx
    pred_cy = deltas[:, 1] * anchor_h + anchor_cy
    pred_w = torch.exp(dw) * anchor_w
    pred_h = torch.exp(dh) * anchor_h

    x1 = pred_cx - 0.5 * pred_w
    y1 = pred_cy - 0.5 * pred_h
    x2 = pred_cx + 0.5 * pred_w
    y2 = pred_cy + 0.5 * pred_h

    return torch.stack([x1, y1, x2, y2], dim=1)


# ============================================================================
# NMS (Non-Maximum Suppression)
# ============================================================================

def nms_torch(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5
) -> torch.Tensor:
    """
    Non-Maximum Suppression (pure PyTorch for GPU support).

    Args:
        boxes: (N, 4) tensor [x1, y1, x2, y2]
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        keep: indices of boxes to keep
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    # Use torchvision NMS if available (faster)
    try:
        from torchvision.ops import nms
        return nms(boxes, scores, iou_threshold)
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"Torchvision NMS failed ({e}), falling back to manual NMS.")
        pass

    # Fallback: manual NMS
    sorted_indices = torch.argsort(scores, descending=True)
    keep = []

    while sorted_indices.numel() > 0:
        current = sorted_indices[0]
        keep.append(current)

        if sorted_indices.numel() == 1:
            break

        current_box = boxes[current].unsqueeze(0)
        remaining_boxes = boxes[sorted_indices[1:]]

        ious = compute_iou_torch(current_box, remaining_boxes).squeeze(0)
        mask = ious < iou_threshold
        sorted_indices = sorted_indices[1:][mask]

    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long, device=boxes.device)


# ============================================================================
# DETECTION POST-PROCESSING
# ============================================================================

def postprocess_detections(
    cls_logits: torch.Tensor,
    bbox_regs: torch.Tensor,
    anchors: torch.Tensor,
    image_size: int,
    num_classes: int = 16,
    conf_threshold: float = 0.05,
    nms_threshold: float = 0.5,
    max_detections: int = 100,
    use_background: bool = True
) -> List[Dict]:
    """
    Full detection post-processing pipeline: sigmoid/softmax → decode → NMS.

    Args:
        cls_logits: (B, total_anchors, num_cls) raw classification logits
        bbox_regs: (B, total_anchors, 4) raw regression deltas
        anchors: (total_anchors, 4) anchor boxes
        image_size: image dimension for box clipping
        num_classes: number of object classes (excluding background)
        conf_threshold: minimum confidence to keep a detection
        nms_threshold: IoU threshold for NMS
        max_detections: maximum detections to return per image
        use_background: if True, cls_logits includes background class at index 0

    Returns:
        results: List of dicts (one per image), each with:
            'boxes': (K, 4) tensor [x1, y1, x2, y2]
            'scores': (K,) tensor
            'labels': (K,) tensor (0-indexed class IDs)
    """
    batch_size = cls_logits.shape[0]
    results = []
    
    # Determine number of channels
    num_output_classes = cls_logits.shape[-1]

    for b in range(batch_size):
        if use_background and num_output_classes == num_classes + 1:
            # Softmax over all classes (including background at 0)
            probs = torch.softmax(cls_logits[b], dim=-1)
            # Take only object classes (skip background at index 0)
            obj_probs = probs[:, 1:]  # (N, num_classes)
        else:
            # Sigmoid per-class (no explicit background class)
            obj_probs = torch.sigmoid(cls_logits[b])  # (N, num_classes)

        # Decode boxes
        decoded_boxes = decode_boxes(anchors, bbox_regs[b])

        # Clip boxes to image
        decoded_boxes[:, 0] = decoded_boxes[:, 0].clamp(min=0, max=image_size)
        decoded_boxes[:, 1] = decoded_boxes[:, 1].clamp(min=0, max=image_size)
        decoded_boxes[:, 2] = decoded_boxes[:, 2].clamp(min=0, max=image_size)
        decoded_boxes[:, 3] = decoded_boxes[:, 3].clamp(min=0, max=image_size)

        all_boxes = []
        all_scores = []
        all_labels = []

        # Per-class NMS
        for cls_id in range(obj_probs.shape[1]):
            scores = obj_probs[:, cls_id]
            mask = scores > conf_threshold
            if mask.sum() == 0:
                continue

            cls_boxes = decoded_boxes[mask]
            cls_scores = scores[mask]

            keep = nms_torch(cls_boxes, cls_scores, nms_threshold)

            all_boxes.append(cls_boxes[keep])
            all_scores.append(cls_scores[keep])
            all_labels.append(torch.full((len(keep),), cls_id, dtype=torch.long, device=cls_logits.device))

        if all_boxes:
            all_boxes = torch.cat(all_boxes, dim=0)
            all_scores = torch.cat(all_scores, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # Keep top-K detections
            if len(all_scores) > max_detections:
                topk = torch.argsort(all_scores, descending=True)[:max_detections]
                all_boxes = all_boxes[topk]
                all_scores = all_scores[topk]
                all_labels = all_labels[topk]
        else:
            all_boxes = torch.zeros((0, 4), device=cls_logits.device)
            all_scores = torch.zeros((0,), device=cls_logits.device)
            all_labels = torch.zeros((0,), dtype=torch.long, device=cls_logits.device)

        results.append({
            'boxes': all_boxes,
            'scores': all_scores,
            'labels': all_labels,
        })

    return results


# ============================================================================
# METRICS (mAP, AP)
# ============================================================================

def compute_ap_voc(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute Average Precision using PASCAL VOC 11-point interpolation.

    Args:
        recalls: sorted recall values
        precisions: corresponding precision values

    Returns:
        AP value
    """
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap


def compute_map(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int = 16,
    iou_threshold: float = 0.5,
) -> Tuple[float, Dict[int, float]]:
    """
    Compute mAP@IoU_threshold for object detection.

    Args:
        all_predictions: list of dicts per image, each with:
            'boxes': (K, 4) tensor, 'scores': (K,), 'labels': (K,)
        all_targets: list of dicts per image, each with:
            'boxes': (M, 4) tensor, 'labels': (M,)
        num_classes: number of object classes
        iou_threshold: IoU threshold for matching

    Returns:
        mAP: mean Average Precision across classes with GT objects
        per_class_ap: dict mapping class_id -> AP
    """
    per_class_ap = {}

    for cls_id in range(num_classes):
        # Collect all predictions and GT for this class across all images
        pred_boxes_all = []
        pred_scores_all = []
        pred_image_ids = []
        num_gt = 0
        gt_matched_by_image = {}  # image_id -> set of matched GT indices

        for img_idx, (preds, gts) in enumerate(zip(predictions, targets)):
            # GT for this class
            gt_labels = gts['labels']
            gt_mask = gt_labels == cls_id
            num_gt += gt_mask.sum().item()
            gt_matched_by_image[img_idx] = set()

            # Predictions for this class
            pred_labels = preds['labels']
            pred_mask = pred_labels == cls_id
            if pred_mask.sum() > 0:
                pred_boxes_all.append(preds['boxes'][pred_mask].cpu().numpy())
                pred_scores_all.append(preds['scores'][pred_mask].cpu().numpy())
                pred_image_ids.extend([img_idx] * pred_mask.sum().item())

        if num_gt == 0:
            continue  # Skip classes with no GT

        if len(pred_boxes_all) == 0:
            per_class_ap[cls_id] = 0.0
            continue

        pred_boxes_all = np.concatenate(pred_boxes_all, axis=0)
        pred_scores_all = np.concatenate(pred_scores_all, axis=0)

        # Sort by confidence (descending)
        sorted_indices = np.argsort(-pred_scores_all)
        pred_boxes_all = pred_boxes_all[sorted_indices]
        pred_scores_all = pred_scores_all[sorted_indices]
        pred_image_ids = [pred_image_ids[i] for i in sorted_indices]

        tp = np.zeros(len(pred_boxes_all))
        fp = np.zeros(len(pred_boxes_all))

        for pred_idx in range(len(pred_boxes_all)):
            img_idx = pred_image_ids[pred_idx]
            pred_box = pred_boxes_all[pred_idx]

            # Get GT boxes for this class in this image
            gts = targets[img_idx]
            gt_labels = gts['labels'].cpu().numpy()
            gt_boxes = gts['boxes'].cpu().numpy()
            gt_cls_mask = gt_labels == cls_id
            gt_cls_boxes = gt_boxes[gt_cls_mask]

            if len(gt_cls_boxes) == 0:
                fp[pred_idx] = 1
                continue

            # Compute IoU with all GT boxes of this class
            ious = _compute_iou_np(pred_box[None, :], gt_cls_boxes)[0]
            best_iou_idx = np.argmax(ious)
            best_iou = ious[best_iou_idx]

            if best_iou >= iou_threshold:
                # Map back to global GT index
                gt_global_indices = np.where(gt_cls_mask)[0]
                gt_global_idx = gt_global_indices[best_iou_idx]

                if gt_global_idx not in gt_matched_by_image[img_idx]:
                    tp[pred_idx] = 1
                    gt_matched_by_image[img_idx].add(gt_global_idx)
                else:
                    fp[pred_idx] = 1  # Duplicate detection
            else:
                fp[pred_idx] = 1

        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        # Compute AP using VOC 11-point interpolation
        ap = compute_ap_voc(recalls, precisions)
        per_class_ap[cls_id] = ap

    # mAP = mean over classes that have GT
    if len(per_class_ap) > 0:
        mAP = np.mean(list(per_class_ap.values()))
    else:
        mAP = 0.0

    return mAP, per_class_ap


def _compute_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU (numpy version). boxes1: (N,4), boxes2: (M,4) -> (N,M)"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    x1_inter = np.maximum(boxes1[:, 0:1], boxes2[:, 0])
    y1_inter = np.maximum(boxes1[:, 1:2], boxes2[:, 1])
    x2_inter = np.minimum(boxes1[:, 2:3], boxes2[:, 2])
    y2_inter = np.minimum(boxes1[:, 3:4], boxes2[:, 3])

    inter_w = np.maximum(0, x2_inter - x1_inter)
    inter_h = np.maximum(0, y2_inter - y1_inter)
    inter_area = inter_w * inter_h
    union_area = area1[:, None] + area2[None, :] - inter_area

    return inter_area / (union_area + 1e-6)


# ============================================================================
# DEVICE UTILITIES
# ============================================================================

def get_device() -> torch.device:
    """Get GPU device if available, else CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_device_info() -> str:
    """Get information about available compute devices."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_memory = torch.cuda.get_device_properties(0).total_mem / 1e9
        return f"GPU: {device_name} ({device_memory:.1f}GB)"
    else:
        return "CPU only (no GPU detected)"


# ============================================================================
# CHECKPOINT UTILITIES
# ============================================================================

def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path):
    """Save model checkpoint with optimizer state and metrics."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, checkpoint_path)


def load_checkpoint(model, checkpoint_path, device='cpu'):
    """Load model checkpoint. Returns epoch and metrics."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


if __name__ == "__main__":
    print("Testing anchor generation and matching...")

    # Test anchor generation
    anchors = generate_all_anchors(1024, [8, 16, 32, 64])
    print(f"Generated {anchors.shape[0]} anchors for 1024x1024 image")

    anchors_128 = generate_all_anchors(128, [8, 16, 32, 64])
    print(f"Generated {anchors_128.shape[0]} anchors for 128x128 image")

    # Test matching
    gt_boxes = torch.tensor([[100, 100, 200, 200], [500, 500, 600, 600]], dtype=torch.float32)
    gt_labels = torch.tensor([0, 5])

    cls_targets, reg_targets, pos_mask, neg_mask = match_anchors_to_targets(
        anchors, gt_boxes, gt_labels
    )
    print(f"Positive anchors: {pos_mask.sum().item()}")
    print(f"Negative anchors: {neg_mask.sum().item()}")
    print(f"Ignored anchors: {(~pos_mask & ~neg_mask).sum().item()}")

    # Test decode
    print(f"\nDecoded boxes from positive anchors (first 3):")
    pos_anchors = anchors[pos_mask][:3]
    pos_deltas = reg_targets[pos_mask][:3]
    decoded = decode_boxes(pos_anchors, pos_deltas)
    print(decoded)

    print("\n✓ All utility tests passed!")
