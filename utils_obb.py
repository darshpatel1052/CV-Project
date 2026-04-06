"""
Oriented Bounding Box (OBB) Utilities
======================================

Implements OBB IoU and OBB NMS for DOTA-style rotated object detection.
These are provided for future extension and optional OBB-aware evaluation.

Primary experiments use AABB (axis-aligned) boxes, but these utilities
enable OBB evaluation to measure the gap between AABB-mAP and OBB-mAP.

OBB Representation:
    - 8 coordinates: [x1, y1, x2, y2, x3, y3, x4, y4] (4 corners, clockwise)
    - Or: [cx, cy, w, h, angle] (center, size, rotation)

Key Functions:
    - obb_to_polygon: Convert OBB corners to Shapely polygon
    - compute_obb_iou: IoU between two sets of oriented boxes
    - obb_nms: Non-Maximum Suppression using OBB IoU
    - aabb_from_obb: Convert OBB to axis-aligned bounding box
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
import logging
import math

logger = logging.getLogger(__name__)


# ============================================================================
# OBB GEOMETRY (Pure NumPy/Torch — no Shapely dependency)
# ============================================================================

def obb_corners_to_params(corners: np.ndarray) -> np.ndarray:
    """
    Convert OBB from 4-corner format to [cx, cy, w, h, angle] parameterization.

    Args:
        corners: (N, 8) array [x1,y1,x2,y2,x3,y3,x4,y4]

    Returns:
        params: (N, 5) array [cx, cy, w, h, angle_rad]
    """
    corners = corners.reshape(-1, 4, 2)  # (N, 4, 2)

    # Center = mean of all 4 corners
    cx = corners[:, :, 0].mean(axis=1)
    cy = corners[:, :, 1].mean(axis=1)

    # Width = distance between corner 0 and corner 1
    edge1 = corners[:, 1] - corners[:, 0]  # (N, 2)
    edge2 = corners[:, 2] - corners[:, 1]  # (N, 2)

    w = np.sqrt(np.sum(edge1 ** 2, axis=1))  # (N,)
    h = np.sqrt(np.sum(edge2 ** 2, axis=1))  # (N,)

    # Angle = atan2 of the first edge
    angle = np.arctan2(edge1[:, 1], edge1[:, 0])  # (N,)

    return np.stack([cx, cy, w, h, angle], axis=1)


def obb_params_to_corners(params: np.ndarray) -> np.ndarray:
    """
    Convert OBB from [cx, cy, w, h, angle] to 4-corner format.

    Args:
        params: (N, 5) array [cx, cy, w, h, angle_rad]

    Returns:
        corners: (N, 8) array [x1,y1,x2,y2,x3,y3,x4,y4]
    """
    cx, cy, w, h, angle = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4]
    N = len(cx)

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # Half dimensions
    hw, hh = w / 2, h / 2

    # 4 corners relative to center (before rotation)
    # Corner order: top-left, top-right, bottom-right, bottom-left
    dx = np.array([-hw, hw, hw, -hw])  # (4, N)
    dy = np.array([-hh, -hh, hh, hh])  # (4, N)

    # Rotate
    corners_x = cos_a * dx - sin_a * dy + cx  # (4, N)
    corners_y = sin_a * dx + cos_a * dy + cy  # (4, N)

    # Interleave x, y
    corners = np.zeros((N, 8))
    for i in range(4):
        corners[:, 2 * i] = corners_x[i]
        corners[:, 2 * i + 1] = corners_y[i]

    return corners


def aabb_from_obb(corners: np.ndarray) -> np.ndarray:
    """
    Convert oriented bounding boxes to axis-aligned bounding boxes.

    Args:
        corners: (N, 8) array [x1,y1, x2,y2, x3,y3, x4,y4]

    Returns:
        aabb: (N, 4) array [xmin, ymin, xmax, ymax]
    """
    corners_reshaped = corners.reshape(-1, 4, 2)  # (N, 4, 2)
    x_min = corners_reshaped[:, :, 0].min(axis=1)
    y_min = corners_reshaped[:, :, 1].min(axis=1)
    x_max = corners_reshaped[:, :, 0].max(axis=1)
    y_max = corners_reshaped[:, :, 1].max(axis=1)
    return np.stack([x_min, y_min, x_max, y_max], axis=1)


# ============================================================================
# OBB IoU COMPUTATION (Polygon Intersection via Sutherland-Hodgman)
# ============================================================================

def _polygon_area(vertices: np.ndarray) -> float:
    """
    Compute area of a polygon using the Shoelace formula.

    Args:
        vertices: (K, 2) array of polygon vertices in order
    """
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    return abs(area) / 2.0


def _line_segment_intersection(
    p1: np.ndarray, p2: np.ndarray,
    p3: np.ndarray, p4: np.ndarray
) -> Optional[np.ndarray]:
    """
    Compute intersection of line segment p1-p2 with line p3-p4.
    Used in Sutherland-Hodgman clipping.
    """
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-10:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    return p1 + t * d1


def _sutherland_hodgman_clip(
    subject: np.ndarray,
    clip: np.ndarray
) -> np.ndarray:
    """
    Clip a convex polygon (subject) by another convex polygon (clip)
    using the Sutherland-Hodgman algorithm.

    Args:
        subject: (N, 2) vertices of polygon to clip
        clip: (M, 2) vertices of clipping polygon

    Returns:
        clipped: (K, 2) vertices of intersection polygon
    """
    output = list(subject)
    if len(output) == 0:
        return np.array([])

    for i in range(len(clip)):
        if len(output) == 0:
            return np.array([])

        input_list = list(output)
        output = []

        edge_start = clip[i]
        edge_end = clip[(i + 1) % len(clip)]

        for j in range(len(input_list)):
            current = input_list[j]
            previous = input_list[j - 1]

            # Check if points are inside the clipping edge
            def is_inside(point):
                return (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) - \
                       (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0]) >= 0

            if is_inside(current):
                if not is_inside(previous):
                    intersection = _line_segment_intersection(
                        previous, current, edge_start, edge_end
                    )
                    if intersection is not None:
                        output.append(intersection)
                output.append(current)
            elif is_inside(previous):
                intersection = _line_segment_intersection(
                    previous, current, edge_start, edge_end
                )
                if intersection is not None:
                    output.append(intersection)

    if len(output) == 0:
        return np.array([])
    return np.array(output)


def compute_obb_iou_single(box1_corners: np.ndarray, box2_corners: np.ndarray) -> float:
    """
    Compute IoU between two oriented bounding boxes.

    Args:
        box1_corners: (8,) array [x1,y1,x2,y2,x3,y3,x4,y4]
        box2_corners: (8,) array [x1,y1,x2,y2,x3,y3,x4,y4]

    Returns:
        iou: float, Intersection over Union
    """
    poly1 = box1_corners.reshape(4, 2)
    poly2 = box2_corners.reshape(4, 2)

    # Compute areas
    area1 = _polygon_area(poly1)
    area2 = _polygon_area(poly2)

    if area1 < 1e-6 or area2 < 1e-6:
        return 0.0

    # Compute intersection using Sutherland-Hodgman
    intersection_poly = _sutherland_hodgman_clip(poly1, poly2)

    if len(intersection_poly) < 3:
        return 0.0

    inter_area = _polygon_area(intersection_poly)
    union_area = area1 + area2 - inter_area

    if union_area < 1e-6:
        return 0.0

    return inter_area / union_area


def compute_obb_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise OBB IoU between two sets of oriented boxes.

    Args:
        boxes1: (N, 8) array of OBB corners
        boxes2: (M, 8) array of OBB corners

    Returns:
        iou_matrix: (N, M) array of IoU values
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    iou_matrix = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        for j in range(M):
            iou_matrix[i, j] = compute_obb_iou_single(boxes1[i], boxes2[j])

    return iou_matrix


# ============================================================================
# OBB NMS (Non-Maximum Suppression with Oriented IoU)
# ============================================================================

def obb_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5
) -> np.ndarray:
    """
    Non-Maximum Suppression using OBB IoU.

    Args:
        boxes: (N, 8) OBB corners [x1,y1,...,x4,y4]
        scores: (N,) confidence scores
        iou_threshold: suppression threshold

    Returns:
        keep: indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    sorted_indices = np.argsort(-scores)
    keep = []

    while len(sorted_indices) > 0:
        current = sorted_indices[0]
        keep.append(current)

        if len(sorted_indices) == 1:
            break

        remaining = sorted_indices[1:]
        ious = np.array([
            compute_obb_iou_single(boxes[current], boxes[idx])
            for idx in remaining
        ])

        mask = ious < iou_threshold
        sorted_indices = remaining[mask]

    return np.array(keep, dtype=np.int64)


# ============================================================================
# OBB-AWARE mAP COMPUTATION
# ============================================================================

def compute_map_obb(
    predictions: List[Dict[str, np.ndarray]],
    targets: List[Dict[str, np.ndarray]],
    num_classes: int = 16,
    iou_threshold: float = 0.5
) -> Tuple[float, dict]:
    """
    Compute mAP using OBB IoU instead of AABB IoU.

    Predictions and targets must include 'obb' key with (N, 8) arrays.

    Args:
        all_predictions: list of dicts with 'obb', 'scores', 'labels'
        all_targets: list of dicts with 'obb', 'labels'
        num_classes: number of object classes
        iou_threshold: IoU threshold

    Returns:
        mAP, per_class_ap
    """
    per_class_ap = {}

    for cls_id in range(num_classes):
        pred_boxes_all = []
        pred_scores_all = []
        pred_image_ids = []
        num_gt = 0
        gt_matched_by_image = {}

        for img_idx, (preds, gts) in enumerate(zip(predictions, targets)):
            gt_mask = gts['labels'] == cls_id
            num_gt += gt_mask.sum()
            gt_matched_by_image[img_idx] = set()

            pred_mask = preds['labels'] == cls_id
            if pred_mask.sum() > 0:
                pred_boxes_all.append(preds['obb'][pred_mask])
                pred_scores_all.append(preds['scores'][pred_mask])
                pred_image_ids.extend([img_idx] * pred_mask.sum())

        if num_gt == 0:
            continue

        if len(pred_boxes_all) == 0:
            per_class_ap[cls_id] = 0.0
            continue

        pred_boxes_all = np.concatenate(pred_boxes_all, axis=0)
        pred_scores_all = np.concatenate(pred_scores_all, axis=0)

        sorted_idx = np.argsort(-pred_scores_all)
        pred_boxes_all = pred_boxes_all[sorted_idx]
        pred_image_ids = [pred_image_ids[i] for i in sorted_idx]

        tp = np.zeros(len(pred_boxes_all))
        fp = np.zeros(len(pred_boxes_all))

        for pred_idx in range(len(pred_boxes_all)):
            img_idx = pred_image_ids[pred_idx]
            pred_box = pred_boxes_all[pred_idx]

            gts = targets[img_idx]
            gt_cls_mask = gts['labels'] == cls_id
            gt_cls_boxes = gts['obb'][gt_cls_mask]

            if len(gt_cls_boxes) == 0:
                fp[pred_idx] = 1
                continue

            ious = np.array([
                compute_obb_iou_single(pred_box, gt_box)
                for gt_box in gt_cls_boxes
            ])
            best_idx = np.argmax(ious)
            best_iou = ious[best_idx]

            if best_iou >= iou_threshold:
                gt_global_indices = np.where(gt_cls_mask)[0]
                gt_global_idx = gt_global_indices[best_idx]
                if gt_global_idx not in gt_matched_by_image[img_idx]:
                    tp[pred_idx] = 1
                    gt_matched_by_image[img_idx].add(gt_global_idx)
                else:
                    fp[pred_idx] = 1
            else:
                fp[pred_idx] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        # 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        per_class_ap[cls_id] = ap

    if per_class_ap:
        mAP = np.mean(list(per_class_ap.values()))
    else:
        mAP = 0.0

    return mAP, per_class_ap


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing OBB Utilities")
    print("=" * 60)

    # Test 1: Identical boxes → IoU = 1.0
    box = np.array([0, 0, 100, 0, 100, 100, 0, 100], dtype=np.float32)
    iou = compute_obb_iou_single(box, box)
    print(f"\n1. Identical boxes IoU: {iou:.4f} (expected ~1.0)")
    assert abs(iou - 1.0) < 0.01, f"Expected 1.0, got {iou}"
    print("   ✓ PASS")

    # Test 2: Non-overlapping boxes → IoU = 0.0
    box1 = np.array([0, 0, 10, 0, 10, 10, 0, 10], dtype=np.float32)
    box2 = np.array([20, 20, 30, 20, 30, 30, 20, 30], dtype=np.float32)
    iou = compute_obb_iou_single(box1, box2)
    print(f"\n2. Non-overlapping IoU: {iou:.4f} (expected 0.0)")
    assert iou < 0.01, f"Expected 0.0, got {iou}"
    print("   ✓ PASS")

    # Test 3: 50% overlap
    box1 = np.array([0, 0, 10, 0, 10, 10, 0, 10], dtype=np.float32)
    box2 = np.array([5, 0, 15, 0, 15, 10, 5, 10], dtype=np.float32)
    iou = compute_obb_iou_single(box1, box2)
    expected_iou = 50.0 / 150.0  # intersection=50, union=100+100-50=150
    print(f"\n3. 50% overlap IoU: {iou:.4f} (expected {expected_iou:.4f})")
    assert abs(iou - expected_iou) < 0.02, f"Expected {expected_iou}, got {iou}"
    print("   ✓ PASS")

    # Test 4: Rotated box
    # 45-degree rotated square centered at (50, 50)
    import math
    side = 20
    angle = math.pi / 4
    cx, cy = 50, 50
    corners_rot = []
    for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        x = cx + side / 2 * (dx * math.cos(angle) - dy * math.sin(angle))
        y = cy + side / 2 * (dx * math.sin(angle) + dy * math.cos(angle))
        corners_rot.extend([x, y])
    box_rot = np.array(corners_rot, dtype=np.float32)
    iou_self = compute_obb_iou_single(box_rot, box_rot)
    print(f"\n4. Rotated self-IoU: {iou_self:.4f} (expected ~1.0)")
    assert abs(iou_self - 1.0) < 0.01
    print("   ✓ PASS")

    # Test 5: OBB NMS
    boxes = np.array([
        [0, 0, 10, 0, 10, 10, 0, 10],
        [1, 1, 11, 1, 11, 11, 1, 11],  # High overlap with first
        [50, 50, 60, 50, 60, 60, 50, 60],  # No overlap
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7])
    keep = obb_nms(boxes, scores, iou_threshold=0.3)
    print(f"\n5. OBB NMS keep indices: {keep} (expected [0, 2])")
    assert 0 in keep and 2 in keep
    print("   ✓ PASS")

    # Test 6: AABB from OBB
    obb = np.array([[0, 0, 10, 0, 10, 10, 0, 10]], dtype=np.float32)
    aabb = aabb_from_obb(obb)
    print(f"\n6. AABB from axis-aligned OBB: {aabb} (expected [0, 0, 10, 10])")
    assert np.allclose(aabb, [[0, 0, 10, 10]])
    print("   ✓ PASS")

    # Test 7: Batch IoU matrix
    boxes1 = np.array([
        [0, 0, 10, 0, 10, 10, 0, 10],
        [50, 50, 60, 50, 60, 60, 50, 60],
    ], dtype=np.float32)
    iou_mat = compute_obb_iou_matrix(boxes1, boxes1)
    print(f"\n7. IoU matrix diagonal: {np.diag(iou_mat)} (expected [1.0, 1.0])")
    assert np.allclose(np.diag(iou_mat), 1.0, atol=0.01)
    print("   ✓ PASS")

    print("\n" + "=" * 60)
    print("All OBB tests passed! ✓")
    print("=" * 60)
