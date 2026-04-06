"""
DOTA Dataset Loader for Object Detection — Enhanced
=====================================================

Loads DOTA v1.0/v1.5 dataset with Object Detection annotations.
Supports both high-resolution (1024×1024) and low-resolution (128×128) versions.

Key improvements:
  - Proper augmentations: horizontal flip, vertical flip, 90° rotations, color jitter
  - OBB corners preserved in targets for optional OBB evaluation
  - Bounding boxes scaled and clipped to target resolution
  - ImageNet normalization for pretrained backbones
  - Degenerate box filtering
  - Enhanced collate with proper batching

IMPORTANT: This uses AABB (axis-aligned bounding boxes) derived from OBB annotations.
           The OBB coordinates are preserved in targets['obb'] for future extension.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional
import yaml
import logging

logger = logging.getLogger(__name__)

# ImageNet normalization constants (required for pretrained Swin-T / MobileNetV2)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DOTADetectionDataset(Dataset):
    """
    DOTA Object Detection Dataset (v1.0/v1.5).

    Args:
        data_root: Path to processed dataset directory
        split: 'train' or 'val'
        image_size: Target image size (1024 for teacher, 128 for student)
        subset_size: If not None, use only first N images (for testing)
        augmentation: Boolean to apply augmentations

    Returns:
        image: torch.Tensor of shape (3, H, W), ImageNet-normalized
        targets: Dict with keys:
            - 'boxes': torch.Tensor of shape (num_objects, 4) in [x1, y1, x2, y2]
            - 'labels': torch.Tensor of shape (num_objects,) with class IDs (0-indexed)
            - 'obb': torch.Tensor of shape (num_objects, 8) with OBB corners (optional)
            - 'image_id': int, image index
            - 'filename': str, image filename
    """

    # DOTA class names (15 classes)
    CLASSES = [
        'plane', 'ship', 'storage-tank', 'baseball-diamond',
        'tennis-court', 'basketball-court', 'ground-track-field',
        'harbor', 'bridge', 'large-vehicle', 'small-vehicle',
        'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool',
        'container-crane'
    ]

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        image_size: int = 1024,
        subset_size: Optional[int] = None,
        augmentation: bool = False
    ):
        """Initialize dataset."""
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.augmentation = augmentation

        if split not in ['train', 'val', 'test']:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")

        self.class_to_id = {name: idx for idx, name in enumerate(self.CLASSES)}
        self.id_to_class = {idx: name for idx, name in enumerate(self.CLASSES)}

        # Determine resolution suffix
        resolution_suffix = 'hr' if image_size >= 512 else 'lr'
        split_dir = f"{split}_{resolution_suffix}"

        self.images_dir = os.path.join(data_root, split_dir, 'images')
        self.labels_dir = os.path.join(data_root, split_dir, 'labels')

        # Get list of image files
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(
                f"Images directory not found: {self.images_dir}\n"
                f"Run: python data/prepare_data.py to process the dataset first."
            )

        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith(('.tif', '.tiff', '.jpg', '.png'))
        ])

        # Apply subset if specified
        if subset_size is not None and subset_size > 0:
            self.image_files = self.image_files[:subset_size]

        logger.info(f"Loaded {len(self.image_files)} images from {split}_{resolution_suffix}")
        if subset_size is not None:
            logger.info(f"  (using subset of {subset_size})")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        image_file = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_file)

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise

        # Get original dimensions BEFORE resize
        orig_w, orig_h = image.size

        # Resize to target size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Scale factors for coordinate transformation
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h

        # Load annotations
        base_name = os.path.splitext(image_file)[0]
        label_path = os.path.join(self.labels_dir, f"{base_name}.txt")

        boxes = []
        labels = []
        obb_list = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('imagesource') or line.startswith('gsd:'):
                        continue

                    parts = line.split()
                    if len(parts) < 9:
                        continue

                    try:
                        # Parse OBB coordinates
                        coords = [float(p) for p in parts[:8]]
                        class_name = parts[8]

                        # Scale OBB coordinates
                        scaled_obb = []
                        for i in range(0, 8, 2):
                            scaled_obb.append(coords[i] * scale_x)
                            scaled_obb.append(coords[i + 1] * scale_y)

                        # Convert OBB to AABB
                        xs = [scaled_obb[0], scaled_obb[2], scaled_obb[4], scaled_obb[6]]
                        ys = [scaled_obb[1], scaled_obb[3], scaled_obb[5], scaled_obb[7]]

                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)

                        # Clip to image bounds
                        x_min = max(0, min(x_min, self.image_size))
                        y_min = max(0, min(y_min, self.image_size))
                        x_max = max(0, min(x_max, self.image_size))
                        y_max = max(0, min(y_max, self.image_size))

                        # Filter degenerate boxes
                        if (x_max - x_min) < 1 or (y_max - y_min) < 1:
                            continue

                        class_id = self.class_to_id.get(class_name, -1)
                        if class_id >= 0:
                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(class_id)
                            obb_list.append(scaled_obb)
                        else:
                            logger.warning(f"Unknown class: {class_name}")

                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse: {line}, Error: {e}")
                        continue

        # Convert to numpy
        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            obb_arr = np.array(obb_list, dtype=np.float32)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            obb_arr = np.zeros((0, 8), dtype=np.float32)

        # Apply augmentations
        if self.augmentation and len(boxes) > 0:
            image, boxes, obb_arr = self._apply_augmentations(image, boxes, obb_arr)

        # Convert image to tensor with ImageNet normalization
        image_np = np.array(image, dtype=np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
        image_np = (image_np - mean) / std
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        # Convert targets to tensors
        boxes_tensor = torch.from_numpy(boxes).float()
        labels_tensor = torch.from_numpy(labels).long()
        obb_tensor = torch.from_numpy(obb_arr).float()

        targets = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'obb': obb_tensor,
            'image_id': idx,
            'filename': image_file,
        }

        return image_tensor, targets

    def _apply_augmentations(
        self, image: Image.Image, boxes: np.ndarray, obb: np.ndarray
    ) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
        """
        Apply data augmentations to image, AABB boxes, and OBB corners.

        Augmentations:
        - Random horizontal flip (50%)
        - Random vertical flip (50%)
        - Random 90° rotation (25% for each of 0°, 90°, 180°, 270°)
        - Random color jitter (brightness, contrast)
        """
        image_np = np.array(image)
        w = image_np.shape[1]
        h = image_np.shape[0]

        # 1. Random horizontal flip
        if np.random.rand() < 0.5:
            image_np = np.fliplr(image_np).copy()
            # Flip AABB
            boxes_flipped = boxes.copy()
            boxes_flipped[:, 0] = w - boxes[:, 2]
            boxes_flipped[:, 2] = w - boxes[:, 0]
            boxes = boxes_flipped
            # Flip OBB x-coordinates
            if obb.shape[0] > 0:
                obb_flipped = obb.copy()
                for i in range(0, 8, 2):
                    obb_flipped[:, i] = w - obb[:, i]
                obb = obb_flipped

        # 2. Random vertical flip
        if np.random.rand() < 0.5:
            image_np = np.flipud(image_np).copy()
            # Flip AABB
            boxes_flipped = boxes.copy()
            boxes_flipped[:, 1] = h - boxes[:, 3]
            boxes_flipped[:, 3] = h - boxes[:, 1]
            boxes = boxes_flipped
            # Flip OBB y-coordinates
            if obb.shape[0] > 0:
                obb_flipped = obb.copy()
                for i in range(1, 8, 2):
                    obb_flipped[:, i] = h - obb[:, i]
                obb = obb_flipped

        # 3. Random 90° rotation (0, 90, 180, 270 degrees)
        k = np.random.randint(0, 4)  # Number of 90° rotations
        if k > 0:
            image_np = np.rot90(image_np, k).copy()
            new_h, new_w = image_np.shape[:2]

            for _ in range(k):
                # Rotate AABB: (x,y) → (y, W-x) for 90° CCW
                new_boxes = boxes.copy()
                new_boxes[:, 0] = boxes[:, 1]       # new x1 = old y1
                new_boxes[:, 1] = w - boxes[:, 2]   # new y1 = W - old x2
                new_boxes[:, 2] = boxes[:, 3]       # new x2 = old y2
                new_boxes[:, 3] = w - boxes[:, 0]   # new y2 = W - old x1
                boxes = new_boxes

                # Rotate OBB
                if obb.shape[0] > 0:
                    new_obb = obb.copy()
                    for i in range(0, 8, 2):
                        new_obb[:, i] = obb[:, i + 1]         # new x = old y
                        new_obb[:, i + 1] = w - obb[:, i]     # new y = W - old x
                    obb = new_obb

                w, h = h, w  # Swap dimensions after rotation

            # Recompute AABB from rotated boxes (ensure x1 < x2, y1 < y2)
            x_mins = np.minimum(boxes[:, 0], boxes[:, 2])
            y_mins = np.minimum(boxes[:, 1], boxes[:, 3])
            x_maxs = np.maximum(boxes[:, 0], boxes[:, 2])
            y_maxs = np.maximum(boxes[:, 1], boxes[:, 3])
            boxes = np.stack([x_mins, y_mins, x_maxs, y_maxs], axis=1)

        # 4. Color jitter
        if np.random.rand() < 0.5:
            # Brightness
            brightness = np.random.uniform(0.7, 1.3)
            image_np = np.clip(image_np * brightness, 0, 255).astype(np.uint8)

        if np.random.rand() < 0.3:
            # Contrast
            gray_mean = image_np.mean()
            contrast = np.random.uniform(0.7, 1.3)
            image_np = np.clip(
                (image_np - gray_mean) * contrast + gray_mean, 0, 255
            ).astype(np.uint8)

        # Clip boxes to image bounds
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.image_size)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.image_size)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.image_size)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.image_size)

        # Remove degenerate boxes after augmentation
        valid = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
        boxes = boxes[valid]
        obb = obb[valid] if obb.shape[0] > 0 else obb

        image = Image.fromarray(image_np)
        return image, boxes, obb

    def collate_fn(self, batch: List[Tuple]) -> Tuple[torch.Tensor, List[Dict]]:
        """Custom collate function for variable-length detections."""
        images = []
        targets_list = []

        for image, targets in batch:
            images.append(image)
            targets_list.append(targets)

        images_batch = torch.stack(images, dim=0)
        return images_batch, targets_list


def get_dota_dataloader(
    data_root: str,
    split: str = 'train',
    image_size: int = 1024,
    batch_size: int = 4,
    num_workers: int = 0,
    subset_size: Optional[int] = None,
    augmentation: bool = False,
    shuffle: bool = True
) -> DataLoader:
    """Create DataLoader for DOTA dataset."""
    dataset = DOTADetectionDataset(
        data_root=data_root,
        split=split,
        image_size=image_size,
        subset_size=subset_size,
        augmentation=augmentation
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn
    )

    return dataloader


if __name__ == "__main__":
    # Quick self-test
    print("DOTA Dataset Module — Self-Test")
    print("=" * 50)

    data_root = "./datasets/processed"

    if os.path.exists(os.path.join(data_root, 'train_hr')):
        ds = DOTADetectionDataset(data_root, 'train', 1024, subset_size=3, augmentation=True)
        img, tgt = ds[0]
        print(f"Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"Boxes: {tgt['boxes'].shape}, Labels: {tgt['labels'].shape}")
        print(f"OBB: {tgt['obb'].shape}")
        if len(tgt['boxes']) > 0:
            print(f"First box: {tgt['boxes'][0]}")
            assert (tgt['boxes'][:, 2] <= 1024).all(), "x2 exceeds image width!"
            assert (tgt['boxes'][:, 3] <= 1024).all(), "y2 exceeds image height!"
            print("✓ Box coordinates are within image bounds")
        print("✓ Dataset self-test passed")
    else:
        print(f"Dataset not found at {data_root}")
        print("Run: python data/prepare_data.py first")
