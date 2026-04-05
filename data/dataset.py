"""
DOTA Dataset Loader for Object Detection
==========================================

Loads DOTA v1.0/v1.5 dataset with Object Detection annotations.
Supports both high-resolution (1024×1024) and low-resolution (128×128) versions.

Key fixes applied:
  - Bounding boxes are properly scaled to match target image resolution
  - ImageNet normalization applied (pretrained backbones expect it)
  - Degenerate boxes (zero area) are filtered out
  - Boxes are clipped to image bounds
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
            - 'boxes': torch.Tensor of shape (num_objects, 4) in [x1, y1, x2, y2] (pixel coords in target resolution)
            - 'labels': torch.Tensor of shape (num_objects,) with class IDs (0-indexed)
            - 'image_id': int, image index
            - 'filename': str, image filename
    """

    # DOTA class names (15 classes)
    CLASSES = [
        'plane',
        'ship',
        'storage-tank',
        'baseball-diamond',
        'tennis-court',
        'basketball-court',
        'ground-track-field',
        'harbor',
        'bridge',
        'large-vehicle',
        'small-vehicle',
        'helicopter',
        'roundabout',
        'soccer-ball-field',
        'swimming-pool'
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

        # Validate split
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")

        # Create class name to ID mapping
        self.class_to_id = {name: idx for idx, name in enumerate(self.CLASSES)}
        self.id_to_class = {idx: name for idx, name in enumerate(self.CLASSES)}

        # Determine resolution suffix (HR for >=512px, LR for <512px)
        resolution_suffix = 'hr' if image_size >= 512 else 'lr'
        split_dir = f"{split}_{resolution_suffix}"

        # Build file list
        self.images_dir = os.path.join(data_root, split_dir, 'images')
        self.labels_dir = os.path.join(data_root, split_dir, 'labels')

        # Get list of image files
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith(('.tif', '.tiff', '.jpg', '.png'))
        ])

        # Apply subset if specified
        if subset_size is not None:
            self.image_files = self.image_files[:subset_size]

        logger.info(f"Loaded {len(self.image_files)} images from {split}_{resolution_suffix} split")
        if subset_size is not None:
            logger.info(f"  (using subset of {subset_size})")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get single sample.

        Returns:
            image: torch.Tensor of shape (3, H, W), ImageNet-normalized
            targets: Dict with 'boxes', 'labels', 'image_id', 'filename'
        """
        # Load image
        image_file = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_file)

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise

        # Get original image dimensions BEFORE resize
        orig_w, orig_h = image.size  # PIL returns (width, height)

        # Resize to target size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Compute scale factors for box coordinate transformation
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h

        # Load annotations
        base_name = os.path.splitext(image_file)[0]
        label_path = os.path.join(self.labels_dir, f"{base_name}.txt")

        boxes = []  # [x1, y1, x2, y2] (axis-aligned, after transformation)
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip metadata lines (v1.5 includes these)
                    if not line or line.startswith('imagesource') or line.startswith('gsd:'):
                        continue

                    parts = line.split()
                    if len(parts) < 9:
                        continue

                    try:
                        # Parse OBB coordinates (8 values: x1,y1,x2,y2,x3,y3,x4,y4)
                        coords = [float(p) for p in parts[:8]]
                        class_name = parts[8]
                        # difficulty = int(parts[9]) if len(parts) > 9 else 0

                        # Convert OBB to axis-aligned bounding box
                        xs = [coords[0], coords[2], coords[4], coords[6]]
                        ys = [coords[1], coords[3], coords[5], coords[7]]

                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)

                        # Scale coordinates to target resolution
                        x_min = x_min * scale_x
                        y_min = y_min * scale_y
                        x_max = x_max * scale_x
                        y_max = y_max * scale_y

                        # Clip to image bounds
                        x_min = max(0, min(x_min, self.image_size))
                        y_min = max(0, min(y_min, self.image_size))
                        x_max = max(0, min(x_max, self.image_size))
                        y_max = max(0, min(y_max, self.image_size))

                        # Filter degenerate boxes (zero or near-zero area)
                        if (x_max - x_min) < 1 or (y_max - y_min) < 1:
                            continue

                        # Get class ID
                        class_id = self.class_to_id.get(class_name, -1)
                        if class_id >= 0:
                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(class_id)
                        else:
                            logger.warning(f"Unknown class: {class_name}")

                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse annotation line: {line}, Error: {e}")
                        continue

        # Convert to numpy for coordinate transformation
        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        # Apply augmentations if requested
        if self.augmentation and len(boxes) > 0:
            image, boxes = self._apply_augmentations(image, boxes)

        # Convert image to tensor with ImageNet normalization
        image_np = np.array(image, dtype=np.float32) / 255.0  # Scale to [0, 1]

        # Apply ImageNet normalization (required for pretrained backbones)
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
        image_np = (image_np - mean) / std

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # (3, H, W)

        # Convert boxes to tensor
        boxes_tensor = torch.from_numpy(boxes).float()
        labels_tensor = torch.from_numpy(labels).long()

        # Create targets dict
        targets = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': idx,
            'filename': image_file,
        }

        return image_tensor, targets

    def _apply_augmentations(self, image: Image.Image, boxes: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        """
        Apply data augmentations to image and boxes.

        Args:
            image: PIL Image
            boxes: Bounding boxes in [x1, y1, x2, y2] format, already in target resolution

        Returns:
            augmented_image: PIL Image
            augmented_boxes: Augmented boxes
        """
        image_np = np.array(image)

        # Random horizontal flip
        if np.random.rand() < 0.5:
            image_np = np.fliplr(image_np).copy()
            w = image_np.shape[1]
            boxes_flipped = boxes.copy()
            boxes_flipped[:, 0] = w - boxes[:, 2]
            boxes_flipped[:, 2] = w - boxes[:, 0]
            boxes = boxes_flipped

        # Random color jitter (brightness)
        if np.random.rand() < 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            image_np = np.clip(image_np * brightness, 0, 255).astype(np.uint8)

        image = Image.fromarray(image_np)
        return image, boxes

    def collate_fn(self, batch: List[Tuple]) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Custom collate function for DataLoader.

        Since images might have varied number of objects,
        we return batched images and a list of target dicts.
        """
        images = []
        targets_list = []

        for image, targets in batch:
            images.append(image)
            targets_list.append(targets)

        # Stack images into batch
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
        ds = DOTADetectionDataset(data_root, 'train', 1024, subset_size=3)
        img, tgt = ds[0]
        print(f"Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"Boxes: {tgt['boxes'].shape}, Labels: {tgt['labels'].shape}")
        if len(tgt['boxes']) > 0:
            print(f"First box: {tgt['boxes'][0]}")
            assert (tgt['boxes'][:, 2] <= 1024).all(), "x2 exceeds image width!"
            assert (tgt['boxes'][:, 3] <= 1024).all(), "y2 exceeds image height!"
            print("✓ Box coordinates are within image bounds")
        print("✓ Dataset self-test passed")
    else:
        print(f"Dataset not found at {data_root}")
