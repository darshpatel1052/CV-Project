"""
DOTA Dataset Loader for Object Detection
==========================================

Loads DOTA v1.0/v1.5 dataset with Object Detection annotations.
Supports both high-resolution (1024×1024) and low-resolution (128×128) versions.

DOTA Dataset Structure (v1.0 uses TIFF, v1.5 uses PNG):
    DOTA_v1.0/ or raw-DOTA_v1.5/
    ├── train/                          # Training set
    │   ├── images/                     # Images (TIFF for v1.0, PNG for v1.5)
    │   │   ├── P0000.tif (v1.0)
    │   │   ├── P0000.png (v1.5)
    │   │   └── ...
    │   └── labelTxt/                   # Annotations (custom format)
    │       ├── P0000.txt
    │       └── ...
    └── val/                            # Validation set
        ├── images/
        └── labelTxt/

Annotation Format (labelTxt):
    Header lines:
        imagesource:GoogleEarth
        gsd:0.146343590398
    
    Each object line: x1 y1 x2 y2 x3 y3 x4 y4 class difficulty
    - 8 coordinates form an OBB (Oriented Bounding Box)
    - class: plane, ship, storage-tank, ..., swimming-pool
    - difficulty: 0 (easy), 1 (hard)
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


class DOTADetectionDataset(Dataset):
    """
    DOTA Object Detection Dataset (v1.0/v1.5).
    
    Args:
        data_root: Path to DOTA_v1.0 or raw-DOTA_v1.5 directory
        split: 'train' or 'val'
        image_size: Target image size (1024 for teacher, 128 for student)
        subset_size: If not None, use only first N images (for testing)
        augmentation: Boolean to apply augmentations
    
    Returns:
        image: torch.Tensor of shape (3, H, W), values in [0, 1]
        targets: Dict with keys:
            - 'boxes': torch.Tensor of shape (num_objects, 4) in [x1, y1, x2, y2]
            - 'labels': torch.Tensor of shape (num_objects,) with class IDs
            - 'image_id': int, image index
            - 'filename': str, image filename
    
    Note: Automatically handles both v1.0 (TIFF) and v1.5 (PNG) formats.
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
        
        logger.info(f"Loaded {len(self.image_files)} images from {split} split")
        if subset_size is not None:
            logger.info(f"  (using subset of {subset_size})")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get single sample.
        
        Args:
            idx: Image index
        
        Returns:
            image: torch.Tensor of shape (3, H, W)
            targets: Dict with 'boxes', 'labels', 'image_id', 'filename'
        """
        # Load image
        image_file = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_file)
        
        # Read image (PIL for flexibility with various formats)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise
        
        # Resize to target size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
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
                        
                        # Normalize to [0, 1] based on original image (before resize)
                        # Note: We'll scale when we resize the image
                        boxes.append([x_min, y_min, x_max, y_max])
                        
                        # Get class ID
                        class_id = self.class_to_id.get(class_name, -1)
                        if class_id >= 0:
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
        if self.augmentation:
            image, boxes = self._apply_augmentations(image, boxes)
        
        # Convert image to tensor
        image_np = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # (3, H, W)
        
        # Convert boxes to tensor
        boxes_tensor = torch.from_numpy(boxes)
        labels_tensor = torch.from_numpy(labels)
        
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
            boxes: Bounding boxes in [x1, y1, x2, y2] format
        
        Returns:
            augmented_image: PIL Image
            augmented_boxes: Augmented boxes
        """
        image_np = np.array(image)
        
        # Random horizontal flip
        if np.random.rand() < 0.5:
            image_np = np.fliplr(image_np)
            boxes[:, [0, 2]] = image_np.shape[1] - boxes[:, [2, 0]]
        
        # Random color jitter
        if np.random.rand() < 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            image_np = np.clip(image_np * brightness, 0, 255).astype(np.uint8)
        
        image = Image.fromarray(image_np)
        return image, boxes
    
    def collate_fn(self, batch: List[Tuple]) -> Tuple[torch.Tensor, Dict]:
        """
        Custom collate function for DataLoader.
        
        Since images might have varied number of objects,
        we return batched images and a list of target dicts.
        
        Args:
            batch: List of (image, targets) tuples
        
        Returns:
            images: torch.Tensor of shape (B, 3, H, W)
            targets_list: List of target dicts
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
    """
    Create DataLoader for DOTA dataset.
    
    Args:
        data_root: Path to DOTA_v1.0 directory
        split: 'train' or 'val'
        image_size: Target image size
        batch_size: Batch size
        num_workers: Number of data loading workers
        subset_size: Limit dataset size (for testing)
        augmentation: Apply augmentations
        shuffle: Shuffle dataset
    
    Returns:
        DataLoader instance
    """
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


# ============================================================================
# TESTING & UTILITIES
# ============================================================================

def visualize_dataset_sample(
    dataloader: DataLoader,
    num_samples: int = 3,
    output_dir: Optional[str] = None
):
    """
    Visualize dataset samples with bounding boxes.
    
    Args:
        dataloader: DataLoader instance
        num_samples: Number of samples to visualize
        output_dir: If provided, save visualizations to this directory
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    for batch_idx, (images, targets_list) in enumerate(dataloader):
        for sample_idx in range(min(len(images), num_samples)):
            image = images[sample_idx]  # (3, H, W), [0, 1]
            targets = targets_list[sample_idx]
            
            # Convert to numpy for visualization
            image_np = image.numpy().transpose(1, 2, 0)  # (H, W, 3)
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(image_np)
            
            # Draw bounding boxes
            boxes = targets['boxes']
            labels = targets['labels']
            
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box.numpy()
                w = x2 - x1
                h = y2 - y1
                
                rect = patches.Rectangle(
                    (x1, y1), w, h,
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add class label
                class_name = DOTADetectionDataset.CLASSES[label]
                ax.text(x1, y1-5, class_name, fontsize=10, color='red')
            
            ax.set_title(f"Image {targets['filename']} ({len(boxes)} objects)")
            ax.axis('off')
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f"sample_{batch_idx}_{sample_idx}.png")
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                print(f"Saved to {save_path}")
            else:
                plt.show()
            
            plt.close()


if __name__ == "__main__":
    # Example usage and testing
    print("DOTA Dataset Module")
    print("=" * 50)
    
    # Note: Set these to your actual paths
    data_root = "./datasets/raw/DOTA_v1.0"
    
    if os.path.exists(data_root):
        print(f"\nLoading dataset from: {data_root}")
        
        # Create dataloader
        dataloader = get_dota_dataloader(
            data_root=data_root,
            split='train',
            image_size=1024,
            batch_size=2,
            subset_size=5,
            augmentation=False
        )
        
        print(f"Dataset size: {len(dataloader.dataset)}")
        
        # Get one batch
        images, targets_list = next(iter(dataloader))
        print(f"Batch shape: {images.shape}")
        print(f"Number of targets: {len(targets_list)}")
        
        for i, targets in enumerate(targets_list):
            print(f"  Sample {i}: {targets['boxes'].shape[0]} objects")
        
        # Visualize
        print("\nVisualizing samples...")
        visualize_dataset_sample(dataloader, num_samples=2)
    else:
        print(f"ERROR: Dataset not found at {data_root}")
        print("\nTo set up the dataset:")
        print("1. Download DOTA v1.0 from http://captain.whu.edu.cn/DiRS")
        print("2. Extract to ./datasets/raw/DOTA_v1.0")
        print("3. Expected structure:")
        print("   DOTA_v1.0/")
        print("   ├── train/")
        print("   │   ├── images/")
        print("   │   └── labelTxt/")
        print("   └── val/")
        print("       ├── images/")
        print("       └── labelTxt/")
