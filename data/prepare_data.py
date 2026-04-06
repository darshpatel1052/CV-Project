"""
DOTA Data Preparation
======================

Processes raw DOTA dataset to create:
1. High-Resolution (HR) patches: 1024×1024 (for teacher)
2. Low-Resolution (LR) patches: 128×128 (for student, 8× downsampled)

Workflow:
    Raw DOTA images (variable size, typically 4096×4096)
        ↓
    [Load image & annotations]
        ↓
    [Extract 1024×1024 patches with objects]
        ↓
    ├─ Save as HR image
    └─ Downsample 8× → 128×128 → Save as LR image
        ↓
    [Save annotations in both resolutions]
        ↓
    Processed dataset structure:
        datasets/processed/
        ├── train_hr/
        │   ├── images/     (1024×1024)
        │   └── labels/     (annotations)
        ├── train_lr/
        │   ├── images/     (128×128)
        │   └── labels/     (annotations)
        └── splits/
            ├── train.txt
            └── val.txt
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


class DOTADataPreprocessor:
    """
    Preprocess DOTA dataset to create HR/LR image pairs.
    """
    
    DOTA_CLASSES = [
        'plane', 'ship', 'storage-tank', 'baseball-diamond',
        'tennis-court', 'basketball-court', 'ground-track-field',
        'harbor', 'bridge', 'large-vehicle', 'small-vehicle',
        'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool',
        'container-crane'
    ]
    
    def __init__(
        self,
        raw_data_root: str,
        output_root: str,
        hr_size: int = 1024,
        lr_size: int = 128,
        min_object_size: int = 16,  # Minimum pixels for an object
        overlap_ratio: float = 0.2
    ):
        """
        Initialize preprocessor.
        
        Args:
            raw_data_root: Path to raw DOTA_v1.0 or raw-DOTA_v1.5 directory
            output_root: Path to save processed data
            hr_size: Size of HR patches (1024)
            lr_size: Size of LR patches (128)
            min_object_size: Minimum object size to keep (pixels)
            overlap_ratio: Overlap between patches for coverage
        """
        self.raw_data_root = raw_data_root
        self.output_root = output_root
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.downsample_factor = hr_size // lr_size  # Should be 8
        self.min_object_size = min_object_size
        self.overlap_ratio = overlap_ratio
        
        # Create output directories
        self._create_output_dirs()
    
    def _create_output_dirs(self):
        """Create output directory structure."""
        for split in ['train', 'val']:
            for quality in ['hr', 'lr']:
                images_dir = os.path.join(self.output_root, f'{split}_{quality}', 'images')
                labels_dir = os.path.join(self.output_root, f'{split}_{quality}', 'labels')
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(labels_dir, exist_ok=True)
        
        splits_dir = os.path.join(self.output_root, 'splits')
        os.makedirs(splits_dir, exist_ok=True)
    
    def process_dataset(self):
        """Process entire DOTA dataset (train, val, and test if available)."""
        for split in ['train', 'val', 'test']:
            print(f"\n{'='*60}")
            print(f"Processing {split} split")
            print(f"{'='*60}")
            
            self._process_split(split)
    
    def _process_split(self, split: str):
        """
        Process one split (train, val, or test).
        
        Args:
            split: 'train', 'val', or 'test'
        """
        # Check if raw data exists
        images_dir = os.path.join(self.raw_data_root, split, 'images')
        if not os.path.exists(images_dir):
            logger.warning(f"Directory not found: {images_dir} (skipping {split})")
            return
        
        # Check if already processed
        output_dir_check = os.path.join(self.output_root, f'{split}_hr', 'images')
        if os.path.exists(output_dir_check) and len(os.listdir(output_dir_check)) > 0:
            num_patches = len(os.listdir(output_dir_check))
            logger.info(f"✓ Skipping {split}: already processed ({num_patches} HR patches exist)")
            return
        
        labels_dir = os.path.join(self.raw_data_root, split, 'labelTxt')
        
        # Get list of images
        image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.tif', '.tiff', '.jpg', '.png'))
        ])
        
        logger.info(f"Found {len(image_files)} images in {split} split")
        
        processed_count = 0
        total_patches = 0
        total_objects = 0
        split_patches = []
        
        # Process each image
        for image_file in tqdm(image_files, desc=f"Processing {split}"):
            image_path = os.path.join(images_dir, image_file)
            base_name = os.path.splitext(image_file)[0]
            label_path = os.path.join(labels_dir, f"{base_name}.txt")
            
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Failed to read image: {image_path}")
                    continue
                
                image_h, image_w = image.shape[:2]
                
                # Load annotations
                objects = self._load_annotations(label_path)
                
                if len(objects) == 0:
                    logger.warning(f"No objects in {base_name}")
                    continue
                
                # Extract patches
                patches = self._extract_patches(image, objects, image_h, image_w)
                
                # Save patches
                for patch_idx, patch_data in enumerate(patches):
                    patch_name = f"{base_name}_patch{patch_idx:03d}"
                    
                    # Save HR patch
                    hr_image = patch_data['hr_image']
                    hr_objects = patch_data['objects']
                    
                    self._save_patch(
                        image=hr_image,
                        objects=hr_objects,
                        patch_name=patch_name,
                        split=split,
                        quality='hr',
                        resolution=self.hr_size
                    )
                    
                    # Create LR version by downsampling
                    lr_image = cv2.resize(
                        hr_image,
                        (self.lr_size, self.lr_size),
                        interpolation=cv2.INTER_AREA
                    )
                    
                    # Scale box coordinates and annotations
                    scale_factor = self.lr_size / self.hr_size
                    lr_objects = self._scale_objects(hr_objects, scale_factor)
                    
                    self._save_patch(
                        image=lr_image,
                        objects=lr_objects,
                        patch_name=patch_name,
                        split=split,
                        quality='lr',
                        resolution=self.lr_size
                    )
                    
                    total_patches += 1
                    split_patches.append(patch_name)
                    total_objects += len(hr_objects)
                
                processed_count += 1
            
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
                continue
        
        # Save split file
        split_file = os.path.join(self.output_root, "splits", f"{split}.txt")
        with open(split_file, "w", encoding="utf-8") as f:
            f.write("\n".join(split_patches))
        logger.info(f"Saved {len(split_patches)} patch names to {split_file}")

        logger.info(f"Processed {processed_count} images")
        logger.info(f"Generated {total_patches} HR/LR patch pairs")
        logger.info(f"Total objects: {total_objects}")
    
    def _load_annotations(self, label_path: str) -> List[Dict]:
        """
        Load annotations from DOTA label file.
        
        Args:
            label_path: Path to label file
        
        Returns:
            List of object dicts with keys:
                - 'obb': [x1, y1, x2, y2, x3, y3, x4, y4] (oriented bbox)
                - 'aabb': [x_min, y_min, x_max, y_max] (axis-aligned bbox)
                - 'class': class name
                - 'difficulty': 0 or 1
        """
        objects = []
        
        if not os.path.exists(label_path):
            return objects
        
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip metadata lines (imagesource and gsd in v1.5)
                if not line or line.startswith('imagesource') or line.startswith('gsd:'):
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) < 9:
                        continue
                    
                    # Parse OBB coordinates
                    obb = [float(p) for p in parts[:8]]
                    class_name = parts[8]
                    difficulty = int(parts[9]) if len(parts) > 9 else 0
                    
                    # Compute axis-aligned bounding box
                    xs = [obb[i] for i in range(0, 8, 2)]
                    ys = [obb[i] for i in range(1, 8, 2)]
                    
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    
                    aabb = [x_min, y_min, x_max, y_max]
                    
                    objects.append({
                        'obb': obb,
                        'aabb': aabb,
                        'class': class_name,
                        'difficulty': difficulty
                    })
                
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse line: {line}")
                    continue
        
        return objects
    
    def _extract_patches(
        self,
        image: np.ndarray,
        objects: List[Dict],
        image_h: int,
        image_w: int
    ) -> List[Dict]:
        """
        Extract 1024×1024 patches from large image.
        
        Strategy: Use sliding window with overlap, keep patches that contain objects.
        
        Args:
            image: Input image
            objects: List of object annotations
            image_h, image_w: Image dimensions
        
        Returns:
            List of patch dicts with 'hr_image' and 'objects'
        """
        patches = []
        
        # Calculate stride for sliding window
        stride = int(self.hr_size * (1 - self.overlap_ratio))
        
        # Generate patch coordinates
        y_positions = list(range(0, max(image_h - self.hr_size, 1), stride))
        x_positions = list(range(0, max(image_w - self.hr_size, 1), stride))
        
        # Ensure we cover the entire image
        if y_positions[-1] + self.hr_size < image_h:
            y_positions.append(image_h - self.hr_size)
        if x_positions[-1] + self.hr_size < image_w:
            x_positions.append(image_w - self.hr_size)
        
        for y_start in y_positions:
            for x_start in x_positions:
                y_end = min(y_start + self.hr_size, image_h)
                x_end = min(x_start + self.hr_size, image_w)
                
                # Adjust to exact size
                y_start = max(0, y_end - self.hr_size)
                x_start = max(0, x_end - self.hr_size)
                
                # Extract patch
                patch = image[y_start:y_end, x_start:x_end]
                
                if patch.shape[:2] != (self.hr_size, self.hr_size):
                    # Pad if necessary
                    pad_h = self.hr_size - patch.shape[0]
                    pad_w = self.hr_size - patch.shape[1]
                    patch = cv2.copyMakeBorder(
                        patch, 0, pad_h, 0, pad_w,
                        cv2.BORDER_CONSTANT, value=0
                    )
                
                # Find objects in this patch
                patch_objects = []
                for obj in objects:
                    aabb = obj['aabb']
                    x_min, y_min, x_max, y_max = aabb
                    
                    # Check overlap with patch
                    overlap_x = max(0, min(x_max, x_end) - max(x_min, x_start))
                    overlap_y = max(0, min(y_max, y_end) - max(y_min, y_start))
                    
                    if overlap_x > self.min_object_size and overlap_y > self.min_object_size:
                        # Translate coordinates to patch space
                        obj_patch = obj.copy()
                        
                        # Update AABB
                        obj_patch['aabb'] = [
                            max(0, x_min - x_start),
                            max(0, y_min - y_start),
                            min(self.hr_size, x_max - x_start),
                            min(self.hr_size, y_max - y_start)
                        ]
                        
                        # Update OBB (translate all coordinates)
                        obb_translated = []
                        for i in range(0, 8, 2):
                            obb_translated.append(obj['obb'][i] - x_start)
                            obb_translated.append(obj['obb'][i+1] - y_start)
                        obj_patch['obb'] = obb_translated
                        
                        patch_objects.append(obj_patch)
                
                # Only keep patches with objects
                if len(patch_objects) > 0:
                    patches.append({
                        'hr_image': patch,
                        'objects': patch_objects
                    })
        
        return patches
    
    def _scale_objects(self, objects: List[Dict], scale_factor: float) -> List[Dict]:
        """
        Scale object coordinates by factor.
        
        Args:
            objects: List of object dicts
            scale_factor: Scale factor (typically 128/1024 = 1/8)
        
        Returns:
            Scaled objects
        """
        scaled_objects = []
        
        for obj in objects:
            obj_scaled = obj.copy()
            
            # Scale AABB
            aabb = obj['aabb']
            obj_scaled['aabb'] = [
                aabb[0] * scale_factor,
                aabb[1] * scale_factor,
                aabb[2] * scale_factor,
                aabb[3] * scale_factor
            ]
            
            # Scale OBB
            obb = obj['obb']
            obj_scaled['obb'] = [coord * scale_factor for coord in obb]
            
            scaled_objects.append(obj_scaled)
        
        return scaled_objects
    
    def _save_patch(
        self,
        image: np.ndarray,
        objects: List[Dict],
        patch_name: str,
        split: str,
        quality: str,
        resolution: int
    ):
        """
        Save patch image and annotations.
        
        Args:
            image: Image array
            objects: List of objects
            patch_name: Base name for patch
            split: 'train' or 'val'
            quality: 'hr' or 'lr'
            resolution: Image resolution
        """
        output_dir_base = os.path.join(self.output_root, f'{split}_{quality}')
        
        # Save image
        image_path = os.path.join(output_dir_base, 'images', f"{patch_name}.jpg")
        cv2.imwrite(image_path, image)
        
        # Save labels in DOTA format
        label_path = os.path.join(output_dir_base, 'labels', f"{patch_name}.txt")
        
        with open(label_path, 'w') as f:
            for obj in objects:
                # Write OBB coordinates + class + difficulty
                obb_str = ' '.join([f"{coord:.1f}" for coord in obj['obb']])
                class_name = obj['class']
                difficulty = obj.get('difficulty', 0)
                
                f.write(f"{obb_str} {class_name} {difficulty}\n")



# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration - automatically detect v1.0 or v1.5
    # Use absolute paths relative to project root (one level up from this script)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_root_v15 = os.path.join(project_root, "datasets", "raw-DOTA_v1.5")
    raw_data_root_v10 = os.path.join(project_root, "datasets", "raw", "DOTA_v1.0")
    output_root = os.path.join(project_root, "datasets", "processed")
    
    print("DOTA Data Preparation")
    print("=" * 60)
    
    # Check which version is available
    if os.path.exists(raw_data_root_v15):
        raw_data_root = raw_data_root_v15
        version = "v1.5"
    elif os.path.exists(raw_data_root_v10):
        raw_data_root = raw_data_root_v10
        version = "v1.0"
    else:
        print(f"ERROR: Raw dataset not found")
        print(f"\nLooking in: {raw_data_root_v15}")
        print(f"Or:         {raw_data_root_v10}")
        print("\nDownload DOTA from: http://captain.whu.edu.cn/DiRS")
        print("Expected structure (v1.5):")
        print("  raw-DOTA_v1.5/  (PNG images)")
        print("  ├── train/")
        print("  │   ├── images/")
        print("  │   └── labelTxt/")
        print("  └── val/")
        print("      ├── images/")
        print("      └── labelTxt/")
        sys.exit(1)
    
    print(f"Found DOTA {version} at: {raw_data_root}")
    print(f"Output will be saved to: {output_root}\n")
    
    # Initialize preprocessor
    preprocessor = DOTADataPreprocessor(
        raw_data_root=raw_data_root,
        output_root=output_root,
        hr_size=1024,
        lr_size=128
    )
    
    # Process dataset
    preprocessor.process_dataset()
    
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Output saved to: {output_root}")
