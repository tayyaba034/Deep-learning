"""
Advanced Data Augmentation Pipeline for Cat vs Dog Classification
Features:
- Albumentations for advanced augmentations
- Custom mixup and cutmix implementations
- Progressive augmentation strategy
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import albumentations as A


class AdvancedAugmentation:
    """Advanced augmentation pipeline using Albumentations"""
    
    def __init__(self, img_size=224, augmentation_level='medium'):
        """
        Initialize augmentation pipeline
        
        Args:
            img_size: Target image size
            augmentation_level: 'light', 'medium', or 'heavy'
        """
        self.img_size = img_size
        self.augmentation_level = augmentation_level
        
        # Define augmentation levels
        if augmentation_level == 'light':
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
            ])
        elif augmentation_level == 'medium':
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=25, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.OneOf([
                    A.GaussianBlur(blur_limit=3, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ], p=0.3),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                               min_holes=4, fill_value=0, p=0.3),
            ])
        else:  # heavy
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=35, p=0.6),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=35, p=0.6),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
                A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=0.6),
                A.OneOf([
                    A.GaussianBlur(blur_limit=5, p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ], p=0.4),
                A.OneOf([
                    A.OpticalDistortion(distort_limit=0.2, p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1.0),
                ], p=0.3),
                A.CoarseDropout(max_holes=12, max_height=40, max_width=40, 
                               min_holes=6, fill_value=0, p=0.4),
            ])
    
    def __call__(self, image):
        """Apply augmentation to image"""
        if isinstance(image, tf.Tensor):
            image = image.numpy()
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        augmented = self.transform(image=image)
        return augmented['image'].astype(np.float32) / 255.0


class MixupCutmix:
    """Mixup and Cutmix augmentation for better generalization"""
    
    def __init__(self, alpha=0.2, cutmix_prob=0.5):
        """
        Initialize Mixup/Cutmix
        
        Args:
            alpha: Mixup/Cutmix parameter
            cutmix_prob: Probability of using cutmix vs mixup
        """
        self.alpha = alpha
        self.cutmix_prob = cutmix_prob
    
    def mixup(self, images, labels):
        """Apply mixup augmentation"""
        batch_size = tf.shape(images)[0]
        
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Shuffle indices
        indices = tf.random.shuffle(tf.range(batch_size))
        
        # Mix images and labels
        mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
        mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)
        
        return mixed_images, mixed_labels
    
    def cutmix(self, images, labels):
        """Apply cutmix augmentation"""
        batch_size = tf.shape(images)[0]
        image_height = tf.shape(images)[1]
        image_width = tf.shape(images)[2]
        
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Calculate cut region size
        cut_ratio = tf.sqrt(1.0 - lam)
        cut_h = tf.cast(cut_ratio * tf.cast(image_height, tf.float32), tf.int32)
        cut_w = tf.cast(cut_ratio * tf.cast(image_width, tf.float32), tf.int32)
        
        # Random center position
        cy = tf.random.uniform([], 0, image_height, dtype=tf.int32)
        cx = tf.random.uniform([], 0, image_width, dtype=tf.int32)
        
        # Calculate box coordinates
        y1 = tf.clip_by_value(cy - cut_h // 2, 0, image_height)
        y2 = tf.clip_by_value(cy + cut_h // 2, 0, image_height)
        x1 = tf.clip_by_value(cx - cut_w // 2, 0, image_width)
        x2 = tf.clip_by_value(cx + cut_w // 2, 0, image_width)
        
        # Shuffle indices
        indices = tf.random.shuffle(tf.range(batch_size))
        shuffled_images = tf.gather(images, indices)
        shuffled_labels = tf.gather(labels, indices)
        
        # Create mask
        mask = tf.ones_like(images)
        
        # Calculate indices for the cut region
        y_range = tf.range(y1, y2)
        x_range = tf.range(x1, x2)
        yy, xx = tf.meshgrid(y_range, x_range, indexing='ij')
        coords = tf.stack([tf.reshape(yy, [-1]), tf.reshape(xx, [-1])], axis=1)
        
        # Repeat for each image in the batch
        batch_indices = tf.repeat(tf.range(batch_size), tf.shape(coords)[0])
        coords_batch = tf.tile(coords, [batch_size, 1])
        
        full_indices = tf.concat([
            tf.expand_dims(tf.cast(batch_indices, tf.int32), axis=1),
            coords_batch
        ], axis=1)
        
        # updates should match the number of pixels * 3 channels
        updates = tf.zeros([tf.shape(full_indices)[0], 3])
        
        mask = tf.tensor_scatter_nd_update(mask, full_indices, updates)
        
        # Mix images
        mixed_images = images * mask + shuffled_images * (1 - mask)
        
        # Mix labels based on area ratio
        area_ratio = 1.0 - (tf.cast((y2-y1)*(x2-x1), tf.float32) / 
                            tf.cast(image_height*image_width, tf.float32))
        mixed_labels = area_ratio * labels + (1 - area_ratio) * shuffled_labels
        
        return mixed_images, mixed_labels
    
    def __call__(self, images, labels):
        """Apply mixup or cutmix randomly"""
        if np.random.random() < self.cutmix_prob:
            return self.cutmix(images, labels)
        else:
            return self.mixup(images, labels)


def create_data_generator(image_paths, labels, batch_size=32, 
                          augmentation=None, shuffle=True, mixup_cutmix=None):
    """
    Create a data generator with augmentation
    
    Args:
        image_paths: List of image file paths
        labels: Corresponding labels
        batch_size: Batch size
        augmentation: AdvancedAugmentation instance
        shuffle: Whether to shuffle data
        mixup_cutmix: MixupCutmix instance (optional)
    
    Yields:
        (batch_images, batch_labels)
    """
    num_samples = len(image_paths)
    indices = np.arange(num_samples)
    
    while True:
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_images = []
            batch_labels = []
            
            for idx in batch_indices:
                # Load image
                img = cv2.imread(image_paths[idx])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Apply augmentation
                if augmentation:
                    img = augmentation(img)
                else:
                    img = cv2.resize(img, (224, 224)) / 255.0
                
                batch_images.append(img)
                batch_labels.append(labels[idx])
            
            batch_images = np.array(batch_images, dtype=np.float32)
            batch_labels = np.array(batch_labels, dtype=np.float32)
            
            # Apply mixup/cutmix if provided
            if mixup_cutmix is not None:
                batch_images, batch_labels = mixup_cutmix(batch_images, batch_labels)
            
            yield batch_images, batch_labels


if __name__ == "__main__":
    # Test augmentation
    print("Testing augmentation pipeline...")
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Test different augmentation levels
    for level in ['light', 'medium', 'heavy']:
        print(f"\nTesting {level} augmentation...")
        aug = AdvancedAugmentation(img_size=224, augmentation_level=level)
        augmented = aug(dummy_image)
        print(f"Output shape: {augmented.shape}, dtype: {augmented.dtype}")
        print(f"Value range: [{augmented.min():.4f}, {augmented.max():.4f}]")
    
    # Test mixup/cutmix
    print("\nTesting Mixup/Cutmix...")
    mc = MixupCutmix(alpha=0.2)
    batch_images = np.random.random((4, 224, 224, 3)).astype(np.float32)
    batch_labels = np.array([0, 1, 0, 1], dtype=np.float32)
    
    mixed_images, mixed_labels = mc(batch_images, batch_labels)
    print(f"Mixed images shape: {mixed_images.shape}")
    print(f"Mixed labels: {mixed_labels}")
