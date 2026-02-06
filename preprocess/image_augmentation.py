"""
Image Augmentation Pipeline
===========================
Medical image augmentation using Albumentations.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


class MedicalImageAugmentation:
    """
    Medical image augmentation pipeline.
    
    Designed specifically for medical images with conservative augmentations
    that preserve diagnostic information.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        is_training: bool = True,
        # Augmentation parameters
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.0,  # Usually not applicable for medical
        rotation_limit: int = 15,
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        gaussian_noise_var_limit: Tuple[float, float] = (10.0, 50.0),
        blur_limit: int = 3,
        elastic_transform_prob: float = 0.0,  # Conservative default
        grid_distortion_prob: float = 0.0,
        # Normalization
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            image_size: Target image size
            is_training: Whether to apply augmentations (disabled for val/test)
            horizontal_flip_prob: Probability of horizontal flip
            vertical_flip_prob: Probability of vertical flip
            rotation_limit: Maximum rotation angle in degrees
            brightness_limit: Maximum brightness adjustment factor
            contrast_limit: Maximum contrast adjustment factor
            gaussian_noise_var_limit: Variance range for Gaussian noise
            blur_limit: Maximum kernel size for blur
            elastic_transform_prob: Probability of elastic transform
            grid_distortion_prob: Probability of grid distortion
            normalize_mean: Normalization mean values
            normalize_std: Normalization std values
        """
        self.image_size = image_size
        self.is_training = is_training
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
        # Build transforms
        self.train_transform = self._build_train_transform(
            horizontal_flip_prob=horizontal_flip_prob,
            vertical_flip_prob=vertical_flip_prob,
            rotation_limit=rotation_limit,
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            gaussian_noise_var_limit=gaussian_noise_var_limit,
            blur_limit=blur_limit,
            elastic_transform_prob=elastic_transform_prob,
            grid_distortion_prob=grid_distortion_prob,
        )
        
        self.val_transform = self._build_val_transform()
    
    def _build_train_transform(
        self,
        horizontal_flip_prob: float,
        vertical_flip_prob: float,
        rotation_limit: int,
        brightness_limit: float,
        contrast_limit: float,
        gaussian_noise_var_limit: Tuple[float, float],
        blur_limit: int,
        elastic_transform_prob: float,
        grid_distortion_prob: float,
    ) -> A.Compose:
        """Build training augmentation pipeline."""
        
        transforms = [
            # Resize first
            A.Resize(self.image_size, self.image_size, interpolation=cv2.INTER_AREA),
            
            # Geometric transforms (conservative for medical)
            A.HorizontalFlip(p=horizontal_flip_prob),
            A.VerticalFlip(p=vertical_flip_prob),
            A.Rotate(
                limit=rotation_limit,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),
            
            # Slight affine transforms
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=0,  # Already handled above
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.3
            ),
            
            # Color/intensity transforms
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=brightness_limit,
                    contrast_limit=contrast_limit,
                    p=1.0
                ),
                A.CLAHE(clip_limit=4.0, p=1.0),
                A.Equalize(p=1.0),
            ], p=0.5),
            
            # Noise augmentation
            A.OneOf([
                A.GaussNoise(
                    var_limit=gaussian_noise_var_limit,
                    p=1.0
                ),
                A.MultiplicativeNoise(
                    multiplier=(0.9, 1.1),
                    p=1.0
                ),
            ], p=0.3),
            
            # Blur
            A.OneOf([
                A.GaussianBlur(blur_limit=blur_limit, p=1.0),
                A.MotionBlur(blur_limit=blur_limit, p=1.0),
            ], p=0.2),
            
            # Elastic/grid distortion (optional for medical)
            A.ElasticTransform(
                alpha=50,
                sigma=10,
                p=elastic_transform_prob
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.1,
                p=grid_distortion_prob
            ),
            
            # Cutout/dropout for regularization
            A.CoarseDropout(
                max_holes=4,
                max_height=self.image_size // 16,
                max_width=self.image_size // 16,
                fill_value=0,
                p=0.2
            ),
            
            # Normalize
            A.Normalize(
                mean=self.normalize_mean,
                std=self.normalize_std,
                max_pixel_value=255.0
            ),
            
            # Convert to tensor
            ToTensorV2(),
        ]
        
        return A.Compose(transforms)
    
    def _build_val_transform(self) -> A.Compose:
        """Build validation/test transform (no augmentation)."""
        
        return A.Compose([
            A.Resize(self.image_size, self.image_size, interpolation=cv2.INTER_AREA),
            A.Normalize(
                mean=self.normalize_mean,
                std=self.normalize_std,
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ])
    
    def __call__(
        self,
        image: Union[np.ndarray, Image.Image],
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Apply augmentation to image.
        
        Args:
            image: Input image (numpy array or PIL Image)
            mask: Optional segmentation mask
            
        Returns:
            Dictionary with 'image' and optionally 'mask' keys
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Select transform based on mode
        transform = self.train_transform if self.is_training else self.val_transform
        
        # Apply transform
        if mask is not None:
            transformed = transform(image=image, mask=mask)
            return {
                'image': transformed['image'],
                'mask': transformed['mask']
            }
        else:
            transformed = transform(image=image)
            return {'image': transformed['image']}
    
    def get_train_transform(self) -> A.Compose:
        """Get training transform."""
        return self.train_transform
    
    def get_val_transform(self) -> A.Compose:
        """Get validation transform."""
        return self.val_transform


class ModalitySpecificAugmentation:
    """
    Modality-specific augmentation strategies.
    
    Different medical imaging modalities require different augmentation approaches.
    """
    
    @staticmethod
    def get_xray_augmentation(image_size: int = 224, is_training: bool = True) -> MedicalImageAugmentation:
        """X-ray specific augmentation (more aggressive)."""
        return MedicalImageAugmentation(
            image_size=image_size,
            is_training=is_training,
            horizontal_flip_prob=0.5,  # Chest X-rays can be flipped
            rotation_limit=10,
            brightness_limit=0.3,
            contrast_limit=0.3,
            gaussian_noise_var_limit=(10, 40),
        )
    
    @staticmethod
    def get_ct_augmentation(image_size: int = 224, is_training: bool = True) -> MedicalImageAugmentation:
        """CT scan specific augmentation."""
        return MedicalImageAugmentation(
            image_size=image_size,
            is_training=is_training,
            horizontal_flip_prob=0.0,  # CT orientation matters
            rotation_limit=5,
            brightness_limit=0.1,
            contrast_limit=0.2,
            gaussian_noise_var_limit=(5, 20),
        )
    
    @staticmethod
    def get_mri_augmentation(image_size: int = 224, is_training: bool = True) -> MedicalImageAugmentation:
        """MRI specific augmentation."""
        return MedicalImageAugmentation(
            image_size=image_size,
            is_training=is_training,
            horizontal_flip_prob=0.0,
            rotation_limit=5,
            brightness_limit=0.15,
            contrast_limit=0.15,
            elastic_transform_prob=0.2,  # MRI can tolerate some elastic transform
        )
    
    @staticmethod
    def get_pathology_augmentation(image_size: int = 224, is_training: bool = True) -> MedicalImageAugmentation:
        """Pathology image augmentation (color augmentation important)."""
        return MedicalImageAugmentation(
            image_size=image_size,
            is_training=is_training,
            horizontal_flip_prob=0.5,
            vertical_flip_prob=0.5,  # Orientation doesn't matter in pathology
            rotation_limit=180,  # Full rotation allowed
            brightness_limit=0.2,
            contrast_limit=0.2,
        )


def get_augmentation_by_modality(
    modality: str,
    image_size: int = 224,
    is_training: bool = True
) -> MedicalImageAugmentation:
    """
    Get modality-specific augmentation.
    
    Args:
        modality: Imaging modality (X-ray, CT, MRI, Pathology, Ultrasound)
        image_size: Target image size
        is_training: Whether to apply training augmentations
        
    Returns:
        Configured augmentation pipeline
    """
    modality = modality.lower().replace("-", "").replace(" ", "")
    
    augmentation_map = {
        "xray": ModalitySpecificAugmentation.get_xray_augmentation,
        "ct": ModalitySpecificAugmentation.get_ct_augmentation,
        "mri": ModalitySpecificAugmentation.get_mri_augmentation,
        "pathology": ModalitySpecificAugmentation.get_pathology_augmentation,
    }
    
    augmentation_fn = augmentation_map.get(
        modality,
        lambda s, t: MedicalImageAugmentation(image_size=s, is_training=t)
    )
    
    return augmentation_fn(image_size, is_training)


if __name__ == "__main__":
    # Example usage
    import torch
    
    # Create augmentation pipeline
    aug = MedicalImageAugmentation(image_size=224, is_training=True)
    
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    result = aug(dummy_image)
    print(f"Augmented image shape: {result['image'].shape}")
    print(f"Image dtype: {result['image'].dtype}")
    
    # Test modality-specific
    xray_aug = get_augmentation_by_modality("X-ray", is_training=True)
    result_xray = xray_aug(dummy_image)
    print(f"X-ray augmented shape: {result_xray['image'].shape}")
