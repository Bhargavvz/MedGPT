"""
DICOM Processor Module
======================
Handles DICOM to PNG conversion with medical image enhancements.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import SimpleITK as sitk
from loguru import logger


class DICOMProcessor:
    """Processes DICOM medical images with various enhancements."""
    
    def __init__(
        self,
        output_size: Tuple[int, int] = (224, 224),
        apply_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        normalize: bool = True,
        output_format: str = "png"
    ):
        """
        Initialize DICOM processor.
        
        Args:
            output_size: Target image size (width, height)
            apply_clahe: Whether to apply CLAHE enhancement
            clahe_clip_limit: CLAHE clip limit
            clahe_tile_grid_size: CLAHE tile grid size
            normalize: Whether to normalize pixel values
            output_format: Output format (png, jpg)
        """
        self.output_size = output_size
        self.apply_clahe = apply_clahe
        self.normalize = normalize
        self.output_format = output_format
        
        # Initialize CLAHE
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid_size
        )
    
    def read_dicom(self, dicom_path: Union[str, Path]) -> np.ndarray:
        """
        Read DICOM file and convert to numpy array.
        
        Args:
            dicom_path: Path to DICOM file
            
        Returns:
            Numpy array of pixel data
        """
        dicom_path = Path(dicom_path)
        
        try:
            # Try pydicom first
            ds = pydicom.dcmread(str(dicom_path))
            
            # Apply VOI LUT (windowing)
            pixel_array = apply_voi_lut(ds.pixel_array, ds)
            
            # Handle PhotometricInterpretation
            if hasattr(ds, 'PhotometricInterpretation'):
                if ds.PhotometricInterpretation == "MONOCHROME1":
                    # Invert grayscale
                    pixel_array = np.max(pixel_array) - pixel_array
            
            return pixel_array.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"pydicom failed, trying SimpleITK: {e}")
            
            # Fallback to SimpleITK
            image = sitk.ReadImage(str(dicom_path))
            pixel_array = sitk.GetArrayFromImage(image)
            
            # Handle 3D volumes - take middle slice
            if len(pixel_array.shape) == 3:
                middle_idx = pixel_array.shape[0] // 2
                pixel_array = pixel_array[middle_idx]
            
            return pixel_array.astype(np.float32)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to 0-255 range.
        
        Args:
            image: Input image array
            
        Returns:
            Normalized image
        """
        # Handle edge case of constant image
        if image.max() == image.min():
            return np.zeros_like(image, dtype=np.uint8)
        
        # Normalize to 0-255
        normalized = (image - image.min()) / (image.max() - image.min())
        normalized = (normalized * 255).astype(np.uint8)
        
        return normalized
    
    def apply_clahe_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input grayscale image (0-255)
            
        Returns:
            Enhanced image
        """
        if len(image.shape) == 3:
            # Convert to grayscale if needed
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                image = image[:, :, 0]
        
        # Apply CLAHE
        enhanced = self.clahe.apply(image)
        
        return enhanced
    
    def resize_image(
        self,
        image: np.ndarray,
        interpolation: int = cv2.INTER_AREA
    ) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            interpolation: OpenCV interpolation method
            
        Returns:
            Resized image
        """
        return cv2.resize(image, self.output_size, interpolation=interpolation)
    
    def process_dicom(
        self,
        dicom_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """
        Full DICOM processing pipeline.
        
        Args:
            dicom_path: Path to DICOM file
            output_path: Optional path to save processed image
            
        Returns:
            Processed image array
        """
        # Read DICOM
        image = self.read_dicom(dicom_path)
        
        # Normalize to 0-255
        image = self.normalize_image(image)
        
        # Apply CLAHE if enabled
        if self.apply_clahe:
            image = self.apply_clahe_enhancement(image)
        
        # Resize
        image = self.resize_image(image)
        
        # Convert to RGB (3 channels)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.output_format.lower() == "png":
                cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            else:
                Image.fromarray(image).save(str(output_path), quality=95)
        
        return image
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        recursive: bool = True
    ) -> int:
        """
        Process all DICOM files in a directory.
        
        Args:
            input_dir: Input directory containing DICOM files
            output_dir: Output directory for processed images
            recursive: Whether to search recursively
            
        Returns:
            Number of processed files
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find DICOM files
        pattern = "**/*.dcm" if recursive else "*.dcm"
        dicom_files = list(input_dir.glob(pattern))
        
        # Also check for files without extension (common in DICOM)
        for file in input_dir.rglob("*") if recursive else input_dir.glob("*"):
            if file.is_file() and not file.suffix:
                try:
                    pydicom.dcmread(str(file), stop_before_pixels=True)
                    dicom_files.append(file)
                except:
                    pass
        
        processed_count = 0
        
        for dicom_path in dicom_files:
            try:
                # Create relative output path
                rel_path = dicom_path.relative_to(input_dir)
                output_path = output_dir / rel_path.with_suffix(f".{self.output_format}")
                
                self.process_dicom(dicom_path, output_path)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count}/{len(dicom_files)} files")
                    
            except Exception as e:
                logger.error(f"Error processing {dicom_path}: {e}")
        
        logger.info(f"Completed processing {processed_count} DICOM files")
        return processed_count


class ImagePreprocessor:
    """General image preprocessing for non-DICOM medical images."""
    
    def __init__(
        self,
        output_size: Tuple[int, int] = (224, 224),
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        apply_clahe: bool = False
    ):
        """
        Initialize image preprocessor.
        
        Args:
            output_size: Target size
            normalize_mean: Normalization mean (ImageNet default)
            normalize_std: Normalization std (ImageNet default)
            apply_clahe: Whether to apply CLAHE
        """
        self.output_size = output_size
        self.normalize_mean = np.array(normalize_mean)
        self.normalize_std = np.array(normalize_std)
        self.apply_clahe = apply_clahe
        
        if apply_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load image from path."""
        image = Image.open(image_path).convert("RGB")
        return np.array(image)
    
    def preprocess(
        self,
        image: Union[str, Path, np.ndarray],
        return_tensor: bool = False
    ) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Image path or numpy array
            return_tensor: Whether to return as normalized tensor format
            
        Returns:
            Preprocessed image
        """
        # Load if path
        if isinstance(image, (str, Path)):
            image = self.load_image(image)
        
        # Apply CLAHE if enabled
        if self.apply_clahe:
            # Convert to LAB, apply CLAHE to L channel
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Resize
        image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_AREA)
        
        if return_tensor:
            # Normalize for model input
            image = image.astype(np.float32) / 255.0
            image = (image - self.normalize_mean) / self.normalize_std
            # Convert to CHW format
            image = np.transpose(image, (2, 0, 1))
        
        return image


if __name__ == "__main__":
    # Example usage
    processor = DICOMProcessor(output_size=(224, 224), apply_clahe=True)
    
    # Process single file
    # processed = processor.process_dicom("path/to/dicom.dcm", "output.png")
    
    # Process directory
    # processor.process_directory("input_dir", "output_dir")
    
    print("DICOM Processor initialized successfully")
