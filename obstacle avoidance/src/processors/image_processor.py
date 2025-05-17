from abc import ABC, abstractmethod
from PIL import ImageEnhance, ImageFilter, Image
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from ..utils.logger import system_logger
from ..config.dataclasses import ProcessingConfig


class BaseImageProcessor(ABC):
    """Abstract base class for image processing operations"""
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = system_logger.image_logger
        self._last_metrics: Optional[Dict[str, Any]] = None

    @abstractmethod
    def process(self, image: Image.Image) -> Image.Image:
        """Process the input image and return the processed result"""
        pass

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get metrics from the last processed image"""
        return self._last_metrics

    def _update_metrics(self, metrics: Dict[str, Any]):
        """Update the metrics for the last processed image"""
        self._last_metrics = metrics

    def _to_numpy(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array"""
        return np.array(image)

    def _to_pil(self, array: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image"""
        return Image.fromarray(array)

    def _validate_image(self, image: Image.Image) -> bool:
        """Validate input image"""
        if not isinstance(image, Image.Image):
            self.logger.error("Input must be a PIL Image")
            return False
        return True

class GaussianNoiseProcessor(BaseImageProcessor):
    """Add Gaussian noise to the image"""
    def process(self, image: Image.Image) -> Image.Image:
        try:
            if not self._validate_image(image):
                raise ValueError("Invalid input image")
                
            img_array = self._to_numpy(image)
            noise = np.random.normal(0, self.config.gaussian_noise_std, img_array.shape)
            noisy_image = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            
            # Calculate and store metrics
            self._update_metrics({
                'noise_level': np.std(noise),
                'mean_noise': np.mean(noise)
            })
            
            return self._to_pil(noisy_image)
        except Exception as e:
            self.logger.error(f"Gaussian noise processing failed: {str(e)}")
            raise

class GaussianBlurProcessor(BaseImageProcessor):
    """Apply Gaussian blur to the image"""
    def process(self, image: Image.Image) -> Image.Image:
        try:
            if not self._validate_image(image):
                raise ValueError("Invalid input image")
                
            blurred = image.filter(ImageFilter.GaussianBlur(radius=self.config.gaussian_blur_radius))
            
            # Calculate and store metrics
            self._update_metrics({
                'blur_radius': self.config.gaussian_blur_radius,
                'blur_level': self.config.gaussian_blur_radius / 5.0  # Normalized to [0,1]
            })
            
            return blurred
        except Exception as e:
            self.logger.error(f"Gaussian blur processing failed: {str(e)}")
            raise

class BrightnessContrastProcessor(BaseImageProcessor):
    """Adjust brightness and contrast of the image"""
    def process(self, image: Image.Image) -> Image.Image:
        try:
            if not self._validate_image(image):
                raise ValueError("Invalid input image")
                
            # Adjust brightness
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(self.config.brightness_factor)
            
            # Adjust contrast
            enhancer = ImageEnhance.Contrast(image)
            result = enhancer.enhance(self.config.contrast_factor)
            
            # Calculate and store metrics
            self._update_metrics({
                'brightness_factor': self.config.brightness_factor,
                'contrast_factor': self.config.contrast_factor
            })
            
            return result
        except Exception as e:
            self.logger.error(f"Brightness/contrast processing failed: {str(e)}")
            raise

class RainEffectProcessor(BaseImageProcessor):
    """Add rain effect to the image"""
    def process(self, image: Image.Image) -> Image.Image:
        try:
            if not self._validate_image(image):
                raise ValueError("Invalid input image")
                
            img_array = self._to_numpy(image)
            height, width = img_array.shape[:2]
            
            # Calculate number of raindrops based on intensity
            n_drops = int(width * height * self.config.rain_intensity)
            
            # Add raindrops
            for _ in range(n_drops):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                img_array[y, x] = [255, 255, 255]
            
            # Calculate and store metrics
            self._update_metrics({
                'rain_intensity': self.config.rain_intensity,
                'n_drops': n_drops
            })
            
            return self._to_pil(img_array)
        except Exception as e:
            self.logger.error(f"Rain effect processing failed: {str(e)}")
            raise

class ColorFilterProcessor(BaseImageProcessor):
    """Apply color shift to the image"""
    def process(self, image: Image.Image) -> Image.Image:
        try:
            if not self._validate_image(image):
                raise ValueError("Invalid input image")
                
            img_array = self._to_numpy(image)
            
            # Apply color shift
            for i in range(3):  # RGB channels
                img_array[:, :, i] = np.clip(img_array[:, :, i] + self.config.color_shift[i], 0, 255)
            
            # Calculate and store metrics
            self._update_metrics({
                'color_shift': self.config.color_shift,
                'shift_magnitude': np.linalg.norm(self.config.color_shift)
            })
            
            return self._to_pil(img_array)
        except Exception as e:
            self.logger.error(f"Color filter processing failed: {str(e)}")
            raise

class ImageProcessingPipeline:
    """Pipeline for applying multiple image processors in sequence"""
    def __init__(self, processors: List[BaseImageProcessor]):
        self.processors = processors
        self.logger = system_logger.get_logger()

    def process(self, image: Image.Image) -> Image.Image:
        """Process the image through all processors in sequence"""
        try:
            for processor in self.processors:
                image = processor.process(image)
            return image
        except Exception as e:
            self.logger.error(f"Image processing pipeline failed: {str(e)}")
            raise
