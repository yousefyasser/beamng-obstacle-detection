from abc import ABC, abstractmethod
from PIL import ImageEnhance, ImageFilter, Image
from typing import Tuple, List
import numpy as np
from logger import system_logger


class ImageProcessor(ABC):
    """Abstract base class for image processing operations"""
    def __init__(self):
        self.logger = system_logger.image_logger

    @abstractmethod
    def process(self, image: Image.Image) -> Image.Image:
        pass

class GaussianNoiseProcessor(ImageProcessor):
    def __init__(self, mean: float = 0, std: float = 25):
        super().__init__()
        self.mean = mean
        self.std = std

    def process(self, image: Image.Image) -> Image.Image:
        try:
            img_array = np.array(image)
            noise = np.random.normal(self.mean, self.std, img_array.shape)
            noisy_image = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy_image)
        except Exception as e:
            self.logger.error(f"Gaussian noise processing failed: {str(e)}")
            raise

class GaussianBlurProcessor(ImageProcessor):
    def __init__(self, radius: float = 2):
        super().__init__()
        self.radius = radius

    def process(self, image: Image.Image) -> Image.Image:
        try:
            return image.filter(ImageFilter.GaussianBlur(radius=self.radius))
        except Exception as e:
            self.logger.error(f"Gaussian blur processing failed: {str(e)}")
            raise

class BrightnessContrastProcessor(ImageProcessor):
    def __init__(self, brightness_factor: float = 0.7, contrast_factor: float = 1.2):
        super().__init__()
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        
    def process(self, image: Image.Image) -> Image.Image:
        try:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(self.brightness_factor)
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(self.contrast_factor)
        except Exception as e:
            self.logger.error(f"Brightness/contrast processing failed: {str(e)}")
            raise

class RainEffectProcessor(ImageProcessor):
    def __init__(self, intensity: float = 0.1):
        super().__init__()
        self.intensity = intensity

    def process(self, image: Image.Image) -> Image.Image:
        try:
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            for _ in range(int(width * height * self.intensity)):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                img_array[y, x] = [255, 255, 255]
            return Image.fromarray(img_array)
        except Exception as e:
            self.logger.error(f"Rain effect processing failed: {str(e)}")
            raise

class ColorFilterProcessor(ImageProcessor):
    def __init__(self, color: Tuple[int, int, int] = (0, 0, 255)):
        super().__init__()
        self.color = color

    def process(self, image: Image.Image) -> Image.Image:
        try:
            img_array = np.array(image)
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] + self.color[0], 0, 255)
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] + self.color[1], 0, 255)
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] + self.color[2], 0, 255)
            return Image.fromarray(img_array)
        except Exception as e:
            self.logger.error(f"Color filter processing failed: {str(e)}")
            raise

class ImageProcessingPipeline:
    def __init__(self, processors: List[ImageProcessor]):
        self.processors = processors
        self.logger = system_logger.image_logger

    def process(self, image: Image.Image) -> Image.Image:
        try:
            for processor in self.processors:
                image = processor.process(image)
            return image
        except Exception as e:
            self.logger.error(f"Image processing pipeline failed: {str(e)}")
            raise
