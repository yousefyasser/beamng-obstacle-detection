from abc import ABC, abstractmethod
from PIL import ImageEnhance, ImageFilter, Image
from typing import Tuple
import numpy as np

class ImageProcessor(ABC):
    """Abstract base class for image processing operations"""
    @abstractmethod
    def process(self, image: Image.Image) -> Image.Image:
        pass

class GaussianNoiseProcessor(ImageProcessor):
    def __init__(self, mean: float = 0, std: float = 25):
        self.mean = mean
        self.std = std

    def process(self, image: Image.Image) -> Image.Image:
        img_array = np.array(image)
        noise = np.random.normal(self.mean, self.std, img_array.shape).astype(np.uint8)
        noisy_image = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)

class GaussianBlurProcessor(ImageProcessor):
    def __init__(self, radius: float = 2):
        self.radius = radius

    def process(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=self.radius))

class BrightnessContrastProcessor(ImageProcessor):
    def __init__(self, brightness_factor: float = 0.7, contrast_factor: float = 1.2):
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor

    def process(self, image: Image.Image) -> Image.Image:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(self.brightness_factor)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(self.contrast_factor)
        return image

class RainEffectProcessor(ImageProcessor):
    def __init__(self, intensity: float = 0.1):
        self.intensity = intensity

    def process(self, image: Image.Image) -> Image.Image:
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        for _ in range(int(width * height * self.intensity)):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            img_array[y, x] = [255, 255, 255]
        return Image.fromarray(img_array)

class ColorFilterProcessor(ImageProcessor):
    def __init__(self, color: Tuple[int, int, int] = (0, 0, 255)):
        self.color = color

    def process(self, image: Image.Image) -> Image.Image:
        img_array = np.array(image)
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] + self.color[0], 0, 255)
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] + self.color[1], 0, 255)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] + self.color[2], 0, 255)
        return Image.fromarray(img_array)
