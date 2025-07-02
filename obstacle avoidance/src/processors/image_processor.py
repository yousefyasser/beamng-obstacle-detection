from abc import ABC, abstractmethod
from PIL import ImageEnhance, ImageFilter, Image
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import cv2
from dataclasses import dataclass
from ..utils.logger import system_logger
from .image_processor import BaseImageProcessor, ProcessingConfig

@dataclass
class ProcessingConfig(ProcessingConfig):
    """Extended configuration for improved image processing"""
    fog_intensity: float = 0.5
    glare_position: Tuple[float, float] = (0.8, 0.2)
    glare_size: float = 0.2
    glare_intensity: float = 0.8

class MetricsTrackingProcessor(BaseImageProcessor):
    """Abstract base class for image processing operations with metrics tracking"""
    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self.last_metrics = {
            "rain_intensity": 0.0,
            "rain_drops": 0,
            "coverage_percent": 0.0,

            "fog_intensity": 0.0,
            "visibility": 1.0,
            "contrast_reduction_percent": 0.0,

            "snr_db": 0.0,
            "noise_magnitude": 0.0,
        }

    @abstractmethod
    def process(self, image: Image.Image) -> Image.Image:
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics about the last processing operation"""
        pass
    
    def get_name(self) -> str:
        """Get the name of this processor"""
        return self.__class__.__name__


class GaussianNoiseProcessor(MetricsTrackingProcessor):
    """Add Gaussian noise to the image with metrics tracking"""
    def process(self, image: Image.Image) -> Image.Image:
        try:
            if not self._validate_image(image):
                raise ValueError("Invalid input image")
                
            img_array = self._to_numpy(image)
            noise = np.random.normal(0, self.config.gaussian_noise_std, img_array.shape)
            
            # Calculate noise magnitude and SNR
            # noise_magnitude = np.mean(np.abs(noise))
            signal_power = np.mean(img_array**2)
            # noise_power = np.mean(noise**2)
            
            # snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

            noisy_image = np.clip(img_array + noise, 0, 255)
            actual_noise = noisy_image.astype(np.float32) - img_array
            actual_noise_power = np.mean(actual_noise**2)

            snr = 10 * np.log10(signal_power / actual_noise_power) if actual_noise_power > 0 else float('inf')
            noise_magnitude = np.mean(np.abs(actual_noise))

            self.logger.info(f"Gaussian noise processing: noise_magnitude={noise_magnitude}, snr={snr}")
            
            # Update metrics
            self._update_metrics({
                "noise_type": "gaussian",
                "std": self.config.gaussian_noise_std,
                "noise_magnitude": noise_magnitude,
                "snr_db": snr
            })
            
            # noisy_image = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            return self._to_pil(noisy_image.astype(np.uint8))
        except Exception as e:
            self.logger.error(f"Gaussian noise processing failed: {str(e)}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "noise_type": "gaussian",
            "std": self.config.gaussian_noise_std,
            "noise_magnitude": self.last_metrics["noise_magnitude"],
            "snr_db": self.last_metrics["snr_db"]
        }


class GaussianBlurProcessor(MetricsTrackingProcessor):
    """Apply Gaussian blur to the image with metrics tracking"""
    def process(self, image: Image.Image) -> Image.Image:
        try:
            if not self._validate_image(image):
                raise ValueError("Invalid input image")
                
            # Calculate sharpness before blur
            img_array = self._to_numpy(image)
            gray_before = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            laplacian_var_before = cv2.Laplacian(gray_before, cv2.CV_64F).var()
            
            # Apply blur
            blurred = image.filter(ImageFilter.GaussianBlur(radius=self.config.gaussian_blur_radius))
            
            # Calculate sharpness after blur
            blurred_array = self._to_numpy(blurred)
            gray_after = cv2.cvtColor(blurred_array, cv2.COLOR_RGB2GRAY)
            laplacian_var_after = cv2.Laplacian(gray_after, cv2.CV_64F).var()
            
            # Calculate sharpness reduction as a percentage
            sharpness_reduction = ((laplacian_var_before - laplacian_var_after) / laplacian_var_before * 100 
                                 if laplacian_var_before > 0 else 0.0)
            
            # Update metrics
            self._update_metrics({
                "blur_type": "gaussian",
                "radius": self.config.gaussian_blur_radius,
                "sharpness_reduction_percent": sharpness_reduction
            })
            
            return blurred
        except Exception as e:
            self.logger.error(f"Gaussian blur processing failed: {str(e)}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "blur_type": "gaussian",
            "radius": self.config.gaussian_blur_radius,
            "sharpness_reduction_percent": self.last_metrics["sharpness_reduction_percent"]
        }


class BrightnessContrastProcessor(MetricsTrackingProcessor):
    """Adjust brightness and contrast of the image with metrics tracking"""
    def process(self, image: Image.Image) -> Image.Image:
        try:
            if not self._validate_image(image):
                raise ValueError("Invalid input image")
                
            # Calculate histogram before changes
            img_array_before = self._to_numpy(image)
            hist_before = cv2.calcHist([img_array_before], [0], None, [256], [0, 256])
            
            # Apply brightness and contrast adjustments
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(self.config.brightness_factor)
            enhancer = ImageEnhance.Contrast(image)
            result = enhancer.enhance(self.config.contrast_factor)
            
            # Calculate histogram after changes
            img_array_after = self._to_numpy(result)
            hist_after = cv2.calcHist([img_array_after], [0], None, [256], [0, 256])
            
            # Calculate histogram difference (using chi-square distance)
            hist_diff = cv2.compareHist(hist_before, hist_after, cv2.HISTCMP_CHISQR)
            mean_brightness = np.mean(img_array_after)
            
            # Update metrics
            self._update_metrics({
                "brightness_factor": self.config.brightness_factor,
                "contrast_factor": self.config.contrast_factor,
                "histogram_change": hist_diff,
                "mean_brightness": mean_brightness
            })
            
            return result
        except Exception as e:
            self.logger.error(f"Brightness/contrast processing failed: {str(e)}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "brightness_factor": self.config.brightness_factor,
            "contrast_factor": self.config.contrast_factor,
            "histogram_change": self.last_metrics["histogram_change"],
            "mean_brightness": self.last_metrics["mean_brightness"]
        }


class RainEffectProcessor(MetricsTrackingProcessor):
    """Add rain effect to the image with metrics tracking"""
    def process(self, image: Image.Image) -> Image.Image:
        try:
            if not self._validate_image(image):
                raise ValueError("Invalid input image")
                
            img_array = self._to_numpy(image)
            height, width = img_array.shape[:2]
            
            # Calculate number of rain drops
            num_drops = int(width * height * self.config.rain_intensity)
            coverage_percent = (num_drops / (width * height)) * 100
            
            # Add raindrops
            for _ in range(num_drops):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                img_array[y, x] = [255, 255, 255]
            
            # Update metrics
            self._update_metrics({
                "rain_intensity": self.config.rain_intensity,
                "rain_drops": num_drops,
                "coverage_percent": coverage_percent
            })
            
            return self._to_pil(img_array)
        except Exception as e:
            self.logger.error(f"Rain effect processing failed: {str(e)}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "rain_intensity": self.config.rain_intensity,
            "rain_drops": self.last_metrics["rain_drops"],
            "coverage_percent": self.last_metrics["coverage_percent"]
        }


class ColorFilterProcessor(MetricsTrackingProcessor):
    """Apply color shift to the image with metrics tracking"""
    def process(self, image: Image.Image) -> Image.Image:
        try:
            if not self._validate_image(image):
                raise ValueError("Invalid input image")
                
            img_array = self._to_numpy(image)
            
            # Calculate color distribution before change
            mean_color_before = np.mean(img_array, axis=(0, 1))
            
            # Apply color shift
            for i in range(3):  # RGB channels
                img_array[:, :, i] = np.clip(img_array[:, :, i] + self.config.color_shift[i], 0, 255)
            
            # Calculate color distribution after change
            mean_color_after = np.mean(img_array, axis=(0, 1))
            color_shift = np.linalg.norm(mean_color_after - mean_color_before)
            
            # Update metrics
            self._update_metrics({
                "color_shift": self.config.color_shift,
                "mean_color_before": mean_color_before.tolist(),
                "mean_color_after": mean_color_after.tolist(),
                "color_shift_magnitude": color_shift
            })
            
            return self._to_pil(img_array)
        except Exception as e:
            self.logger.error(f"Color filter processing failed: {str(e)}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "color_shift": self.config.color_shift,
            "mean_color_before": self.last_metrics["mean_color_before"],
            "mean_color_after": self.last_metrics["mean_color_after"],
            "color_shift_magnitude": self.last_metrics["color_shift_magnitude"]
        }


class FogEffectProcessor(MetricsTrackingProcessor):
    def process(self, image: Image.Image) -> Image.Image:
        try:
            if not self._validate_image(image):
                raise ValueError("Invalid input image")
                
            img_array = self._to_numpy(image)
            
            # Create fog mask
            fog = np.ones_like(img_array) * 255
            fog_intensity = self.config.fog_intensity
            
            # Apply fog effect
            foggy = cv2.addWeighted(img_array, 1 - fog_intensity, fog, fog_intensity, 0)
            
            # Calculate visibility metrics
            visibility = 1 - fog_intensity
            contrast_reduction = fog_intensity * 100
            
            # Update metrics
            self._update_metrics({
                "fog_intensity": fog_intensity,
                "visibility": visibility,
                "contrast_reduction_percent": contrast_reduction
            })
            
            return self._to_pil(foggy)
        except Exception as e:
            self.logger.error(f"Fog effect processing failed: {str(e)}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "fog_intensity": self.last_metrics["fog_intensity"],
            "visibility": self.last_metrics["visibility"],
            "contrast_reduction_percent": self.last_metrics["contrast_reduction_percent"]
        }


class ImageProcessingPipeline:
    """Pipeline for applying multiple image processors in sequence with metrics tracking"""
    def __init__(self, processors: List[MetricsTrackingProcessor]):
        self.processors = processors
        self.logger = system_logger.image_logger
        self.stage_results = []

    def process(self, image: Image.Image) -> Image.Image:
        """Process the image through all processors in sequence"""
        try:
            self.stage_results = []  # Reset stage results
            current_image = image
            
            for processor in self.processors:
                # Process image
                current_image = processor.process(current_image)
                
                # Store stage results
                self.stage_results.append((
                    processor.get_name(),
                    current_image,
                    processor.get_metrics()
                ))
            
            return current_image
        except Exception as e:
            self.logger.error(f"Image processing pipeline failed: {str(e)}")
            raise

    def get_stage_results(self) -> List[Tuple[str, Image.Image, Dict[str, Any]]]:
        """Get results from each processing stage"""
        return self.stage_results 