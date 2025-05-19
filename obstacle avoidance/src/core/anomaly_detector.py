from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from PIL import Image
import cv2
from src.utils.logger import system_logger
from src.config.dataclasses import Anomaly, AnomalyThresholds


class AnomalyDetector:
    """Class responsible for detecting anomalies in processed images"""
    def __init__(self, thresholds: AnomalyThresholds):
        self.logger = system_logger.detection_logger
        self.thresholds = thresholds

    def detect(self, image: Image.Image) -> List[Anomaly]:
        """Detect anomalies in the processed image"""
        try:
            anomalies = []
            img_array = np.array(image)
            
            # Calculate image metrics
            metrics = self._calculate_image_metrics(img_array)
            
            # Check each processor's thresholds
            for proc_name, proc_thresholds in self.thresholds.__dict__.items():
                if proc_name != "logger":
                    anomalies.extend(self._check_processor_anomalies(proc_name, proc_thresholds, metrics))

            return anomalies
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {str(e)}")
            return []

    def _calculate_image_metrics(self, img_array: np.ndarray) -> Dict[str, float]:
        """Calculate various metrics from the image"""
        metrics = {}
        
        try:
            # Convert to grayscale for some metrics
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Calculate noise metrics
            noise = cv2.fastNlMeansDenoising(gray)
            noise_diff = cv2.absdiff(gray, noise)
            snr = 10 * np.log10(np.mean(gray**2) / (np.mean(noise_diff**2) + 1e-10))
            metrics['snr'] = float(snr)
            
            # Calculate blur metrics
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            metrics['sharpness'] = float(laplacian_var)
            
            # Calculate brightness and contrast metrics
            mean_brightness = float(np.mean(gray))  # Ensure single float value
            std_brightness = float(np.std(gray))
            metrics['brightness'] = mean_brightness
            metrics['contrast'] = std_brightness
            
            # Calculate color metrics
            if len(img_array.shape) == 3:
                color_means = np.mean(img_array, axis=(0, 1))
                color_std = np.std(img_array, axis=(0, 1))
                metrics['color_shift'] = float(np.linalg.norm(color_std))
            
            # Calculate fog metrics
            if len(img_array.shape) == 3:
                # Enhanced fog detection using multiple metrics
                # 1. Gradient analysis
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
                gradient_mean = float(np.mean(gradient_magnitude))
                
                # 2. Contrast reduction
                contrast_reduction = 1.0 - (std_brightness / 128.0)  # Normalized contrast
                
                # 3. Brightness increase
                brightness_increase = max(0, (mean_brightness - 128.0) / 128.0)
                
                # Combine metrics for fog detection
                fog_estimate = (1.0 - (gradient_mean / 255.0)) * 0.4 + \
                             contrast_reduction * 0.3 + \
                             brightness_increase * 0.3
                
                metrics['fog_intensity'] = float(fog_estimate)

                # Calculate rain metrics
                rain_intensity = np.sum(img_array == [255, 255, 255]) / (img_array.shape[0] * img_array.shape[1])
                metrics['rain_intensity'] = float(rain_intensity)
                
        except Exception as e:
            self.logger.error(f"Error calculating image metrics: {str(e)}")
        
        return metrics

    def _check_processor_anomalies(self, processor_name: str, thresholds: Dict[str, float], metrics: Dict[str, float]) -> List[Anomaly]:
        """Check for anomalies specific to a processor"""
        anomalies = []
        
        try:
            if processor_name == "noise":
                if "snr_threshold" in thresholds and "snr" in metrics:
                    anomalies.append(Anomaly(
                        type="noise",
                        severity="high" if metrics["snr"] < thresholds["snr_threshold"] else "medium",
                        metric="SNR",
                        value=metrics["snr"],
                        threshold=thresholds["snr_threshold"]
                    ))
            
            elif processor_name == "blur":
                if "sharpness_threshold" in thresholds and "sharpness" in metrics:
                    anomalies.append(Anomaly(
                        type="blur",
                        severity="high" if metrics["sharpness"] < thresholds["sharpness_threshold"] else "medium",
                        metric="Sharpness",
                        value=metrics["sharpness"],
                        threshold=thresholds["sharpness_threshold"]
                    ))
            
            elif processor_name == "brightness":
                if "brightness_threshold" in thresholds and "brightness" in metrics:
                    # Convert to and ensure single value comparison
                    brightness_value = metrics["brightness"]
                    brightness_threshold = thresholds["brightness_threshold"]
                    deviation = abs(brightness_value - 128.0)  # Deviation from middle gray
                    
                    anomalies.append(Anomaly(
                        type="brightness",
                        severity="high" if deviation > brightness_threshold else "medium",
                        metric="Brightness",
                        value=brightness_value,
                        threshold=brightness_threshold
                    ))
            
            elif processor_name == "color":
                if "color_shift_threshold" in thresholds and "color_shift" in metrics:
                    anomalies.append(Anomaly(
                        type="color",
                        severity="high" if metrics["color_shift"] > thresholds["color_shift_threshold"] else "medium",
                        metric="Color Shift",
                        value=metrics["color_shift"],
                        threshold=thresholds["color_shift_threshold"]
                    ))
            
            elif processor_name == "fog":
                if "fog_threshold" in thresholds and "fog_intensity" in metrics:
                    anomalies.append(Anomaly(
                        type="fog",
                        severity="high" if metrics["fog_intensity"] > thresholds["fog_threshold"] else "medium",
                        metric="Fog Intensity",
                        value=metrics["fog_intensity"],
                        threshold=thresholds["fog_threshold"]
                    ))

            elif processor_name == "rain":
                if "rain_intensity_threshold" in thresholds and "rain_intensity" in metrics:
                    anomalies.append(Anomaly(
                        type="rain",
                        severity="high" if metrics["rain_intensity"] > thresholds["rain_intensity_threshold"] else "medium",
                        metric="Rain Intensity",
                        value=metrics["rain_intensity"],
                        threshold=thresholds["rain_intensity_threshold"]
                    ))
            
            
        except Exception as e:
            self.logger.error(f"Error checking {processor_name} anomalies: {str(e)}")
        
        return anomalies

class AnomalyMitigator:
    """Class responsible for mitigating detected anomalies"""
    def __init__(self):
        self.logger = system_logger.detection_logger

    def mitigate(self, image: Image.Image, anomalies: List[Anomaly]) -> Image.Image:
        """Apply mitigation strategies for detected anomalies"""
        try:
            mitigated_image = image.copy()
            
            for anomaly in anomalies:
                if anomaly.type == "noise":
                    # Apply noise reduction
                    mitigated_image = self._reduce_noise(mitigated_image)
                elif anomaly.type == "blur":
                    # Apply sharpening
                    mitigated_image = self._sharpen_image(mitigated_image)
                elif anomaly.type == "brightness":
                    # Adjust brightness
                    mitigated_image = self._adjust_brightness(mitigated_image)
                elif anomaly.type == "rain":
                    # Remove rain effect
                    mitigated_image = self._remove_rain(mitigated_image)
                elif anomaly.type == "color":
                    # Correct color
                    mitigated_image = self._correct_color(mitigated_image)
                elif anomaly.type == "fog":
                    # Remove fog
                    mitigated_image = self._remove_fog(mitigated_image)
                elif anomaly.type == "glare":
                    # Remove glare
                    mitigated_image = self._remove_glare(mitigated_image)
            
            return mitigated_image
        except Exception as e:
            self.logger.error(f"Anomaly mitigation failed: {str(e)}")
            return image

    def _reduce_noise(self, image: Image.Image) -> Image.Image:
        """Reduce noise in the image"""
        # Implement noise reduction
        return image

    def _sharpen_image(self, image: Image.Image) -> Image.Image:
        """Sharpen the image"""
        # Implement image sharpening
        return image

    def _adjust_brightness(self, image: Image.Image) -> Image.Image:
        """Adjust image brightness"""
        # Implement brightness adjustment
        return image

    def _remove_rain(self, image: Image.Image) -> Image.Image:
        """Remove rain effect from the image"""
        # Implement rain removal
        return image

    def _correct_color(self, image: Image.Image) -> Image.Image:
        """Correct color in the image"""
        # Implement color correction
        return image

    def _remove_fog(self, image: Image.Image) -> Image.Image:
        """Remove fog effect from the image"""
        # Implement fog removal
        return image

    def _remove_glare(self, image: Image.Image) -> Image.Image:
        """Remove glare effect from the image"""
        # Implement glare removal
        return image 