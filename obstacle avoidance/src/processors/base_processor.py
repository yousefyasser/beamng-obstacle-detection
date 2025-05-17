from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np
from ..utils.logger import system_logger

class BaseImageProcessor(ABC):
    """Base class for all image processors"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = system_logger.image_logger
        self._metrics: Dict[str, Any] = {}
    
    @abstractmethod
    def process(self, image: Image.Image) -> Image.Image:
        """Process the input image and return the processed image"""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the metrics for the last processed image"""
        return self._metrics.copy()
    
    def _update_metrics(self, metrics: Dict[str, Any]):
        """Update the metrics for the last processed image"""
        self._metrics = metrics
    
    def _convert_to_numpy(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array with proper type handling"""
        img_array = np.array(image)
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        return img_array
    
    def _convert_to_pil(self, img_array: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image with proper type handling"""
        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def _log_error(self, error: Exception, context: str = ""):
        """Log an error with context"""
        self.logger.error(f"Error in {self.name} {context}: {str(error)}")
    
    def _validate_image(self, image: Image.Image) -> bool:
        """Validate input image"""
        if not isinstance(image, Image.Image):
            self.logger.error(f"Invalid image type: {type(image)}")
            return False
        if image.mode not in ['RGB', 'L']:
            self.logger.error(f"Unsupported image mode: {image.mode}")
            return False
        return True
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        return self.__str__() 