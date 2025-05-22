import torch
from PIL import Image
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass
from ..utils.logger import system_logger
from ultralytics import YOLO

@dataclass
class Detection:
    """Data class for storing detection information"""
    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax format

class DetectionManager:
    """Class responsible for object detection using YOLOv11n model"""
    # Common obstacle class IDs in COCO/YOLOv11n
    OBSTACLE_CLASS_IDS = {2, 5, 7}  # car(2), bus(5), truck(7) in YOLOv11n
    
    def __init__(self, confidence_threshold: float = 0.5, device: str = 'auto'):
        """
        Initialize the detection manager
        
        Args:
            confidence_threshold: Minimum confidence score to keep detections
            device: Device to run the model on ('cuda', 'cpu', or 'auto')
        """
        self.confidence_threshold = confidence_threshold
        self.logger = system_logger.detection_logger
        self.device = device
        self.setup_model()
    
    def setup_model(self):
        """Initialize the YOLOv11n model"""
        try:
            # Load YOLOv11n model
            self.model = YOLO("yolo11m.pt")
            
            # Set the device if not auto
            if self.device != 'auto':
                self.model.to(self.device)
            
            # Get actual device being used
            self.actual_device = next(self.model.parameters()).device.type
            self.logger.info(f"YOLOv11n model loaded successfully on {self.actual_device}")
        
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            raise
    
    def detect(self, image: Image.Image) -> List[Detection]:
        """
        Perform object detection on the input image
        
        Args:
            image: PIL image to detect objects in
            
        Returns:
            List of Detection objects
        """
        try:
            # YOLOv11n processes images directly, no need for custom transforms
            # Run inference with the YOLOv11n model
            results = self.model(image, conf=self.confidence_threshold)

            # Extract detections
            detections = []
            
            # Process results - YOLOv11n results format is different from DETR
            for result in results:
                # Get boxes (in xyxy format) and confidence scores
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Get box coordinates and convert to tuple
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    
                    # Get class ID
                    class_id = int(boxes.cls[i].item())
                    
                    # Get confidence score
                    confidence = boxes.conf[i].item()

                    self.logger.info(f"Detection: {class_id}, Confidence: {confidence}, BBox: {x1, y1, x2, y2}")
                    
                    # Create Detection object
                    detections.append(Detection(
                        class_id=class_id,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2)
                    ))
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection failed: {str(e)}")
            return []  # Return empty list instead of raising exception
    
    def is_obstacle_detected(self, detections: List[Detection], target_classes: Optional[Set[int]] = None) -> bool:
        """
        Check if any obstacle is detected in the image
        
        Args:
            detections: List of Detection objects
            target_classes: Optional set of class IDs to consider as obstacles
                           (default: self.OBSTACLE_CLASS_IDS)
                           
        Returns:
            True if any obstacle is detected, False otherwise
        """
        if not detections:
            return False
            
        classes_to_check = target_classes or self.OBSTACLE_CLASS_IDS
        return any(detection.class_id in classes_to_check for detection in detections)
    
    def is_car_detected(self, detections: List[Detection]) -> bool:
        """
        Check if a car is detected in the image
        
        Args:
            detections: List of Detection objects
            
        Returns:
            True if a car is detected, False otherwise
        """
        for detection in detections:
            if detection.class_id == 2:  # 2 is the class ID for car in YOLOv11n
                return True
        return False
    
    def visualize(self, image: Image.Image, detections: List[Detection], show_labels: bool = True) -> Image.Image:
        """
        Visualize detections on the image
        
        Args:
            image: PIL image to visualize detections on
            detections: List of Detection objects
            show_labels: Whether to show class labels and confidence scores
            
        Returns:
            PIL image with detections visualized
        """
        # Use the built-in visualization method from YOLOv11n
        # Create a results object compatible with YOLOv11n plotter
        results = self.model(image, conf=0)  # Empty prediction
        
        # Get the first result
        result = results[0]
        
        # Replace the boxes with our detections
        if detections:
            # Convert our detections to YOLOv11n format
            boxes_xyxy = torch.tensor([d.bbox for d in detections])
            cls = torch.tensor([d.class_id for d in detections])
            conf = torch.tensor([d.confidence for d in detections])
            
            # Replace the boxes
            result.boxes.xyxy = boxes_xyxy
            result.boxes.cls = cls
            result.boxes.conf = conf
            
            # Plot the detections
            return result.plot()
        
        return image
    
    def detect_and_visualize(self, image: Image.Image) -> Tuple[List[Detection], Image.Image]:
        """
        Detect objects and visualize them on the image
        
        Args:
            image: PIL image to detect objects in
            
        Returns:
            Tuple of (detections, image with detections visualized)
        """
        detections = self.detect(image)
        visualized_image = self.visualize(image, detections)
        return detections, visualized_image
    
    def get_class_names(self) -> List[str]:
        """
        Get the names of the classes that the model can detect
        
        Returns:
            List of class names
        """
        return self.model.names
