import cv2
import numpy as np
from typing import List, Optional, Dict, Any
from PIL import Image
import time
from pathlib import Path
from datetime import datetime
import traceback
from ..core.error_propagation_manager import ErrorPropagationManager
from ..utils.logger import system_logger
from ..config.dataclasses import VisualizationConfig, Anomaly

from DTO.detection import Detection

# Colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
        [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


class VisualizationManager:
    """Class responsible for visualizing detections and error propagation"""
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.logger = system_logger.visualization_logger
        self.error_propagation_manager = ErrorPropagationManager()
        self.last_detection = None
        self.prev_control_decision = False
        self.last_vis_save_time = 0
        self.last_image = None
        self.error_vis_failed = False
        self.anomaly_vis_failed = False
        self.current_anomalies = []
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create windows with proper flags
        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(config.window_name, *config.main_window_size)
        cv2.moveWindow(config.window_name, 0, 0)

        cv2.namedWindow(config.error_window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(config.error_window_name, *config.error_window_size)
        cv2.moveWindow(config.error_window_name, 0, 480)

        cv2.namedWindow(config.anomaly_window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(config.anomaly_window_name, *config.anomaly_window_size)
        cv2.moveWindow(config.anomaly_window_name, 640, 0)

    def set_original_image(self, image: Image.Image):
        """Set the original, error-free reference image for comparison"""
        try:
            self.error_propagation_manager.set_reference_image(image)
        except Exception as e:
            self.logger.error(f"Failed to set original image: {str(e)}")
    
    def add_processing_stage(self, name: str, image: Image.Image, additional_data: Optional[dict] = None):
        """Add a processing stage to the error propagation analysis"""
        try:
            self.error_propagation_manager.add_stage(name, image, additional_data)
        except Exception as e:
            self.logger.error(f"Failed to add processing stage: {str(e)}")
            
    def update_anomalies(self, anomalies: List[Anomaly]):
        """Update the current list of detected anomalies"""
        self.current_anomalies = anomalies

    def visualize(self, image: Image.Image, detections: List[Detection], is_car_ahead: bool = False):
        """Visualize detections on the image and optionally show error propagation"""
        try:
            # Store the current image
            self.last_image = image
            
            # Get the main car detection if present
            car_detection = next((d for d in detections if d.class_id == 3), None)
            
            # Set reference detection if not already set
            if self.error_propagation_manager.reference_detection is None and car_detection is not None:
                self.error_propagation_manager.set_reference_detection(car_detection)
            
            # Record if control decision changed from reference
            control_decision_changed = is_car_ahead != self.prev_control_decision
            self.prev_control_decision = is_car_ahead
            
            # Store the last car detection
            self.last_detection = car_detection
            
            # Update error propagation data if we have enough information
            if len(self.error_propagation_manager.stages) > 0:
                self.error_propagation_manager.create_propagation_data(
                    car_detection, control_decision_changed
                )
            
            # Convert PIL Image to OpenCV format properly
            img = np.array(image)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Draw detections
            for detection in detections:
                xmin, ymin, xmax, ymax = map(int, detection.bbox)
                
                # Draw bounding box with thicker line
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), self.config.line_thickness)
                
                # Prepare and draw text with better visibility
                text = f'{self.config.class_names[detection.class_id]}: {detection.confidence:0.2f}'
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, self.config.font_thickness
                )
                
                # Draw text background with padding
                padding = 5
                cv2.rectangle(img, 
                            (xmin, ymin - text_height - 2*padding), 
                            (xmin + text_width + 2*padding, ymin), 
                            (0, 255, 255), -1)
                
                # Draw text with better contrast
                cv2.putText(img, text, 
                          (xmin + padding, ymin - padding), 
                          cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, (0, 0, 0), 
                          self.config.font_thickness)
            
            # Add control decision indicator with better visibility
            decision_text = "BRAKE ACTIVATED" if is_car_ahead else "DRIVING NORMALLY"
            decision_color = (0, 0, 255) if is_car_ahead else (0, 255, 0)
            cv2.putText(img, decision_text, (20, 40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, decision_color, self.config.line_thickness)
            
            # Show main detection window
            cv2.imshow(self.config.window_name, img)
            
            # Show error propagation visualization if enabled
            if not self.error_vis_failed:
                try:
                    error_vis = self.error_propagation_manager.visualize_error_propagation()
                    error_vis = np.array(error_vis)
                    error_vis = cv2.cvtColor(error_vis, cv2.COLOR_RGB2BGR)
                    cv2.imshow(self.config.error_window_name, error_vis)
                except Exception as e:
                    self.logger.error(f"Failed to show error propagation: {str(e)}")
                    self.error_vis_failed = True
            
            # Show anomaly detection visualization if enabled
            if not self.anomaly_vis_failed:
                try:
                    anomaly_vis = self._create_anomaly_visualization()
                    cv2.imshow(self.config.anomaly_window_name, anomaly_vis)
                except Exception as e:
                    self.logger.error(f"Failed to show anomaly detection: {str(e)}")
                    self.anomaly_vis_failed = True
            
            # Save visualizations periodically
            current_time = time.time()
            if current_time - self.last_vis_save_time >= self.config.save_interval:
                self.save_visualizations(img)
                self.last_vis_save_time = current_time
            
            # Handle key events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                raise KeyboardInterrupt
                
        except Exception as e:
            self.logger.error(f"Error in visualization: {str(e)}")
            traceback.print_exc()

    def _create_anomaly_visualization(self) -> np.ndarray:
        """Create visualization for anomaly detection results"""
        try:
            if not self.current_anomalies:
                return np.zeros((768, 1024, 3), dtype=np.uint8)
            
            # Create a blank image
            vis = np.zeros((768, 1024, 3), dtype=np.uint8)
            
            # Add title
            cv2.putText(vis, "Anomaly Detection Results", (20, 40),
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Add each anomaly
            y_offset = 80
            for i, anomaly in enumerate(self.current_anomalies):
                # Format the text with proper value formatting
                text = f"{anomaly.type}: {anomaly.severity} ({anomaly.metric} = {anomaly.value:.2f})"
                
                # Choose color based on severity
                color = (0, 0, 255) if anomaly.severity == 'high' else (0, 255, 255)
                
                # Draw text with background for better visibility
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(vis, 
                            (15, y_offset + i*30 - 20),
                            (25 + text_size[0], y_offset + i*30 + 5),
                            (0, 0, 0), -1)
                cv2.putText(vis, text, (20, y_offset + i*30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            return vis
            
        except Exception as e:
            self.logger.error(f"Error creating anomaly visualization: {str(e)}")
            return np.zeros((768, 1024, 3), dtype=np.uint8)

    def save_visualizations(self, image: np.ndarray):
        """Save current visualizations to disk"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Convert image to RGB if it's RGBA
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # Save main visualization
            main_path = self.output_dir / f"main_{timestamp}.jpg"
            cv2.imwrite(str(main_path), image)
            
            # Save error propagation visualization
            if not self.error_vis_failed:
                try:
                    error_vis = self.error_propagation_manager.visualize_error_propagation()
                    error_vis = np.array(error_vis)
                    if error_vis.shape[2] == 4:  # Convert RGBA to RGB
                        error_vis = cv2.cvtColor(error_vis, cv2.COLOR_RGBA2RGB)
                    error_path = self.output_dir / f"error_{timestamp}.jpg"
                    cv2.imwrite(str(error_path), error_vis)
                except Exception as e:
                    self.logger.error(f"Failed to save error visualization: {str(e)}")
            
            # Save anomaly detection visualization
            if not self.anomaly_vis_failed:
                try:
                    anomaly_vis = self._create_anomaly_visualization()
                    if anomaly_vis.shape[2] == 4:  # Convert RGBA to RGB
                        anomaly_vis = cv2.cvtColor(anomaly_vis, cv2.COLOR_RGBA2RGB)
                    anomaly_path = self.output_dir / f"anomaly_{timestamp}.jpg"
                    cv2.imwrite(str(anomaly_path), anomaly_vis)
                except Exception as e:
                    self.logger.error(f"Failed to save anomaly visualization: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Failed to save visualization: {str(e)}")
            traceback.print_exc()

    def reset_error_propagation(self):
        """Reset error propagation tracking"""
        self.error_propagation_manager.reset()
        self.last_detection = None
        self.prev_control_decision = False

    def cleanup(self):
        """Clean up resources"""
        try:
            cv2.destroyAllWindows()
            self.logger.info("Visualization cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")