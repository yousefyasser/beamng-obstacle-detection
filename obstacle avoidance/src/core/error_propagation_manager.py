import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Any
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import io
import copy
from scipy.ndimage import gaussian_laplace
from dataclasses import dataclass
from ..utils.logger import system_logger

from DTO.detection import Detection
from DTO.visualization_data import ErrorMetrics, PropagationStage, ErrorPropagationData

@dataclass
class ErrorMetrics:
    """Metrics for error analysis at each stage"""
    noise_level: float = 0.0
    blur_level: float = 0.0
    contrast_deviation: float = 0.0
    detection_confidence_impact: float = 0.0
    position_uncertainty: float = 0.0

@dataclass
class PropagationStage:
    """Data for a single stage in the error propagation pipeline"""
    name: str
    image: Image.Image
    metrics: ErrorMetrics
    additional_data: Optional[Dict[str, Any]] = None

@dataclass
class ErrorPropagationData:
    """Complete data for error propagation analysis"""
    stages: List[PropagationStage]
    original_detection_confidence: float
    final_detection_confidence: float
    detection_error: Tuple[float, float, float, float]
    control_decision_changed: bool

class ErrorPropagationManager:
    """Manages error propagation analysis and visualization throughout the detection pipeline"""
    
    def __init__(self):
        self.logger = system_logger.visualization_logger
        self.reference_image = None
        self.reference_detection = None
        self.stages = []
        self.current_propagation_data = None
        self.current_anomalies = []
        plt.style.use('dark_background')  # Use dark mode for better visualization
    
    def set_reference_image(self, image: Image.Image):
        """Set the error-free reference image"""
        self.reference_image = copy.deepcopy(image)
    
    def set_reference_detection(self, detection: Optional[Detection]):
        """Set the reference detection result (from error-free image)"""
        if detection:
            self.reference_detection = copy.deepcopy(detection)
    
    def add_stage(self, name: str, image: Image.Image, additional_data: Optional[Dict] = None):
        """Add a new stage in the error propagation pipeline"""
        metrics = self._calculate_metrics(image)
        stage = PropagationStage(
            name=name,
            image=copy.deepcopy(image),
            metrics=metrics,
            additional_data=additional_data
        )
        self.stages.append(stage)
        self.logger.debug(f"Added stage: {name} with metrics: {metrics}")
        
    def update_anomalies(self, anomalies: List[Dict[str, Any]]):
        """Update the current list of detected anomalies"""
        self.current_anomalies = anomalies
        
    def _calculate_metrics(self, image: Image.Image) -> ErrorMetrics:
        """Calculate error metrics for the current stage compared to reference"""
        # Default values
        metrics = ErrorMetrics()
        
        if self.reference_image is None:
            return metrics
            
        # Convert images to numpy arrays for analysis
        ref_arr = np.array(self.reference_image).astype(np.float32)
        cur_arr = np.array(image).astype(np.float32)
        
        try:
            # Calculate noise level (using SNR)
            ref_power = np.mean(ref_arr**2)
            noise_power = np.mean((ref_arr - cur_arr)**2)
            if noise_power > 0:
                metrics.noise_level = 10 * np.log10(ref_power / noise_power)
            
            # Calculate blur level using Laplacian variance
            ref_gray = cv2.cvtColor(ref_arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            cur_gray = cv2.cvtColor(cur_arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            ref_lap_var = cv2.Laplacian(ref_gray, cv2.CV_64F).var()
            cur_lap_var = cv2.Laplacian(cur_gray, cv2.CV_64F).var()
            
            if ref_lap_var > 0:
                metrics.blur_level = (ref_lap_var - cur_lap_var) / ref_lap_var
                metrics.blur_level = max(0.0, min(1.0, metrics.blur_level))  # Normalize to [0,1]
            
            # Calculate contrast deviation
            ref_contrast = np.std(ref_gray)
            cur_contrast = np.std(cur_gray)
            
            if ref_contrast > 0:
                metrics.contrast_deviation = abs(ref_contrast - cur_contrast) / ref_contrast
            
            # Position uncertainty is a placeholder for now
            # In a real implementation, this would be calculated from detection confidence
            metrics.position_uncertainty = max(5.0 * metrics.blur_level, 3.0 * metrics.noise_level / 10)
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            
        return metrics
    
    def create_propagation_data(self, current_detection: Optional[Detection], 
                               control_decision_changed: bool) -> ErrorPropagationData:
        """Create complete error propagation data based on collected stages"""
        orig_conf = 0.0 if self.reference_detection is None else self.reference_detection.confidence
        final_conf = 0.0 if current_detection is None else current_detection.confidence
        
        # Calculate detection error (difference in bounding boxes)
        if self.reference_detection is not None and current_detection is not None:
            ref_bbox = self.reference_detection.bbox
            cur_bbox = current_detection.bbox
            detection_error = (
                cur_bbox[0] - ref_bbox[0],  # x_min difference
                cur_bbox[1] - ref_bbox[1],  # y_min difference
                cur_bbox[2] - ref_bbox[2],  # x_max difference
                cur_bbox[3] - ref_bbox[3]   # y_max difference
            )
        else:
            detection_error = (0.0, 0.0, 0.0, 0.0)
            
        # Update detection confidence impact for each stage
        for stage in self.stages:
            if stage.metrics:
                impact = abs(orig_conf - final_conf)
                stage.metrics.detection_confidence_impact = impact
        
        # Create and store propagation data
        self.current_propagation_data = ErrorPropagationData(
            stages=self.stages,
            original_detection_confidence=orig_conf,
            final_detection_confidence=final_conf,
            detection_error=detection_error,
            control_decision_changed=control_decision_changed
        )
        
        return self.current_propagation_data
    
    def visualize_error_propagation(self) -> Image.Image:
        """Create a visualization of error propagation through the pipeline"""
        if not self.stages or not self.current_propagation_data:
            self.logger.warning("No data available for visualization")
            return Image.new('RGB', (800, 600), color='black')
            
        try:
            # Create figure with subplots for each stage
            n_stages = len(self.stages)
            fig, axes = plt.subplots(2, n_stages, figsize=(n_stages * 4, 8), 
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    dpi=100)
            
            # If there's only one stage, axes won't be a 2D array
            if n_stages == 1:
                axes = np.array([[axes[0]], [axes[1]]])
                
            # Plot each stage
            for i, stage in enumerate(self.stages):
                # Show image
                axes[0, i].imshow(stage.image)
                axes[0, i].set_title(stage.name, fontsize=12, pad=10)
                axes[0, i].axis('off')
                
                # Add metrics visualization below each image
                if stage.metrics:
                    metrics = [
                        ('Noise', stage.metrics.noise_level / 30),  # Normalized to [0,1]
                        ('Blur', stage.metrics.blur_level),
                        ('Contrast Δ', stage.metrics.contrast_deviation),
                        ('Det. Impact', stage.metrics.detection_confidence_impact),
                        ('Pos. Uncert.', stage.metrics.position_uncertainty / 20)  # Normalized to [0,1]
                    ]
                    
                    # Create bar chart of metrics
                    labels, values = zip(*metrics)
                    values = [min(max(v, 0), 1) for v in values]  # Clip to [0,1]
                    
                    bars = axes[1, i].bar(range(len(labels)), values, color='skyblue')
                    axes[1, i].set_ylim(0, 1)
                    axes[1, i].set_title('Error Metrics', fontsize=10, pad=10)
                    
                    # Set ticks and labels
                    axes[1, i].set_xticks(range(len(labels)))
                    axes[1, i].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                    axes[1, i].tick_params(axis='y', labelsize=8)
                    
                    # Add value labels on bars
                    for bar, val, (_, orig_val) in zip(bars, values, metrics):
                        height = bar.get_height()
                        axes[1, i].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                                       f'{orig_val:.2f}', ha='center', va='bottom', fontsize=8)
            
            # Add overall impact information
            fig.suptitle(f'Error Propagation Analysis\n'
                        f'Detection confidence: {self.current_propagation_data.original_detection_confidence:.2f} → '
                        f'{self.current_propagation_data.final_detection_confidence:.2f} '
                        f'({"Changed" if self.current_propagation_data.control_decision_changed else "Unchanged"} control decision)',
                        fontsize=14, y=0.95)
            
            # Add anomaly information if available
            if self.current_anomalies:
                anomaly_text = "\nDetected Anomalies:\n"
                for anomaly in self.current_anomalies:
                    anomaly_text += f"- {anomaly['type']} ({anomaly['severity']}): {anomaly['metric']} = {anomaly['value']:.2f}\n"
                fig.text(0.02, 0.02, anomaly_text, fontsize=10, 
                        color='red' if any(a['severity'] == 'high' for a in self.current_anomalies) else 'orange',
                        bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=5))
            
            plt.tight_layout()
            
            # Convert matplotlib figure to PIL Image with higher quality
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)
            return img
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            return Image.new('RGB', (800, 600), color='black')
    
    def reset(self):
        """Reset the error propagation tracking"""
        self.reference_image = None
        self.reference_detection = None
        self.stages = []
        self.current_propagation_data = None
        self.current_anomalies = []
        self.logger.debug("Error propagation tracking reset")
    
    def save_visualization(self, filepath: str):
        """Save the current visualization to a file"""
        try:
            if self.current_propagation_data:
                viz = self.visualize_error_propagation()
                viz.save(filepath)
                self.logger.info(f"Saved error propagation visualization to {filepath}")
            else:
                self.logger.warning("No visualization data available to save")
        except Exception as e:
            self.logger.error(f"Failed to save visualization: {str(e)}") 