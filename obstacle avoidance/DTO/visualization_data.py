from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image


@dataclass
class ErrorMetrics:
    """Metrics measuring error impact at each stage"""
    noise_level: float  # Signal-to-noise ratio or similar metric
    blur_level: float  # Quantified blur amount
    contrast_deviation: float  # Deviation from ideal contrast
    detection_confidence_impact: float  # How errors affected detection confidence
    position_uncertainty: float  # Uncertainty in object position (pixels)


@dataclass
class PropagationStage:
    """Data for a single stage in the error propagation pipeline"""
    name: str  # Stage name (e.g., "Raw Image", "Noise Added", "Post-Processing")
    image: Optional[Image.Image]  # Image at this stage
    metrics: Optional[ErrorMetrics]  # Error metrics for this stage
    additional_data: Optional[Dict[str, Any]] = None  # Any additional data for this stage


@dataclass
class ErrorPropagationData:
    """Complete error propagation data through the pipeline"""
    stages: List[PropagationStage]  # Ordered list of pipeline stages
    original_detection_confidence: float  # Detection confidence without errors
    final_detection_confidence: float  # Detection confidence with errors
    detection_error: Tuple[float, float, float, float]  # Difference in bounding box coordinates
    control_decision_changed: bool  # Whether errors changed the vehicle control decision 