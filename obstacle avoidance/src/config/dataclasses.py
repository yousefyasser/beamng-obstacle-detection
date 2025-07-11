from dataclasses import dataclass
from typing import Dict, Tuple, Any

@dataclass
class VehicleConfig:
    """Configuration for vehicle setup"""
    model: str = "scintilla"
    color: str = "Blue"
    license: str = "SENSORS"
    # position: Tuple[float, float, float] = (282.0726420581341, -289.3486400062684, 120.6660895264940)
    # rotation: Tuple[float, float, float, float] = (0.0031202579848468304, -0.0012951375683769584, 0.7027040123939514, 0.7114742994308472)
    position: Tuple[float, float, float] = (192.61466097831726, -856.0309405624866, 267.552119133994)
    rotation: Tuple[float, float, float, float] = (-0.022755950689315796, 0.030111581087112427, 0.984528124332428, 0.1711140275001526)

@dataclass
class CameraConfig:
    """Configuration for camera setup"""
    position: Tuple[float, float, float] = (-0.3, -1.6, 1.2)
    direction: Tuple[float, float, float] = (0, -1, 0)
    fov: float = 70
    resolution: Tuple[int, int] = (640, 480)
    update_time: float = 0.02

@dataclass
class Anomaly:
    """Data class for storing anomaly information"""
    type: str
    severity: str
    metric: str
    value: float
    threshold: float

@dataclass
class ProcessingConfig:
    """Configuration for image processing parameters"""
    gaussian_noise_std: float = 50
    gaussian_blur_radius: float = 2.0
    brightness_factor: float = 0.7
    contrast_factor: float = 1.2
    rain_intensity: float = 0.05
    color_shift: Tuple[int, int, int] = (0, 0, 255)
    fog_intensity: float = 0.5

@dataclass
class AnomalyThresholds:
    noise: Dict[str, float] = None
    blur: Dict[str, float] = None
    brightness: Dict[str, Any] = None
    rain: Dict[str, float] = None
    color: Dict[str, float] = None
    fog: Dict[str, float] = None

    def __post_init__(self):
        if self.noise is None:
            self.noise = {
                'snr_threshold': 15.0,
                'magnitude_threshold': 20.0
            }
        if self.blur is None:
            self.blur = {
                'sharpness_reduction_threshold': 30.0
            }
        if self.brightness is None:
            self.brightness = {
                'histogram_change_threshold': 1000.0,
                'brightness_threshold': 100.0
            }
        if self.rain is None:
            self.rain = {
                'rain_intensity_threshold': 0.05
            }
        if self.color is None:
            self.color = {
                'shift_threshold': 30.0
            }
        if self.fog is None:
            self.fog = {
                'fog_threshold': 0.45
            }

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""
    window_name: str = 'Object Detection'
    error_window_name: str = 'Error Propagation'
    anomaly_window_name: str = 'Anomaly Detection'
    output_dir: str = 'output'
    save_interval: int = 2  # Save visualization every 5 seconds
    main_window_size: tuple[int, int] = (640, 480)
    error_window_size: tuple[int, int] = (640, 480)
    anomaly_window_size: tuple[int, int] = (640, 480)
    font_scale: float = 0.7
    font_thickness: int = 1
    line_thickness: int = 2
    class_names: list[str] = None