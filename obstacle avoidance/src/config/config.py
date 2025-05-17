from dataclasses import dataclass
from typing import Dict, Tuple, Any
import json
from pathlib import Path
from src.config.dataclasses import AnomalyThresholds, ProcessingConfig, VisualizationConfig


@dataclass
class Config:
    anomaly_thresholds: AnomalyThresholds = None
    visualization: VisualizationConfig = None
    processing: ProcessingConfig = None

    def __post_init__(self):
        if self.anomaly_thresholds is None:
            self.anomaly_thresholds = AnomalyThresholds()
        if self.visualization is None:
            self.visualization = VisualizationConfig()
        if self.processing is None:
            self.processing = ProcessingConfig()

    @classmethod
    def from_json(cls, file_path: str) -> 'Config':
        """Load configuration from a JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, file_path: str):
        """Save configuration to a JSON file"""
        with open(file_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def create_default(cls) -> 'Config':
        """Create a default configuration"""
        return cls() 