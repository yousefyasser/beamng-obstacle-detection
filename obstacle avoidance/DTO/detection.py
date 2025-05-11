from dataclasses import dataclass
from typing import Tuple


@dataclass
class Detection:
    """Data class to hold detection information"""
    class_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax)