# Detection Manager Technical Documentation

## Overview
The Detection Manager is responsible for implementing real-time object detection using the DETR (DEtection TRansformer) model. It handles model initialization, inference, and post-processing of detection results.

## Architecture

### Class Structure
```python
class DetectionManager:
    def __init__(self, confidence_threshold: float = 0.3)
    def setup_model(self)
    def detect(self, image: Image.Image) -> List[Detection]
    def rescale_bboxes(self, out_bbox, size)
    def box_cxcywh_to_xyxy(self, x)
```

### Key Components

1. **Model Initialization**
   - Loads DETR model with ResNet50 backbone
   - Sets up image transformation pipeline

2. **Detection Pipeline**
   - Image preprocessing
   - Model inference
   - Post-processing of detections
   - Confidence thresholding
   - Bounding box rescaling

## Technical Details

### Model Configuration
- Model: DETR (DEtection TRansformer)
- Backbone: ResNet50
- Input size: 800x800
- Confidence threshold: 0.3 (configurable)

### Image Processing
```python
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

## Usage Examples

### Basic Usage
```python
detector = DetectionManager(confidence_threshold=0.3)
detections = detector.detect(image)
```
