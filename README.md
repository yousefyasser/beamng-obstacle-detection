# Autonomous Vehicle Obstacle Avoidance System

A real-time obstacle detection and avoidance system for autonomous vehicles using computer vision and deep learning, developed in collaboration with Siemens Innexis VSI and BeamNG.

## Project Overview

This system implements real-time object detection and obstacle avoidance for autonomous vehicles using:
- YOLOv11 for object detection
- BeamNG.tech for vehicle simulation and visualization
- Real-time performance monitoring and optimization
- Advanced image processing pipeline

## System Architecture

### System Context Diagram
![System Context Diagram](docs/System%20context%20diagram.png)

### Container Diagram
![System Context Diagram](docs/container%20diagram.png)


The project is organized into several key components:

### 1. Detection Manager (`detectionManager.py`)
- Implements YOLO object detection
- Handles model initialization and inference
- Processes camera images to detect obstacles
- Optimized for real-time performance

### 2. Image Processing Pipeline (`imageProcessor.py`)
- Modular image processing system
- Implements various effects:
  - Gaussian Noise
  - Gaussian Blur
  - Brightness/Contrast adjustment
  - Rain Effect
  - Color Filtering

### 3. Visualization Manager (`visualizationManager.py`)
- Handles real-time visualization of detections
- Implements bounding box drawing
- Manages text overlays and annotations

### 4. Environment Manager (`environmentManager.py`)
- Manages BeamNG.tech integration
- Handles vehicle control and camera setup
- Manages simulation environment

## Installation

### Prerequisites
- Python 3.9 or higher
- BeamNG.tech license and installation

### Environment Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up BeamNG environment variable:
```bash
# Windows
set BNG_HOME=C:\path\to\beamng
# Linux/MacOS
export BNG_HOME=/path/to/beamng
```

## Usage

1. Run the main script:
```bash
python obstacle_avoidance/main.py
```

2. The system will:
   - Initialize the detection model
   - Set up the visualization
   - Begin processing camera feed
   - Display real-time detections

## Project Structure

```
Bachelor project/
├── obstacle avoidance/
│   ├── main.py
│   ├── DTO/
│   │   ├── detection.py
│   │   ├── visualization_data.py
│   ├── src/
│   │   ├── core/
│   │   │   ├── anomaly_detector.py
│   │   │   ├── detection_manager.py
│   │   │   ├── environment_manager.py
│   │   │   ├── error_propagation_manager.py
│   │   ├── processors/
│   │   │   ├── image_processor.py
│   │   │   ├── base_processor.py
│   │   ├── visualization/
│   │   │   ├── visualization_manager.py
│   │   ├── utils/
│   │   │   ├── logger.py
│   │   ├── config/
│   │   │   ├── config.py
│   │   │   ├── dataclasses.py
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments

- Siemens Innexis VSI
- BeamNG.tech
- Ultralytics (YOLOv11)