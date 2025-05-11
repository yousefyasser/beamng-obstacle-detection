# Image Processing Pipeline Technical Documentation

## Overview
The Image Processing Pipeline is a modular system for real-time image processing and enhancement.

## Architecture

### Class Hierarchy
```python
class ImageProcessor(ABC):
    def process(self, image: Image.Image) -> Image.Image

class GaussianNoiseProcessor(ImageProcessor)
class GaussianBlurProcessor(ImageProcessor)
class BrightnessContrastProcessor(ImageProcessor)
class RainEffectProcessor(ImageProcessor)
class ColorFilterProcessor(ImageProcessor)

class ImageProcessingPipeline:
    def __init__(self, processors: List[ImageProcessor])
    def process(self, image: Image.Image) -> Image.Image
```

### Key Components

1. **Base Processor**
   - Abstract base class for all processors
   - Common interface for image processing

2. **Effect Processors**
   - Gaussian Noise
   - Gaussian Blur
   - Brightness/Contrast
   - Rain Effect
   - Color Filter

3. **Pipeline Manager**
   - Sequential processing

## Technical Details

### Image Processing Effects

1. **Gaussian Noise**
   ```python
   class GaussianNoiseProcessor(ImageProcessor):
       def __init__(self, mean: float = 0, std: float = 25)
   ```
   - Adds random noise to images
   - Configurable mean and standard deviation

2. **Gaussian Blur**
   ```python
   class GaussianBlurProcessor(ImageProcessor):
       def __init__(self, radius: float = 2)
   ```
   - Applies Gaussian blur
   - Configurable radius

3. **Brightness/Contrast**
   ```python
   class BrightnessContrastProcessor(ImageProcessor):
       def __init__(self, brightness_factor: float = 0.7, 
                    contrast_factor: float = 1.2)
   ```
   - Adjusts image brightness and contrast
   - Configurable factors

4. **Rain Effect**
   ```python
   class RainEffectProcessor(ImageProcessor):
       def __init__(self, intensity: float = 0.1)
   ```
   - Simulates rain effect
   - Configurable intensity

5. **Color Filter**
   ```python
   class ColorFilterProcessor(ImageProcessor):
       def __init__(self, color: Tuple[int, int, int] = (0, 0, 255))
   ```
   - Applies color filter
   - Configurable color


## Usage Examples

### Basic Usage
```python
pipeline = ImageProcessingPipeline([
    GaussianBlurProcessor(radius=1),
    BrightnessContrastProcessor()
])
processed_image = pipeline.process(image)
```

### Custom Pipeline
```python
pipeline = ImageProcessingPipeline([
    GaussianNoiseProcessor(std=15),
    GaussianBlurProcessor(radius=2),
    BrightnessContrastProcessor(1.2, 1.1),
    RainEffectProcessor(0.05),
    ColorFilterProcessor((0, 0, 255))
])
```
