from PIL import Image
from time import sleep

from detectionManager import DetectionManager
from environmentManager import EnvironmentManager
from visualizationManager import VisualizationManager
from imageProcessor import GaussianNoiseProcessor, GaussianBlurProcessor, BrightnessContrastProcessor, RainEffectProcessor, ColorFilterProcessor


class ObstacleAvoidanceSystem:
    """Main class that coordinates all components"""
    def __init__(self, host: str, port: int):
        self.environment = EnvironmentManager(host, port)
        self.detector = DetectionManager()
        self.visualizer = VisualizationManager()
        
        # Initialize image processors
        self.image_processors = [
            GaussianNoiseProcessor(),
            GaussianBlurProcessor(radius=1),
            BrightnessContrastProcessor(),
            RainEffectProcessor(intensity=0.01),
            ColorFilterProcessor()
        ]

    def process_image(self, image: Image.Image) -> Image.Image:
        for processor in self.image_processors:
            image = processor.process(image)
        return image

    def run(self):
        try:
            while True:
                image = self.environment.get_camera_image()
                if image is not None:
                    # Process image
                    processed_image = self.process_image(image)
                    
                    # Detect objects
                    detections = self.detector.detect(processed_image)
                    
                    # Visualize results
                    self.visualizer.visualize(processed_image, detections)
                
                sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        self.visualizer.cleanup()
        self.environment.cleanup()

if __name__ == "__main__":
    system = ObstacleAvoidanceSystem("localhost", 25252)
    system.run()
