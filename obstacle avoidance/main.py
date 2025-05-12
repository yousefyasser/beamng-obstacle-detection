from PIL import Image
from time import sleep

from detectionManager import DetectionManager
from environmentManager import EnvironmentManager
from visualizationManager import VisualizationManager
from imageProcessor import ImageProcessingPipeline, GaussianNoiseProcessor, GaussianBlurProcessor, BrightnessContrastProcessor, RainEffectProcessor, ColorFilterProcessor
from logger import system_logger


class ObstacleAvoidanceSystem:
    """Main class that coordinates all components"""
    def __init__(self, host: str, port: int):
        self.logger = system_logger.logger
        self.logger.info("Starting obstacle avoidance system")
        
        self.environment = EnvironmentManager(host, port)
        self.detector = DetectionManager()
        self.visualizer = VisualizationManager()
        
        # Initialize image processors
        self.image_processors = ImageProcessingPipeline([
            GaussianNoiseProcessor(),
            GaussianBlurProcessor(radius=1),
            BrightnessContrastProcessor(),
            RainEffectProcessor(intensity=0.01),
            ColorFilterProcessor()
        ])

    def run(self):
        while True:
            try:
                image = self.environment.get_camera_image()
                if image is None:
                    self.logger.warning("No image received from camera")
                    continue
                
                processed_image = self.image_processors.process(image)
                
                detections = self.detector.detect(processed_image)

                is_car_ahead = self.detector.is_car_detected(detections)
                self.environment.control_vehicles(is_car_ahead)

                self.visualizer.visualize(processed_image, detections)
                
                sleep(0.01)
            except KeyboardInterrupt:
                self.logger.info("Received keyboard interrupt, stopping...")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {str(e)}")
                continue

        self.cleanup()

    def cleanup(self):
        self.visualizer.cleanup()
        self.environment.cleanup()

if __name__ == "__main__":
    system = ObstacleAvoidanceSystem("localhost", 25252)
    system.run()
