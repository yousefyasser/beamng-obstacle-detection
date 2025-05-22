from time import sleep
from pathlib import Path
import sys

from src.core.detection_manager import DetectionManager
from src.core.environment_manager import EnvironmentManager
from src.visualization.visualization_manager import VisualizationManager
from src.processors.improved_image_processor import (
    ImprovedImageProcessingPipeline,
    GaussianNoiseProcessor,
    GaussianBlurProcessor,
    BrightnessContrastProcessor,
    RainEffectProcessor,
    ColorFilterProcessor,
    FogEffectProcessor
)
from src.core.anomaly_detector import AnomalyDetector, AnomalyMitigator
from src.utils.logger import system_logger
from src.config.dataclasses import ProcessingConfig, AnomalyThresholds, VisualizationConfig


class ObstacleAvoidanceSystem:
    """Main class that coordinates all components"""
    def __init__(self, host: str, port: int):
        self.logger = system_logger.visualization_logger
        self.logger.info("Starting obstacle avoidance system")
        
        try:
            # Load configuration
            self.processing_config = ProcessingConfig()
            self.anomaly_thresholds = AnomalyThresholds()
            self.visualization_config = VisualizationConfig()
            
            # Create output directory
            output_dir = Path(self.visualization_config.output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # Initialize components with configuration
            self.environment = EnvironmentManager(host, port)
            self.detector = DetectionManager()

            self.visualization_config.class_names = self.detector.get_class_names()
            self.visualizer = VisualizationManager(config=self.visualization_config)
            
            # Initialize image processors with configuration
            self.image_processors = ImprovedImageProcessingPipeline([
                # GaussianNoiseProcessor(config=self.processing_config),
                # GaussianBlurProcessor(config=self.processing_config),
                # BrightnessContrastProcessor(config=self.processing_config),
                # RainEffectProcessor(config=self.processing_config),
                # ColorFilterProcessor(config=self.processing_config),
                # FogEffectProcessor(config=self.processing_config)
            ])
            
            # Initialize anomaly detection and mitigation
            self.anomaly_detector = AnomalyDetector(thresholds=self.anomaly_thresholds)
            self.anomaly_mitigator = AnomalyMitigator()
            
            self.logger.info("System initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {str(e)}")
            raise

    def run(self):
        """Main system loop"""
        try:
            while True:
                try:
                    self.environment.vehicle.sensors.poll()
                    self.environment.obstacle.sensors.poll()
                    self.logger.info(self.environment.vehicle.state['pos'])
                    self.logger.info(self.environment.vehicle.state['rotation'])
                    # Get raw image from camera
                    image = self.environment.get_camera_image()
                    if image is None:
                        self.logger.warning("No image received from camera")
                        continue
                    
                    # Set original image for error propagation tracking
                    self.visualizer.set_original_image(image)
                    self.visualizer.add_processing_stage("Raw Image", image)
                    
                    # Process image through pipeline
                    processed_image = image
                    stage_results = []
                    
                    for processor in self.image_processors.processors:
                        try:
                            processed_image = processor.process(processed_image)
                            processor_name = processor.get_name()
                            self.visualizer.add_processing_stage(processor_name, processed_image)
                            stage_results.append((processor_name, processed_image, processor.get_metrics()))
                        except Exception as e:
                            self.logger.error(f"Error in processor {processor_name}: {str(e)}")
                            continue
                    
                    # Detect anomalies
                    all_anomalies = self.anomaly_detector.detect(processed_image)
                    self.visualizer.update_anomalies(all_anomalies)
                    
                    # if all_anomalies:
                    #     self.logger.info(f"Detected {len(all_anomalies)} anomalies")
                    #     try:
                    #         mitigated_image = self.anomaly_mitigator.mitigate(processed_image, all_anomalies)
                    #         self.visualizer.add_processing_stage("Anomaly Mitigation", mitigated_image)
                    #         processed_image = mitigated_image
                    #     except Exception as e:
                    #         self.logger.error(f"Anomaly mitigation failed: {str(e)}")
                    
                    # Detect objects
                    detections = self.detector.detect(processed_image)
                    
                    # Determine if car is ahead
                    is_car_ahead = self.detector.is_car_detected(detections)
                    
                    # Control vehicles based on detection
                    self.environment.control_vehicles(is_car_ahead)
                    
                    # Visualize results
                    self.visualizer.visualize(processed_image, detections, is_car_ahead)
                    self.visualizer.reset_error_propagation()
                    
                    sleep(0.02)
                    
                except KeyboardInterrupt:
                    self.logger.info("Received keyboard interrupt, stopping...")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {str(e)}")
                    continue
                    
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources properly"""
        try:
            self.logger.info("Cleaning up resources...")
            self.visualizer.cleanup()
            self.environment.cleanup()
            self.logger.info("Cleanup complete")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")


if __name__ == "__main__":
    try:
        system = ObstacleAvoidanceSystem("localhost", 25252)
        system.run()
    except Exception as e:
        system_logger.visualization_logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
