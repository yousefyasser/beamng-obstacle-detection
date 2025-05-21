from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
from typing import Optional
from PIL import Image
from ..utils.logger import system_logger
from ..config.dataclasses import VehicleConfig, CameraConfig
import math


class EnvironmentManager:
    """Class responsible for managing BeamNG environment"""
    def __init__(self, host: str, port: int):
        self.beamng = BeamNGpy(host, port)
        self.vehicle = None
        self.obstacle = None
        self.camera = None
        self.logger = system_logger.environment_logger
        self.vehicle_detected = False
        
        # Default configurations
        self.ego_config = VehicleConfig()
        self.obstacle_config = VehicleConfig(
            color="White",
            license="SENSORS2",
            position=(186, -940, 273),
            rotation=(0.0096, -0.0077, 0.9999, 0.0050)
        )
        self.camera_config = CameraConfig()
        
        self.setup_environment()

    def setup_environment(self):
        """Set up the BeamNG environment"""
        try:
            self.beamng.open()
            scenario = Scenario("italy", "driver_comfort")
            
            # Setup ego vehicle
            self.vehicle = Vehicle(
                "ego",
                model=self.ego_config.model,
                license=self.ego_config.license,
                color=self.ego_config.color
            )
            
            # Setup obstacle vehicle
            self.obstacle = Vehicle(
                "obstacle",
                model=self.obstacle_config.model,
                license=self.obstacle_config.license,
                color=self.obstacle_config.color
            )
            
            # Add vehicles to scenario
            scenario.add_vehicle(
                self.vehicle,
                pos=self.ego_config.position,
                rot_quat=self.ego_config.rotation
            )
            scenario.add_vehicle(
                self.obstacle,
                pos=self.obstacle_config.position,
                rot_quat=self.obstacle_config.rotation
            )
            
            # Setup and start scenario
            scenario.make(self.beamng)
            self.beamng.scenario.load(scenario)
            self.beamng.settings.set_deterministic()
            self.beamng.settings.set_steps_per_second(60)
            self.beamng.env.set_tod(0.0)
            # self.beamng.env.set_weather_preset('rainy')
            # self.beamng.env.set_weather_preset('foggy_night')
            self.beamng.scenario.start()
            
            self.setup_camera()
            # waypoints = scenario.find_waypoints()
            # for waypoint in waypoints:
            #     self.logger.info(waypoint)

            # self.vehicle.set_lights(headlights=2)
            self.vehicle.set_lights(fog_lights=2)
            self.vehicle.ai.set_mode('manual')
            self.vehicle.ai.set_waypoint('mountain_village_road1_x')
            self.logger.info("Environment setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup environment: {str(e)}")
            raise

    def setup_camera(self):
        """Set up the camera sensor"""
        try:
            self.camera = Camera(
                'camera1',
                self.beamng,
                self.vehicle,
                requested_update_time=self.camera_config.update_time,
                is_using_shared_memory=True,
                pos=self.camera_config.position,
                dir=self.camera_config.direction,
                field_of_view_y=self.camera_config.fov,
                near_far_planes=(0.1, 1000),
                resolution=self.camera_config.resolution,
                is_streaming=True,
                is_render_colours=True
            )
            self.logger.info("Camera setup completed successfully")
        except Exception as e:
            self.logger.error(f"Failed to setup camera: {str(e)}")
            raise

    def get_camera_image(self) -> Optional[Image.Image]:
        """Get the current camera image"""
        try:
            img = self.camera.stream()
            if 'colour' in img and img['colour'] is not None:
                return img['colour'].convert('RGB')
            return None
        except Exception as e:
            self.logger.error(f"Failed to get camera image: {str(e)}")
            return None
    
    def control_vehicles(self, car_detected: bool):
        """Control vehicles based on detection status"""
        try:
            if car_detected or self.vehicle_detected:
                self.vehicle_detected = True
                self.vehicle.ai.set_mode('disabled')
                self.vehicle.control(throttle=0.0, brake=1.0, gear=0)
                
                crash_distance = self.calculate_crash_distance()
                self.logger.info(f"Crash distance: {crash_distance}")
        except Exception as e:
            self.logger.error(f"Failed to control vehicles: {str(e)}")

    def calculate_crash_distance(self):
        """Calculate the crash distance"""
        car1_pos = self.vehicle.state['pos']
        car2_pos = self.obstacle.state['pos']
        distance = math.sqrt((car1_pos[0] - car2_pos[0])**2 + (car1_pos[1] - car2_pos[1])**2 + (car1_pos[2] - car2_pos[2])**2)
        return distance

    def cleanup(self):
        """Clean up resources"""
        try:
            self.beamng.close()
            self.logger.info("Environment cleanup completed")
        except Exception as e:
            self.logger.error(f"Failed to cleanup environment: {str(e)}")