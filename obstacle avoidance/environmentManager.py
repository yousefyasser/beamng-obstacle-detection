from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
from typing import Optional
from PIL import Image


class EnvironmentManager:
    """Class responsible for managing BeamNG environment"""
    def __init__(self, host: str, port: int):
        self.beamng = BeamNGpy(host, port)
        self.vehicle = None
        self.camera = None
        self.setup_environment()

    def setup_environment(self):
        self.beamng.open()
        scenario = Scenario("italy", "driver_comfort")
        
        self.vehicle = Vehicle("ego", model="etk800", license="SENSORS", color="Blue")
        self.obstacle = Vehicle("obstacle", model="etk800", license="SENSORS2", color="White")
        
        scenario.add_vehicle(self.vehicle, pos=(237.90, -894.42, 246.10), 
                           rot_quat=(0.0173, -0.0019, -0.6354, 0.7720))
        scenario.add_vehicle(self.obstacle, pos=(244.90, -894.42, 246.10), 
                           rot_quat=(0.0173, -0.0019, -0.6354, 0.7720))
        
        scenario.make(self.beamng)
        self.beamng.scenario.load(scenario)
        self.beamng.settings.set_deterministic()
        self.beamng.settings.set_steps_per_second(60)
        self.beamng.scenario.start()
        
        self.setup_camera()

    def setup_camera(self):
        self.camera = Camera(
            'camera1',
            self.beamng,
            self.vehicle,
            requested_update_time=0.05,
            is_using_shared_memory=False,
            pos=(-0.3, -1.3, 1.2),
            dir=(0, -1, 0),
            field_of_view_y=70,
            near_far_planes=(0.1, 1000),
            resolution=(640, 480),
            is_streaming=True,
            is_render_colours=True
        )

    def get_camera_image(self) -> Optional[Image.Image]:
        img = self.camera.poll()
        if 'colour' in img and img['colour'] is not None:
            return img['colour'].convert('RGB')
        return None

    def cleanup(self):
        self.beamng.close()