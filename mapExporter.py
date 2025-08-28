import random
from time import sleep
import open3d as o3d
import mgrs
import xml.etree.ElementTree as ET

from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Lidar
from beamngpy.tools import OpenStreetMapExporter

class MapExporter:
    def __init__(self, map_name):
        self.map_name = map_name
        self.osm_file = map_name
        self.points_file = f"{map_name}.xyz"
        self.pcd_file = f"{map_name}.pcd"

        set_up_simple_logging()

        self.bng = BeamNGpy("localhost", 25252)
        self.bng.open(launch=True)

        self.bng.settings.set_deterministic(60)
        self.bng.ui.hide_hud()

        self.setup_world()

    def setup_world(self):
        # self.scenario = self.bng.scenario.get_current()
        # self.vehicle = self.bng.vehicles.get_current()
        self.scenario = Scenario(self.map_name, "Road Network Exporter Demo", description="Exports map data")
        self.vehicle = Vehicle("ego_vehicle", model="etk800", license="RED", color="Red")
        
        # self.scenario.add_vehicle(self.vehicle, pos=(0, 0, 0), rot_quat=(0, 0, 0, 1))
        self.scenario.add_vehicle(self.vehicle, pos=(237.90, -894.42, 246.10), rot_quat=(0.0173, -0.0019, -0.6354, 0.7720))

        self.scenario.make(self.bng)

        self.bng.scenario.load(self.scenario)
        self.bng.scenario.start()

    def export_map_osm(self):
        self.bng.logger.info("Exporting road network data...")
        
        OpenStreetMapExporter.export(self.osm_file, self.bng)

        self.convert_osm_to_josm()
        
        self.bng.logger.info("Road network data exported.")

    def convert_osm_to_josm(self):
        input_file = f"{self.osm_file}.osm"

        m = mgrs.MGRS()
        # Parse the input file
        tree = ET.parse(input_file)
        root = tree.getroot()
        
        # Update the root attributes
        root.attrib['generator'] = 'JOSM'
        
        # Process each node
        for node in root.findall('node'):
            # Store elevation value
            ele = node.attrib.pop('ele', None)
            lat = float(node.attrib['lat'])
            lon = float(node.attrib['lon'])
            
            # Remove unwanted attributes
            for attr in ['user', 'uid', 'changeset', 'timestamp']:
                node.attrib.pop(attr, None)
            
            # Add elevation as tag if it exists
            if ele:
                ele_tag = ET.SubElement(node, 'tag')
                ele_tag.attrib['k'] = 'ele'
                ele_tag.attrib['v'] = ele
            
            # Convert to MGRS (precision 5 = 1m)
            mgrs_code = m.toMGRS(lat, lon, MGRSPrecision=5)
                
            # Add new mgrs_code tag
            mgrs_tag = ET.SubElement(node, 'tag')
            mgrs_tag.attrib['k'] = 'mgrs_code'
            mgrs_tag.attrib['v'] = mgrs_code

        for way in root.findall('way'):
            # Remove unwanted attributes
            for attr in ['user', 'uid', 'changeset']:
                way.attrib.pop(attr, None)
            
            subtype = ET.SubElement(way, 'tag')
            subtype.attrib['k'] = 'subtype'
            subtype.attrib['v'] = 'solid'

            type = ET.SubElement(way, 'tag')
            type.attrib['k'] = 'type'
            type.attrib['v'] = 'line_thin'

            width = ET.SubElement(way, 'tag')
            width.attrib['k'] = 'width'
            width.attrib['v'] = '0.200'
        
        # Write the modified XML
        tree.write(input_file, encoding='UTF-8', xml_declaration=True)
        
        # JOSM-style formatting (replace quotes if desired)
        with open(input_file, 'r') as f:
            content = f.read()
        content = content.replace('"', "'")
        with open(input_file, 'w') as f:
            f.write(content)

    def export_map_pcd(self):
        self.collect_map_points()

        pcd = o3d.io.read_point_cloud(self.points_file)
        o3d.io.write_point_cloud(self.pcd_file, pcd)

    def collect_map_points(self):
        random.seed(1703)

        # NOTE: Create sensor after scenario has started.
        lidar = Lidar(
            "lidar1",
            self.bng,
            self.vehicle,
            requested_update_time=0.1,
            is_using_shared_memory=True,
            is_360_mode=True,
        )

        self.vehicle.ai.set_mode("span")
        
        self.bng.logger.info("Driving around, polling the LiDAR sensor every 5 seconds...")
        
        with open(self.points_file, "w") as file:
            for i in range(50):
                sleep(0.1)
                readings_data = lidar.poll()

                for point in readings_data["pointCloud"]:
                    x, y, z = point
                    file.write(f"{x} {y} {z}\n")

        lidar.remove()
        self.vehicle.ai.set_mode("disabled")
        self.bng.ui.show_hud()
        
        self.bng.logger.info("Finished collecting LiDAR data.")

if __name__ == "__main__":
    mapExporter = MapExporter("italy")
    # mapExporter = MapExporter("italy")
    # mapExporter.export_map_osm()
    mapExporter.export_map_pcd()