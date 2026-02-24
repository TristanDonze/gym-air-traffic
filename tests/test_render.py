import math
import time
from gym_air_traffic.envs.entities import Aircraft, LandingZone
from gym_air_traffic.envs.renderer import Renderer

renderer = Renderer(800, 600)
plane = Aircraft(400, 300, speed=2, heading=0, plane_type="jet", destination_id=0)
zone = LandingZone(600, 300, angle=math.pi, zone_type="runway", zone_id=0)

print("Opening window...")
for i in range(200):
    plane.change_heading(plane.heading + 0.05)
    plane.move()
    
    renderer.draw("human", [plane], [zone])

renderer.close()