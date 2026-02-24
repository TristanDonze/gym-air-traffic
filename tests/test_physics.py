import math
from gym_air_traffic.envs.entities import Aircraft

plane = Aircraft(0, 0, speed=10, heading=0, plane_type="jet", destination_id=0)

print(f"Start: x={plane.x}, heading={plane.heading}")

target_north = math.pi / 2

for i in range(10):
    plane.change_heading(target_north)
    plane.move()
    print(f"Frame {i+1}: x={plane.x:.2f}, y={plane.y:.2f}, heading={plane.heading:.2f}")