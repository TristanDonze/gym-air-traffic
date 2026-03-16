import numpy as np
import math

class Aircraft:
    def __init__(self, x, y, speed, heading, plane_type, destination_id):
        self.x = float(x)
        self.y = float(y)
        self.heading = float(heading)
        self.type = plane_type
        self.destination_id = destination_id
        self.active = True
        self.approach_armed = False
        self.passed_gate = False
        
        self.speed = float(speed)
        self.min_speed = 1.0
        self.max_speed = 5.0
        self.accel_rate = 0.2
        
        self.landing_speed_limit = 2.0

        self.turn_rate = 0.05 if plane_type == "jet_red" or plane_type == "jet_blue" else 0.08

    def move(self, wind_vector):
        dx_plane = self.speed * math.cos(self.heading)
        dy_plane = self.speed * math.sin(self.heading)
        
        self.x += dx_plane + wind_vector[0]
        self.y += dy_plane + wind_vector[1]
        
    def change_heading(self, steering_command):
        step = np.clip(steering_command, -1.0, 1.0) * self.turn_rate
        self.heading += step
        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi
    
    def change_speed(self, throttle_command, scale=1.0):
        if scale <= 0.0:
            return

        delta = np.clip(throttle_command, -1.0, 1.0) * self.accel_rate * scale
        self.speed += delta
        self.speed = np.clip(self.speed, self.min_speed, self.max_speed)

    def get_pos(self):
        return np.array([self.x, self.y])


class LandingZone:
    def __init__(self, x, y, angle, zone_type, zone_id):
        self.x = x
        self.y = y
        self.angle = angle 
        self.type = zone_type
        self.id = zone_id
        self.radius = 50.0

    def validate_landing(self, aircraft):
        # 1. Check if the plane type matches the runway type
        is_match = False
        if self.type == "runway_red" and aircraft.type == "jet_red":
            is_match = True
        elif self.type == "runway_blue" and aircraft.type == "jet_blue":
            is_match = True
        elif self.type == "helipad" and aircraft.type == "helicopter":
            is_match = True

        if not is_match:
            return False

        # 2. Helipads remain a simple circle
        if self.type == "helipad":
            dist = math.sqrt((self.x - aircraft.x)**2 + (self.y - aircraft.y)**2)
            return dist <= 20.0

        # 3. RUNWAYS: The Perpendicular "Finish Line"
        dx = self.x - aircraft.x
        dy = self.y - aircraft.y
        longitudinal_dist = dx * math.cos(self.angle) + dy * math.sin(self.angle)
        lateral_dist = -dx * math.sin(self.angle) + dy * math.cos(self.angle)
        
        # Create a short line strictly in the middle of the runway:
        # - Longitudinal depth: +/- 5.0 pixels (thick enough so planes don't skip over it in one frame)
        # - Lateral width: +/- 10.0 pixels (forces them to be perfectly aligned with the centerline)
        if not (abs(longitudinal_dist) <= 5.0 and abs(lateral_dist) <= 10.0):
            return False

        # 4. Check if the plane is facing straight down the runway
        heading_diff = abs(aircraft.heading - self.angle)
        heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi
        
        if abs(heading_diff) > 0.15: 
            return False
       
        return True
