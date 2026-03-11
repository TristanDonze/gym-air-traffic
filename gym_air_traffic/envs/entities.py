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
        
        self.speed = float(speed)
        self.min_speed = 1.0
        self.max_speed = 5.0
        self.accel_rate = 0.1
        
        self.landing_speed_limit = 2.5 

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
    
    def change_speed(self, throttle_command):
        delta = np.clip(throttle_command, -1.0, 1.0) * self.accel_rate
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
        dist = math.sqrt((self.x - aircraft.x)**2 + (self.y - aircraft.y)**2)
        
        if dist > 20.0:
            return False

        is_match = False
        if self.type == "runway_red" and aircraft.type == "jet_red":
            is_match = True
        elif self.type == "runway_blue" and aircraft.type == "jet_blue":
            is_match = True
        elif self.type == "helipad" and aircraft.type == "helicopter":
            is_match = True

        if not is_match:
            return False

        if self.type == "helipad":
            return True

        heading_diff = abs(aircraft.heading - self.angle)
        heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi
        
        if abs(heading_diff) > 0.15: 
            return False
            
        pos_angle = math.atan2(self.y - aircraft.y, self.x - aircraft.x)
        pos_diff = abs(pos_angle - self.angle)
        pos_diff = (pos_diff + math.pi) % (2 * math.pi) - math.pi
        
        if abs(pos_diff) > 0.15:
            return False

        # print(f"Landing successful! Debugging every variables : dist={dist:.2f}, is_match={is_match}, heading_diff={heading_diff:.2f}, pos_angle={pos_angle:.2f}, self.angle={self.angle}, pos_diff={pos_diff:.2f}")
        return True