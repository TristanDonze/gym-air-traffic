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

        self.turn_rate = 0.15 if plane_type == "jet_red" or plane_type == "jet_blue" else 0.2

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
        
        if dist > 100.0:
            return False

        if self.type != aircraft.type:
            return False

        if self.type == "helipad":
            return True

        angle_diff = abs(aircraft.heading - self.angle)
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        
        if abs(angle_diff) < 1.2: 
            return True
            
        return False