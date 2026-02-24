import os
import math
import random
import imageio
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from gym_air_traffic.envs.entities import Aircraft, LandingZone
from gym_air_traffic.envs.renderer import Renderer

class AirTrafficEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.width = 800
        self.height = 600
        self.render_mode = render_mode
        self.renderer = Renderer(self.width, self.height)
        
        self.max_planes = 10
        self.spawn_rate = 0.05 
        
        self.zones = [
            LandingZone(600, 200,0, "runway_red", 0),
            LandingZone(600, 500, 0, "runway_blue", 1),
            LandingZone(150, 450, 0, "helipad", 2)
        ]

        self.wind_vector = np.array([0.0, 0.0])
        self.max_wind_speed = 1.0
        self.wind_change_rate = 0.05

        self.observation_space = spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(self.max_planes, 11),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([[-math.pi, -1.0]] * self.max_planes, dtype=np.float32),
            high=np.array([[math.pi, 1.0]] * self.max_planes, dtype=np.float32),
            dtype=np.float32
        )

        self.planes = []
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.planes = []
        self.steps = 0
        
        angle = random.uniform(0, 2 * math.pi)
        strength = random.uniform(0, self.max_wind_speed)
        self.wind_vector = np.array([
            strength * math.cos(angle),
            strength * math.sin(angle)
        ])
        
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        reward = -1.0
        terminated = False
        truncated = False

        if self.steps >= 1000:
            truncated = True

        noise = np.random.uniform(-self.wind_change_rate, self.wind_change_rate, size=2)
        self.wind_vector += noise
        
        current_wind_speed = np.linalg.norm(self.wind_vector)
        if current_wind_speed > self.max_wind_speed:
            self.wind_vector = (self.wind_vector / current_wind_speed) * self.max_wind_speed

        self._spawn_planes()

        for i, plane in enumerate(self.planes):
            if i < len(action) and plane.active:
                command = action[i]
                plane.change_heading(command[0])
                plane.change_speed(command[1])
                plane.move(self.wind_vector)

        reward += self._check_collisions()
        reward += self._check_landings()
        reward += self._check_out_of_bounds()

        if reward <= -200:
            terminated = True

        self._clean_inactive_planes()

        observation = self._get_obs()
        info = {"plane_count": len(self.planes)}

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        return self.renderer.draw(self.render_mode, self.planes, self.zones, self.wind_vector)

    def close(self):
        self.renderer.close()

    def save_video(self, folder_path, frames, filename="episode.mp4", fps=30):
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, filename)
        
        if not frames:
            print("Error : The frames list is empty. Nothing to save.")
            return

        try:
            imageio.mimsave(file_path, frames, fps=fps, macro_block_size=1)
            print(f"Video saved successfully: {file_path} ({len(frames)} frames)")
        except Exception as e:
            print(f"Error saving video: {e}")

    def _spawn_planes(self):
        if len(self.planes) < self.max_planes and random.random() < self.spawn_rate:
            side = random.choice(["top", "bottom", "left", "right"])
            if side == "top":
                x, y, h = random.uniform(0, self.width), 0, math.pi/2
            elif side == "bottom":
                x, y, h = random.uniform(0, self.width), self.height, -math.pi/2
            elif side == "left":
                x, y, h = 0, random.uniform(0, self.height), 0
            else:
                x, y, h = self.width, random.uniform(0, self.height), math.pi

            rand_type = random.random()
            if rand_type < 0.45:
                p_type = "jet_red"
                dest_id = 0
                speed = 2.5
            elif rand_type < 0.9:
                p_type = "jet_blue"
                dest_id = 1
                speed = 2.5
            else:
                p_type = "helicopter"
                dest_id = 2
                speed = 1.5
            
            new_plane = Aircraft(x, y, speed=speed, heading=h, plane_type=p_type, destination_id=dest_id)
            self.planes.append(new_plane)

    def _check_collisions(self):
        penalty = 0
        active_planes = [p for p in self.planes if p.active]
        n = len(active_planes)
        for i in range(n):
            for j in range(i + 1, n):
                p1 = active_planes[i]
                p2 = active_planes[j]
                dist = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                if dist < 30: 
                    penalty -= 100
                    p1.active = False
                    p2.active = False
        return penalty

    def _check_landings(self):
        reward = 0
        for plane in self.planes:
            if not plane.active:
                continue
                
            for zone in self.zones:
                if zone.id == plane.destination_id:
                    if zone.validate_landing(plane):
                        
                        if plane.speed <= plane.landing_speed_limit:
                            reward += 100
                            plane.active = False
                        else:
                            reward -= 50 
                            plane.active = False
                            
        return reward

    def _check_out_of_bounds(self):
        penalty = 0
        for plane in self.planes:
            if not plane.active:
                continue
            if plane.x < -50 or plane.x > self.width + 50 or plane.y < -50 or plane.y > self.height + 50:
                penalty -= 10
                plane.active = False 
        return penalty

    def _clean_inactive_planes(self):
        self.planes = [p for p in self.planes if p.active]

    def _get_obs(self):
        obs = np.zeros((self.max_planes, 11), dtype=np.float32)
        
        wx_norm = self.wind_vector[0] / self.max_wind_speed
        wy_norm = self.wind_vector[1] / self.max_wind_speed
        
        for i, plane in enumerate(self.planes):
            if i >= self.max_planes:
                break
            
            target_zone = next((z for z in self.zones if z.id == plane.destination_id), None)
            tx, ty = (target_zone.x, target_zone.y) if target_zone else (0, 0)
            
            if plane.type == "jet_red":
                t_val = 0.0
            elif plane.type == "jet_blue":
                t_val = 0.5
            else:
                t_val = 1.0

            state = [
                plane.x / self.width,
                plane.y / self.height,
                (plane.speed - plane.min_speed) / (plane.max_speed - plane.min_speed),
                math.cos(plane.heading),
                math.sin(plane.heading),
                tx / self.width,
                ty / self.height,
                t_val,
                wx_norm,
                wy_norm,
                1.0,
            ]
            obs[i] = state
            
        return obs