import os
import math
import random
import imageio
import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from gym_air_traffic.envs.entities import Aircraft, LandingZone
from gym_air_traffic.envs.renderer import Renderer

class AirTrafficEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30, "name": "air_traffic_v0"}

    def __init__(self, render_mode=None):
        super().__init__()
        self.width = 800
        self.height = 600
        self.render_mode = render_mode
        self.renderer = Renderer(self.width, self.height)
        
        self.max_planes = 10
        self.spawn_rate = 0.05 
        
        self.zones = [
            LandingZone(600, 200, 0, "runway_red", 0),
            LandingZone(600, 500, 0, "runway_blue", 1),
            LandingZone(150, 450, 0, "helipad", 2)
        ]

        self.wind_vector = np.array([0.0, 0.0])
        self.max_wind_speed = 0.001
        self.wind_change_rate = 0.05

        self.possible_agents = [f"plane_{i}" for i in range(self.max_planes)]
        # Les agents sont permanents
        self.agents = self.possible_agents[:]
        self.planes_dict = {agent: None for agent in self.possible_agents}

        obs_dim = 10 + ((self.max_planes - 1) * 6)

        self.observation_spaces = {
            agent: spaces.Box(low=-float("inf"), high=float("inf"), shape=(obs_dim,), dtype=np.float32) 
            for agent in self.possible_agents
        }

        self.action_spaces = {
            agent: spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32) 
            for agent in self.possible_agents
        }

        self.steps = 0

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.planes_dict = {agent: None for agent in self.possible_agents}
        self.steps = 0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        angle = random.uniform(0, 2 * math.pi)
        strength = random.uniform(0, self.max_wind_speed)
        self.wind_vector = np.array([
            strength * math.cos(angle),
            strength * math.sin(angle)
        ])
        
        self._spawn_planes()
        
        observations = {agent: self._get_single_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def step(self, actions):
        self.steps += 1
        
        # Les agents ne meurent jamais avant la fin (truncations)
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: self.steps >= 1000 for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        noise = np.random.uniform(-self.wind_change_rate, self.wind_change_rate, size=2)
        self.wind_vector += noise
        
        current_wind_speed = np.linalg.norm(self.wind_vector)
        if current_wind_speed > self.max_wind_speed:
            self.wind_vector = (self.wind_vector / current_wind_speed) * self.max_wind_speed

        for agent in self.agents:
            plane = self.planes_dict[agent]
            if plane is not None and plane.active:
                if agent in actions:
                    command = actions[agent]
                    plane.change_heading(command[0])
                    plane.change_speed(command[1])
                    plane.move(self.wind_vector)
                
                target_zone = next((z for z in self.zones if z.id == plane.destination_id), None)
                if target_zone:
                    dist_before = getattr(plane, "last_dist", math.sqrt((target_zone.x - plane.x)**2 + (target_zone.y - plane.y)**2))
                    dist_after = math.sqrt((target_zone.x - plane.x)**2 + (target_zone.y - plane.y)**2)
                    plane.last_dist = dist_after
                    
                    rewards[agent] += (dist_before - dist_after) * 0.1
                    
                    if target_zone.type != "helipad" and dist_after < 400:
                        angle_diff = abs(plane.heading - target_zone.angle)
                        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
                        if abs(angle_diff) < 0.5:
                            rewards[agent] += 0.5

        self._check_collisions(rewards)
        self._check_landings(rewards)
        self._check_out_of_bounds(rewards)

        # Retirer les avions désactivés (mais garder les agents dans le dictionnaire)
        for agent in self.agents:
            if self.planes_dict[agent] is not None and not self.planes_dict[agent].active:
                self.planes_dict[agent] = None

        self._spawn_planes()

        observations = {agent: self._get_single_obs(agent) for agent in self.agents}

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def render(self):
        active_planes = [self.planes_dict[agent] for agent in self.agents if self.planes_dict[agent] is not None]
        return self.renderer.draw(self.render_mode, active_planes, self.zones, self.wind_vector)

    def close(self):
        self.renderer.close()

    def save_video(self, folder_path, frames, filename="episode.mp4", fps=30):
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, filename)
        if frames:
            try:
                imageio.mimsave(file_path, frames, fps=fps, macro_block_size=1)
                print(f"Video saved: {file_path}")
            except Exception as e:
                pass

    def _spawn_planes(self):
        empty_agents = [agent for agent in self.agents if self.planes_dict[agent] is None]
        if empty_agents and random.random() < self.spawn_rate:
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
                p_type, dest_id, speed = "jet_red", 0, 2.5
            elif rand_type < 0.9:
                p_type, dest_id, speed = "jet_blue", 1, 2.5
            else:
                p_type, dest_id, speed = "helicopter", 2, 1.5
            
            new_agent_id = random.choice(empty_agents)
            new_plane = Aircraft(x, y, speed=speed, heading=h, plane_type=p_type, destination_id=dest_id)
            self.planes_dict[new_agent_id] = new_plane

    def _check_collisions(self, rewards):
        n = len(self.agents)
        for i in range(n):
            for j in range(i + 1, n):
                a1, a2 = self.agents[i], self.agents[j]
                p1, p2 = self.planes_dict[a1], self.planes_dict[a2]
                
                if p1 is not None and p2 is not None and p1.active and p2.active:
                    dist = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                    if dist < 30: 
                        rewards[a1] -= 100.0
                        rewards[a2] -= 100.0
                        p1.active = False
                        p2.active = False

    def _check_landings(self, rewards):
        for agent in self.agents:
            plane = self.planes_dict[agent]
            if plane is not None and plane.active:
                for zone in self.zones:
                    if zone.id == plane.destination_id and zone.validate_landing(plane):
                        if plane.speed <= plane.landing_speed_limit:
                            rewards[agent] += 150.0
                        else:
                            rewards[agent] -= 50.0
                        plane.active = False
                        break

    def _check_out_of_bounds(self, rewards):
        for agent in self.agents:
            plane = self.planes_dict[agent]
            if plane is not None and plane.active:
                if plane.x < -50 or plane.x > self.width + 50 or plane.y < -50 or plane.y > self.height + 50:
                    rewards[agent] -= 200.0
                    plane.active = False

    def _get_single_obs(self, agent):
        plane = self.planes_dict[agent]
        wx_norm = self.wind_vector[0] / self.max_wind_speed
        wy_norm = self.wind_vector[1] / self.max_wind_speed
        
        if plane is None or not plane.active:
            # Observation vide pour les agents en attente
            return np.zeros(64, dtype=np.float32)

        target_zone = next((z for z in self.zones if z.id == plane.destination_id), None)
        tx, ty = (target_zone.x, target_zone.y) if target_zone else (0.0, 0.0)
        
        dx_target = (tx - plane.x) / self.width
        dy_target = (ty - plane.y) / self.height
        
        t_val = 0.0 if plane.type == "jet_red" else 0.5 if plane.type == "jet_blue" else 1.0

        obs_list = [
            plane.x / self.width, plane.y / self.height,
            (plane.speed - plane.min_speed) / (plane.max_speed - plane.min_speed),
            math.cos(plane.heading), math.sin(plane.heading),
            dx_target, dy_target, t_val, wx_norm, wy_norm
        ]

        for other_agent in self.possible_agents:
            if other_agent == agent:
                continue
            
            other_plane = self.planes_dict[other_agent]
            if other_plane is not None and other_plane.active:
                dx = (other_plane.x - plane.x) / self.width
                dy = (other_plane.y - plane.y) / self.height
                dv = (other_plane.speed - plane.speed) / (plane.max_speed - plane.min_speed)
                dhead = other_plane.heading - plane.heading
                
                obs_list.extend([dx, dy, dv, math.cos(dhead), math.sin(dhead), 1.0])
            else:
                obs_list.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        return np.array(obs_list, dtype=np.float32)