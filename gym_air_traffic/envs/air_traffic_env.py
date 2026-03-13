import os
import math
import random
import logging
import imageio
import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from gym_air_traffic.envs.entities import Aircraft, LandingZone
from gym_air_traffic.envs.renderer import Renderer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AirTrafficEnv")

class AirTrafficEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30, "name": "air_traffic_v0"}

    def __init__(self, render_mode=None, max_planes=10, enable_acceleration=True, enable_wind=True):
        super().__init__()
        self.width = 800
        self.height = 600
        self.render_mode = render_mode
        self.renderer = Renderer(self.width, self.height)
        
        self.max_planes = max_planes
        self.enable_acceleration = enable_acceleration
        self.enable_wind = enable_wind
        
        self.total_spawned = 0
        self.nb_active_agents = 0
        self.helicopter_spawned = False
        self.spawn_rate = 0.05 
        
        self.zones = [
            LandingZone(600, 200, 0, "runway_red", 0),
            LandingZone(600, 500, 0, "runway_blue", 1),
            LandingZone(150, 450, 0, "helipad", 2)
        ]

        self.wind_vector = np.array([0.0, 0.0])
        self.max_wind_speed = 1.0
        self.wind_change_rate = 0.05

        self.possible_agents = [f"plane_{i}" for i in range(self.max_planes)]
        self.agents = self.possible_agents[:]
        self.planes_dict = {agent: None for agent in self.possible_agents}

        base_features = 14 if self.enable_wind else 12
        self.obs_dim = base_features + ((self.max_planes - 1) * 6)
        self.action_dim = 2 if self.enable_acceleration else 1

        self.observation_spaces = {
            agent: spaces.Box(low=-float("inf"), high=float("inf"), shape=(self.obs_dim,), dtype=np.float32) 
            for agent in self.possible_agents
        }

        self.action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32) 
            for agent in self.possible_agents
        }

        self.steps = 0

        logger.info(f"Environment initialized with max_planes={self.max_planes}, enable_acceleration={self.enable_acceleration}, enable_wind={self.enable_wind}. Observation dimension: {self.obs_dim}, Action dimension: {self.action_dim}.")

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.planes_dict = {agent: None for agent in self.possible_agents}
        self.steps = 0
        self.total_spawned = 0
        self.helicopter_spawned = False
        self.wind_vector = np.array([0.0, 0.0])
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        if self.enable_wind and self.max_wind_speed > 0:
            angle = random.uniform(0, 2 * math.pi)
            strength = random.uniform(0, self.max_wind_speed)
            self.wind_vector = np.array([
                strength * math.cos(angle),
                strength * math.sin(angle)
            ])
        
        while self.total_spawned < self.max_planes:
            self._spawn_planes()
            
        observations = {agent: self._get_single_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        self.nb_active_agents = sum(1 for p in self.planes_dict.values() if p is not None and p.active)

        return observations, infos

    def step(self, actions):
        self.steps += 1
        
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: self.steps >= 1000 for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        # 1. Base time penalty
        for agent in self.agents:
            plane = self.planes_dict[agent]
            if plane is not None and plane.active:
                rewards[agent] -= 0.15

        # 2. Wind dynamics
        if self.enable_wind and self.max_wind_speed > 0:
            noise = np.random.uniform(-self.wind_change_rate, self.wind_change_rate, size=2)
            self.wind_vector += noise
            current_wind_speed = np.linalg.norm(self.wind_vector)
            if current_wind_speed > self.max_wind_speed:
                self.wind_vector = (self.wind_vector / current_wind_speed) * self.max_wind_speed
        
        # 3. Agent action loop and reward calculations
        for agent in self.agents:
            plane = self.planes_dict[agent]
            if plane is not None and plane.active:
                # Apply actions
                if agent in actions:
                    command = actions[agent]
                    plane.change_heading(command[0])
                    if self.enable_acceleration:
                        plane.change_speed(command[1])
                    plane.move(self.wind_vector)
                
                # --- THE MULTI-RUNWAY APPROACH GATE CHECK ---
                for zone in self.zones:
                    if zone.type != "helipad":
                        # Calculate relative distance to THIS specific zone's gate
                        dx_zone = zone.x - plane.x
                        dy_zone = zone.y - plane.y
                        
                        zone_long = dx_zone * math.cos(zone.angle) + dy_zone * math.sin(zone.angle)
                        zone_lat = -dx_zone * math.sin(zone.angle) + dy_zone * math.cos(zone.angle)
                        
                        # Is the plane inside this zone's approach gate? (100 to 200 pixels away)
                        if 100.0 < zone_long < 200.0 and abs(zone_lat) < 30.0:
                            if zone.id == plane.destination_id:
                                # Correct Gate! Grant one-time reward.
                                if not plane.passed_gate:
                                    rewards[agent] += 150.0
                                    plane.passed_gate = True
                                    logger.info(f"{agent} perfectly passed through the correct gate!")
                            else:
                                # Wrong Gate! Continuous airspace violation penalty.
                                rewards[agent] -= 2.0 
                # --------------------------------------------
                
                target_zone = next((z for z in self.zones if z.id == plane.destination_id), None)
                if target_zone:
                    dist_before = getattr(plane, "last_dist", math.sqrt((target_zone.x - plane.x)**2 + (target_zone.y - plane.y)**2))
                    dist_after = math.sqrt((target_zone.x - plane.x)**2 + (target_zone.y - plane.y)**2)
                    plane.last_dist = dist_after
                    
                    # Base distance reward (encourages generally flying toward the target)
                    rewards[agent] += (dist_before - dist_after) * 0.1 
                    
                    # --- THE GLIDE SLOPE & OVERSHOOT LOGIC ---
                    if target_zone.type != "helipad":
                        dx = target_zone.x - plane.x
                        dy = target_zone.y - plane.y
                        
                        longitudinal_dist = dx * math.cos(target_zone.angle) + dy * math.sin(target_zone.angle)
                        lateral_dist = -dx * math.sin(target_zone.angle) + dy * math.cos(target_zone.angle)
                        
                        if longitudinal_dist > 0: 
                            # We are approaching the runway from the front
                            cross_track_penalty = abs(lateral_dist) / 100.0
                            rewards[agent] -= cross_track_penalty * 0.05
                            
                            if abs(lateral_dist) < 50.0:
                                runway_diff = plane.heading - target_zone.angle
                                runway_diff = (runway_diff + math.pi) % (2 * math.pi) - math.pi
                                alignment_bonus = max(0.0, 1.0 - (abs(runway_diff) / 0.5))
                                rewards[agent] += alignment_bonus * 0.15
                        else:
                            # OVERSHOOT: We flew past the runway
                            if abs(lateral_dist) < 100.0: 
                                # Missed approach! Kill the episode so it can't circle.
                                rewards[agent] -= 200.0   
                                terminations[agent] = True 
                                plane.active = False
                                logger.info(f"Missed approach for {agent}. Episode terminated.")
                            else:
                                # Just flying away off-course
                                rewards[agent] -= 0.5
                    # -----------------------------------------

        # 4. End-of-step state checks
        self._check_collisions(rewards, terminations)
        self._check_landings(rewards, terminations)
        self._check_out_of_bounds(rewards, terminations)

        # Cleanup inactive planes
        for agent in self.agents:
            plane = self.planes_dict[agent]
            if plane is None or not plane.active:
                terminations[agent] = True

        observations = {agent: self._get_single_obs(agent) for agent in self.agents}

        if self.render_mode == "human":
            self.render()

        self.nb_active_agents = sum(1 for p in self.planes_dict.values() if p is not None and p.active)

        return observations, rewards, terminations, truncations, infos

    def render(self):
        active_planes = [self.planes_dict[agent] for agent in self.agents if self.planes_dict[agent] is not None and self.planes_dict[agent].active]
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
            except Exception:
                pass

    def _spawn_planes(self):
        spawn_successful = False
        
        rand_type = random.random()
        if rand_type < 0.45:
            p_type, dest_id, speed = "jet_red", 0, 2.5
        elif rand_type < 0.9:
            p_type, dest_id, speed = "jet_blue", 1, 2.5
        else:
            if not self.helicopter_spawned:
                p_type, dest_id, speed = "helicopter", 2, 1.5
                self.helicopter_spawned = True
            else:
                p_type, dest_id, speed = random.choice([("jet_red", 0, 2.5), ("jet_blue", 1, 2.5)])

        target_zone = next((z for z in self.zones if z.id == dest_id), None)
        tx = target_zone.x if target_zone else self.width / 2
        ty = target_zone.y if target_zone else self.height / 2
        
        for _ in range(10):
            side = random.choice(["top", "bottom", "left", "right"])
            
            if side == "top":
                x, y = random.uniform(50, self.width - 50), 0
            elif side == "bottom":
                x, y = random.uniform(50, self.width - 50), self.height
            elif side == "left":
                x, y = 0, random.uniform(50, self.height - 50)
            else:
                x, y = self.width, random.uniform(50, self.height - 50)

            too_close = False
            for p in self.planes_dict.values():
                if p is not None and p.active:
                    dist = math.sqrt((p.x - x)**2 + (p.y - y)**2)
                    if dist < 100:
                        too_close = True
                        break
            
            if not too_close:
                ideal_h = math.atan2(ty - y, tx - x)
                noise = random.uniform(-0.5, 0.5)
                h = (ideal_h + noise + math.pi) % (2 * math.pi) - math.pi
                spawn_successful = True
                break

        if spawn_successful:
            new_agent_id = self.possible_agents[self.total_spawned]
            new_plane = Aircraft(x, y, speed=speed, heading=h, plane_type=p_type, destination_id=dest_id)
            self.planes_dict[new_agent_id] = new_plane
            self.total_spawned += 1

    def _check_collisions(self, rewards, terminations):
        n = len(self.agents)
        for i in range(n):
            for j in range(i + 1, n):
                a1, a2 = self.agents[i], self.agents[j]
                p1, p2 = self.planes_dict[a1], self.planes_dict[a2]
                
                if p1 is not None and p2 is not None and p1.active and p2.active:
                    dist = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                    if dist < 30: 
                        rewards[a1] -= 500.0
                        rewards[a2] -= 500.0
                        p1.active = False
                        p2.active = False
                        terminations[a1] = True
                        terminations[a2] = True
                        logger.info(f"Collision detected between {a1} and {a2} at step {self.steps}. Distance: {dist:.2f}")

    def _check_landings(self, rewards, terminations):
        for agent in self.agents:
            plane = self.planes_dict[agent]
            if plane is not None and plane.active:
                for zone in self.zones:
                    if zone.id == plane.destination_id and zone.validate_landing(plane):
                        if not self.enable_acceleration or plane.speed <= plane.landing_speed_limit:
                            rewards[agent] += 500.0
                            logger.info(f"Successful landing for {agent} at step {self.steps}. Speed: {plane.speed:.2f}")
                        else:
                            rewards[agent] -= 50.0
                            logger.info(f"Hard landing for {agent} at step {self.steps}. Speed: {plane.speed:.2f}")
                        plane.active = False
                        terminations[agent] = True
                        break

    def _check_out_of_bounds(self, rewards, terminations):
        for agent in self.agents:
            plane = self.planes_dict[agent]
            if plane is not None and plane.active:
                if plane.x < -50 or plane.x > self.width + 50 or plane.y < -50 or plane.y > self.height + 50:
                    rewards[agent] -= 200.0
                    logger.info(f"Plane {agent} went out of bounds at step {self.steps}.")
                    plane.active = False
                    terminations[agent] = True

    def _get_single_obs(self, agent):
        plane = self.planes_dict[agent]
        
        if plane is None or not plane.active:
            return np.full(self.obs_dim, -1.0, dtype=np.float32)

        target_zone = next((z for z in self.zones if z.id == plane.destination_id), None)
        tx, ty = (target_zone.x, target_zone.y) if target_zone else (0.0, 0.0)
        t_angle = target_zone.angle if target_zone else 0.0
        
        dx = (tx - plane.x)
        dy = (ty - plane.y)

        longitudinal_dist = dx * math.cos(t_angle) + dy * math.sin(t_angle)
        lateral_dist = -dx * math.sin(t_angle) + dy * math.cos(t_angle)

        longitudinal_norm = longitudinal_dist / self.width
        lateral_norm = lateral_dist / self.height
        
        ideal_heading = math.atan2(ty - plane.y, tx - plane.x)
        relative_heading = plane.heading - ideal_heading
        
        t_val = 0.0 if plane.type == "jet_red" else 0.5 if plane.type == "jet_blue" else 1.0

        obs_list = [
            plane.x / self.width, # coordinate x normalized
            plane.y / self.height, # coordinate y normalized
            (plane.speed - plane.min_speed) / (plane.max_speed - plane.min_speed), # speed normalized
            math.cos(plane.heading), # cosine of heading
            math.sin(plane.heading), # sine of heading
            math.cos(relative_heading), # cosine of relative heading to target
            math.sin(relative_heading),# sine of relative heading to target
            longitudinal_norm, # x distance to target normalized
            lateral_norm, # y distance to target normalized
            math.cos(t_angle), # cosine of target zone angle
            math.sin(t_angle), # sine of target zone angle
            t_val # type of plane as a value between 0 and 1
        ]

        if self.enable_wind:
            if self.max_wind_speed > 0:
                wx_norm = self.wind_vector[0] / self.max_wind_speed
                wy_norm = self.wind_vector[1] / self.max_wind_speed
            else:
                wx_norm = 0.0
                wy_norm = 0.0
            obs_list.extend([wx_norm, wy_norm])

        for other_agent in self.possible_agents:
            if other_agent == agent:
                continue
            
            other_plane = self.planes_dict[other_agent]
            if other_plane is not None and other_plane.active:
                dx = (other_plane.x - plane.x) / self.width # x distance to other plane normalized
                dy = (other_plane.y - plane.y) / self.height # y distance to other plane normalized
                dv = (other_plane.speed - plane.speed) / (plane.max_speed - plane.min_speed) # relative speed normalized
                dhead = other_plane.heading - plane.heading # relative heading
                
                obs_list.extend([dx, 
                                 dy, 
                                 dv, 
                                 math.cos(dhead), 
                                 math.sin(dhead), 
                                 1.0 # indicator that this slot contains an active plane
                                 ])
            else:
                obs_list.extend([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

        return np.array(obs_list, dtype=np.float32)

if __name__ == "__main__":
    env = AirTrafficEnv(max_planes=4, enable_acceleration=False, enable_wind=False)
    observations, infos = env.reset()
    
    print("Testing environment without wind and without acceleration.")
    print(f"Action dimension: {env.action_dim}")
    print(f"Observation dimension: {env.obs_dim}")
    
    for i in range(100):
        actions = {}
        for agent in env.agents:
            if env.planes_dict[agent] is not None and env.planes_dict[agent].active:
                actions[agent] = env.action_space(agent).sample()
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        if i % 25 == 0:
            print(f"Step {i} | Active: {env.nb_active_agents} | Total: {env.total_spawned}")

    env.close()
    print("Simulation completed successfully.")