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

    def __init__(
        self,
        render_mode=None,
        max_planes=10,
        spawn_planes=None,
        enable_acceleration=True,
        enable_wind=True,
        include_wind_in_obs=None,
        acceleration_scale=1.0,
    ):
        super().__init__()
        self.width = 800
        self.height = 600
        self.diagonal = math.hypot(self.width, self.height)
        self.render_mode = render_mode
        self.renderer = Renderer(self.width, self.height)
        
        self.max_planes = max_planes
        self.spawn_planes = max_planes if spawn_planes is None else spawn_planes
        if not 1 <= self.spawn_planes <= self.max_planes:
            raise ValueError(
                f"spawn_planes must be between 1 and max_planes. Got spawn_planes={self.spawn_planes}, "
                f"max_planes={self.max_planes}"
            )
        self.enable_acceleration = enable_acceleration
        self.enable_wind = enable_wind
        self.include_wind_in_obs = self.enable_wind if include_wind_in_obs is None else include_wind_in_obs
        self.acceleration_scale = float(acceleration_scale)
        self.speed_control_enabled = self.enable_acceleration and self.acceleration_scale > 0.0
        
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

        self.possible_agents = [f"plane_{i}" for i in range(self.spawn_planes)]
        self.agents = self.possible_agents[:]
        self.planes_dict = {agent: None for agent in self.possible_agents}

        self.time_penalty = 0.08
        self.correct_gate_reward = 18.0
        self.wrong_gate_penalty = 0.35
        self.skipped_gate_penalty = 15.0
        self.gate_long_min = 100.0
        self.gate_long_max = 200.0
        self.gate_half_width = 30.0
        self.approach_arm_long_min = self.gate_long_max
        self.approach_arm_long_max = self.gate_long_max + 160.0
        self.approach_arm_half_width = 80.0
        self.approach_arm_heading_threshold = 0.5
        self.skipped_gate_lateral_threshold = 60.0
        self.skipped_gate_heading_threshold = 0.35
        self.spawn_min_longitudinal = self.gate_long_min + 5.0
        self.runway_side_spawn_half_width = 100.0
        self.spawn_slot_margin = 35.0
        self.spawn_slot_jitter = 8.0
        self.progress_reward_scale = 0.08
        self.progress_reward_clip = 0.75
        self.steering_penalty_scale = 0.015
        self.cross_track_penalty_scale = 0.03
        self.cross_track_penalty_clip = 0.75
        self.alignment_bonus_scale = 0.20
        self.missed_approach_lateral_threshold = 60.0
        self.missed_approach_heading_threshold = 0.35
        self.missed_approach_penalty = 35.0
        self.overshoot_drift_penalty = 0.25
        self.collision_penalty = 45.0
        self.repulsion_penalty_scale = 0.6
        self.successful_landing_reward = 45.0
        self.hard_landing_penalty = 15.0
        self.out_of_bounds_penalty = 30.0

        self.neighbor_feature_dim = 11
        base_features = 16 if self.include_wind_in_obs else 14
        self.obs_dim = base_features + ((self.max_planes - 1) * self.neighbor_feature_dim)
        self.action_dim = 2 if self.enable_acceleration else 1
        self.spawn_slots = self._build_spawn_slots()
        if self.spawn_planes > len(self.spawn_slots):
            raise ValueError(
                f"spawn_planes={self.spawn_planes} exceeds the number of available spawn slots "
                f"({len(self.spawn_slots)})"
            )

        self.observation_spaces = {
            agent: spaces.Box(low=-float("inf"), high=float("inf"), shape=(self.obs_dim,), dtype=np.float32) 
            for agent in self.possible_agents
        }

        self.action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32) 
            for agent in self.possible_agents
        }

        self.steps = 0

        logger.info(
            "Environment initialized with max_planes=%s, spawn_planes=%s, enable_acceleration=%s, "
            "acceleration_scale=%.2f, enable_wind=%s, include_wind_in_obs=%s. "
            "Observation dimension: %s, Action dimension: %s.",
            self.max_planes,
            self.spawn_planes,
            self.enable_acceleration,
            self.acceleration_scale,
            self.enable_wind,
            self.include_wind_in_obs,
            self.obs_dim,
            self.action_dim,
        )

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
                rewards[agent] -= self.time_penalty

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
                    if self.enable_acceleration and self.speed_control_enabled:
                        plane.change_speed(command[1], scale=self.acceleration_scale)
                    plane.move(self.wind_vector)
                
                # --- THE MULTI-RUNWAY APPROACH GATE CHECK ---
                for zone in self.zones:
                    if zone.type != "helipad":
                        # Calculate relative distance to THIS specific zone's gate
                        dx_zone = zone.x - plane.x
                        dy_zone = zone.y - plane.y
                        
                        zone_long = dx_zone * math.cos(zone.angle) + dy_zone * math.sin(zone.angle)
                        zone_lat = -dx_zone * math.sin(zone.angle) + dy_zone * math.cos(zone.angle)
                        
                        if (
                            self.gate_long_min < zone_long < self.gate_long_max
                            and abs(zone_lat) < self.gate_half_width
                        ):
                            if zone.id == plane.destination_id:
                                # Correct Gate! Grant one-time reward.
                                if not plane.passed_gate:
                                    rewards[agent] += self.correct_gate_reward
                                    plane.passed_gate = True
                                    infos[agent]["gate_pass_event"] = True
                                    logger.info(f"{agent} perfectly passed through the correct gate!")
                            else:
                                # Wrong Gate! Continuous airspace violation penalty.
                                rewards[agent] -= self.wrong_gate_penalty
                # --------------------------------------------
                
                target_zone = next((z for z in self.zones if z.id == plane.destination_id), None)
                if target_zone:
                    # Calculate current relative distances
                    dx = target_zone.x - plane.x
                    dy = target_zone.y - plane.y
                    longitudinal_dist = dx * math.cos(target_zone.angle) + dy * math.sin(target_zone.angle)
                    lateral_dist = -dx * math.sin(target_zone.angle) + dy * math.cos(target_zone.angle)
                    runway_diff = plane.heading - target_zone.angle
                    runway_diff = (runway_diff + math.pi) % (2 * math.pi) - math.pi

                    if (
                        target_zone.type != "helipad"
                        and not plane.passed_gate
                        and not plane.approach_armed
                        and self.approach_arm_long_min < longitudinal_dist < self.approach_arm_long_max
                        and abs(lateral_dist) < self.approach_arm_half_width
                        and abs(runway_diff) < self.approach_arm_heading_threshold
                    ):
                        plane.approach_armed = True
                        infos[agent]["approach_arm_event"] = True
                    
                    # 1. Retrieve the previous distances (or initialize them)
                    dist_before_long = getattr(plane, "last_long_dist", longitudinal_dist)
                    dist_before_lat = getattr(plane, "last_lat_dist", abs(lateral_dist))
                    
                    # 2. Reward forward progress (closing the longitudinal gap)
                    rewards[agent] += self._clip_dense_reward(
                        (dist_before_long - longitudinal_dist) * self.progress_reward_scale,
                        self.progress_reward_clip,
                    )
                    
                    # 3. Reward steering progress (closing the lateral gap)
                    rewards[agent] += self._clip_dense_reward(
                        (dist_before_lat - abs(lateral_dist)) * self.progress_reward_scale,
                        self.progress_reward_clip,
                    )
                    
                    # 4. Save the current distances for the next step
                    plane.last_long_dist = longitudinal_dist
                    plane.last_lat_dist = abs(lateral_dist)
                    
                    # 5. Smoothness Penalty (Steering)
                    if agent in actions:
                        steering_effort = abs(actions[agent][0])
                        rewards[agent] -= steering_effort * self.steering_penalty_scale

                    if (
                        target_zone.type != "helipad"
                        and plane.approach_armed
                        and not plane.passed_gate
                        and longitudinal_dist <= self.gate_long_min
                        and abs(lateral_dist) < self.skipped_gate_lateral_threshold
                        and abs(runway_diff) < self.skipped_gate_heading_threshold
                    ):
                        rewards[agent] -= self.skipped_gate_penalty
                        terminations[agent] = True
                        plane.active = False
                        infos[agent]["termination_reason"] = "skipped_gate"
                        logger.info(f"{agent} skipped the mandatory approach gate. Episode terminated.")
                        continue
                    
                    # --- THE GLIDE SLOPE & OVERSHOOT LOGIC ---
                    if target_zone.type != "helipad":
                        # Check overshoot against the back edge of the runway (-60.0)
                        if longitudinal_dist > -60.0: 
                            cross_track_penalty = self._clip_dense_reward(
                                (abs(lateral_dist) / 100.0) * self.cross_track_penalty_scale,
                                self.cross_track_penalty_clip,
                            )
                            rewards[agent] -= cross_track_penalty
                            
                            if abs(lateral_dist) < 50.0:
                                alignment_bonus = max(0.0, 1.0 - (abs(runway_diff) / 0.5))
                                rewards[agent] += alignment_bonus * self.alignment_bonus_scale
                        else:
                            # OVERSHOOT: We flew past the runway
                            if (
                                plane.passed_gate
                                and abs(lateral_dist) < self.missed_approach_lateral_threshold
                                and abs(runway_diff) < self.missed_approach_heading_threshold
                            ):
                                rewards[agent] -= self.missed_approach_penalty
                                terminations[agent] = True 
                                plane.active = False
                                infos[agent]["termination_reason"] = "missed_approach"
                                logger.info(f"Missed approach for {agent}. Episode terminated.")
                            else:
                                rewards[agent] -= self.overshoot_drift_penalty
                    # -----------------------------------------

        # 4. End-of-step state checks
        self._check_collisions(rewards, terminations, infos)
        self._check_landings(rewards, terminations, infos)
        self._check_out_of_bounds(rewards, terminations, infos)

        # Cleanup inactive planes
        for agent in self.agents:
            plane = self.planes_dict[agent]
            if plane is None or not plane.active:
                terminations[agent] = True
            if terminations[agent] or truncations[agent]:
                infos[agent].setdefault("gate_passed", bool(plane is not None and plane.passed_gate))
                if truncations[agent] and not terminations[agent]:
                    infos[agent].setdefault("termination_reason", "time_limit")
                elif terminations[agent]:
                    infos[agent].setdefault("termination_reason", "inactive")

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

    def _build_spawn_slots(self):
        left_x = self.spawn_slot_margin
        right_x = self.width - self.spawn_slot_margin
        top_y = self.spawn_slot_margin
        bottom_y = self.height - self.spawn_slot_margin

        slots = []
        # Keep most entry points upstream, but allow a few runway-side starts away from the centerline.
        for y in (70.0, 150.0, 230.0, 330.0, 430.0, 530.0):
            slots.append((left_x, y))
        for x in (110.0, 220.0, 330.0, 440.0):
            slots.append((x, top_y))
            slots.append((x, bottom_y))
        for y in (70.0, 190.0, 310.0, 430.0, 550.0):
            slots.append((right_x, y))
        return slots

    def _is_valid_spawn_position(self, x, y, target_zone):
        if target_zone is None or target_zone.type == "helipad":
            return True

        dx_zone = target_zone.x - x
        dy_zone = target_zone.y - y
        zone_long = dx_zone * math.cos(target_zone.angle) + dy_zone * math.sin(target_zone.angle)
        zone_lat = -dx_zone * math.sin(target_zone.angle) + dy_zone * math.cos(target_zone.angle)
        if zone_long > self.spawn_min_longitudinal:
            return True
        if zone_long <= 0.0 and abs(zone_lat) >= self.runway_side_spawn_half_width:
            return True
        return False

    def _sample_plane_specs(self):
        plane_specs = []
        for _ in range(self.spawn_planes):
            if random.random() < 0.5:
                p_type, dest_id, speed = "jet_red", 0, 2.5
            else:
                p_type, dest_id, speed = "jet_blue", 1, 2.5

            target_zone = next((z for z in self.zones if z.id == dest_id), None)
            plane_specs.append(
                {
                    "plane_type": p_type,
                    "destination_id": dest_id,
                    "speed": speed,
                    "target_zone": target_zone,
                }
            )
        return plane_specs

    def _assign_spawn_slots(self, plane_specs):
        # Sample from a finite set of valid edge slots so high-traffic resets stay stable.
        candidate_map = {}
        for spec_idx, spec in enumerate(plane_specs):
            valid_slot_indices = [
                slot_idx
                for slot_idx, slot in enumerate(self.spawn_slots)
                if self._is_valid_spawn_position(slot[0], slot[1], spec["target_zone"])
            ]
            if not valid_slot_indices:
                raise RuntimeError(f"No valid spawn slots available for destination {spec['destination_id']}")
            candidate_map[spec_idx] = valid_slot_indices

        assignment = {}
        used_slots = set()
        ordered_specs = list(range(len(plane_specs)))
        random.shuffle(ordered_specs)
        ordered_specs.sort(key=lambda spec_idx: len(candidate_map[spec_idx]))

        def backtrack(position):
            if position == len(ordered_specs):
                return True

            spec_idx = ordered_specs[position]
            candidates = [slot_idx for slot_idx in candidate_map[spec_idx] if slot_idx not in used_slots]
            random.shuffle(candidates)
            for slot_idx in candidates:
                used_slots.add(slot_idx)
                assignment[spec_idx] = slot_idx
                if backtrack(position + 1):
                    return True
                used_slots.remove(slot_idx)
                assignment.pop(spec_idx, None)
            return False

        if not backtrack(0):
            raise RuntimeError("Unable to assign unique spawn slots for the requested traffic mix.")

        return [self.spawn_slots[assignment[spec_idx]] for spec_idx in range(len(plane_specs))]

    def _sample_spawn_pose(self, slot, target_zone):
        x_base, y_base = slot
        x = x_base
        y = y_base

        if target_zone is not None:
            for _ in range(8):
                trial_x = float(
                    np.clip(
                        x_base + random.uniform(-self.spawn_slot_jitter, self.spawn_slot_jitter),
                        self.spawn_slot_margin,
                        self.width - self.spawn_slot_margin,
                    )
                )
                trial_y = float(
                    np.clip(
                        y_base + random.uniform(-self.spawn_slot_jitter, self.spawn_slot_jitter),
                        self.spawn_slot_margin,
                        self.height - self.spawn_slot_margin,
                    )
                )
                if self._is_valid_spawn_position(trial_x, trial_y, target_zone):
                    x = trial_x
                    y = trial_y
                    break

        tx = target_zone.x if target_zone else self.width / 2
        ty = target_zone.y if target_zone else self.height / 2
        ideal_h = math.atan2(ty - y, tx - x)
        noise = random.uniform(-0.5, 0.5)
        heading = (ideal_h + noise + math.pi) % (2 * math.pi) - math.pi
        return x, y, heading

    def _spawn_planes(self):
        plane_specs = self._sample_plane_specs()
        spawn_slots = self._assign_spawn_slots(plane_specs)

        for new_agent_id, spec, slot in zip(self.possible_agents, plane_specs, spawn_slots):
            x, y, heading = self._sample_spawn_pose(slot, spec["target_zone"])
            new_plane = Aircraft(
                x,
                y,
                speed=spec["speed"],
                heading=heading,
                plane_type=spec["plane_type"],
                destination_id=spec["destination_id"],
            )
            self.planes_dict[new_agent_id] = new_plane
            self.total_spawned += 1

    def _check_collisions(self, rewards, terminations, infos):
        n = len(self.agents)
        for i in range(n):
            for j in range(i + 1, n):
                a1, a2 = self.agents[i], self.agents[j]
                p1, p2 = self.planes_dict[a1], self.planes_dict[a2]
                
                if p1 is not None and p2 is not None and p1.active and p2.active:
                    dist = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                    
                    # 1. Fatal Collision
                    if dist < 30.0: 
                        rewards[a1] -= self.collision_penalty
                        rewards[a2] -= self.collision_penalty
                        p1.active = False
                        p2.active = False
                        terminations[a1] = True
                        terminations[a2] = True
                        infos[a1]["termination_reason"] = "collision"
                        infos[a2]["termination_reason"] = "collision"
                        logger.info(f"Collision detected between {a1} and {a2} at step {self.steps}. Distance: {dist:.2f}")
                    
                    # 2. NEW: Repulsion / Airspace Violation Zone (under 150 pixels)
                    # The closer they get, the harsher the penalty.
                    elif dist < 150.0:
                        repulsion_penalty = ((150.0 - dist) / 150.0) * self.repulsion_penalty_scale
                        rewards[a1] -= repulsion_penalty
                        rewards[a2] -= repulsion_penalty

    def _check_landings(self, rewards, terminations, infos):
        for agent in self.agents:
            plane = self.planes_dict[agent]
            if plane is not None and plane.active:
                for zone in self.zones:
                    if zone.id == plane.destination_id and zone.validate_landing(plane):
                        if zone.type != "helipad" and not plane.passed_gate:
                            rewards[agent] -= self.skipped_gate_penalty
                            plane.active = False
                            terminations[agent] = True
                            infos[agent]["termination_reason"] = "skipped_gate"
                            logger.info(f"{agent} reached the runway without passing the mandatory gate.")
                            break
                        if not self.speed_control_enabled or plane.speed <= plane.landing_speed_limit:
                            rewards[agent] += self.successful_landing_reward
                            infos[agent]["termination_reason"] = "successful_landing"
                            logger.info(f"Successful landing for {agent} at step {self.steps}. Speed: {plane.speed:.2f}")
                        else:
                            rewards[agent] -= self.hard_landing_penalty
                            infos[agent]["termination_reason"] = "hard_landing"
                            logger.info(f"Hard landing for {agent} at step {self.steps}. Speed: {plane.speed:.2f}")
                        plane.active = False
                        terminations[agent] = True
                        break

    def _check_out_of_bounds(self, rewards, terminations, infos):
        for agent in self.agents:
            plane = self.planes_dict[agent]
            if plane is not None and plane.active:
                if plane.x < -50 or plane.x > self.width + 50 or plane.y < -50 or plane.y > self.height + 50:
                    rewards[agent] -= self.out_of_bounds_penalty
                    logger.info(f"Plane {agent} went out of bounds at step {self.steps}.")
                    plane.active = False
                    terminations[agent] = True
                    infos[agent]["termination_reason"] = "out_of_bounds"

    def _clip_dense_reward(self, value, limit):
        return float(np.clip(value, -limit, limit))

    def _encode_plane_type(self, plane_type):
        if plane_type == "jet_red":
            return 0.0
        if plane_type == "jet_blue":
            return 0.5
        return 1.0

    def _encode_destination(self, destination_id):
        return float(destination_id / max(1, len(self.zones) - 1))

    def _compute_conflict_features(self, plane, other_plane):
        rel_pos = np.array([other_plane.x - plane.x, other_plane.y - plane.y], dtype=np.float32)
        rel_vel = np.array(
            [
                (other_plane.speed * math.cos(other_plane.heading)) - (plane.speed * math.cos(plane.heading)),
                (other_plane.speed * math.sin(other_plane.heading)) - (plane.speed * math.sin(plane.heading)),
            ],
            dtype=np.float32,
        )

        distance = max(float(np.linalg.norm(rel_pos)), 1e-6)
        closing_speed = -float(np.dot(rel_pos, rel_vel)) / distance
        max_rel_speed = max(1.0, plane.max_speed + other_plane.max_speed)
        closing_speed_norm = float(np.clip(closing_speed / max_rel_speed, -1.0, 1.0))

        rel_speed_sq = float(np.dot(rel_vel, rel_vel))
        if rel_speed_sq < 1e-6:
            raw_time_to_cpa = 0.0
            time_to_cpa = 0.0
            distance_at_cpa = distance
        else:
            raw_time_to_cpa = -float(np.dot(rel_pos, rel_vel)) / rel_speed_sq
            time_to_cpa = float(np.clip(raw_time_to_cpa, 0.0, 60.0))
            closest_offset = rel_pos + (rel_vel * time_to_cpa)
            distance_at_cpa = float(np.linalg.norm(closest_offset))
            if raw_time_to_cpa < 0.0:
                time_to_cpa = 0.0
                distance_at_cpa = distance

        tcpa_norm = time_to_cpa / 60.0
        dcpa_norm = float(np.clip(distance_at_cpa / self.diagonal, 0.0, 1.0))
        return closing_speed_norm, tcpa_norm, dcpa_norm

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
        
        t_val = self._encode_plane_type(plane.type)

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
            t_val, # type of plane as a value between 0 and 1
            float(plane.approach_armed), # whether the plane is committed to the current approach
            float(plane.passed_gate), # whether the mandatory approach gate has been satisfied
        ]

        if self.include_wind_in_obs:
            if self.max_wind_speed > 0:
                wx_norm = self.wind_vector[0] / self.max_wind_speed
                wy_norm = self.wind_vector[1] / self.max_wind_speed
            else:
                wx_norm = 0.0
                wy_norm = 0.0
            obs_list.extend([wx_norm, wy_norm])

        # --- NEW: SORT NEIGHBORS BY DISTANCE ---
        active_neighbors = []
        
        # 1. Gather all active neighbors and calculate their distance
        for other_agent in self.possible_agents:
            if other_agent == agent:
                continue
            
            other_plane = self.planes_dict[other_agent]
            if other_plane is not None and other_plane.active:
                dist = math.sqrt((other_plane.x - plane.x)**2 + (other_plane.y - plane.y)**2)
                active_neighbors.append((dist, other_plane))

        # 2. Sort the list ascending (closest planes first)
        active_neighbors.sort(key=lambda x: x[0])

        # 3. Fill the observation array sequentially
        for i in range(self.max_planes - 1):
            if i < len(active_neighbors):
                # Unpack the sorted neighbor
                _, other_plane = active_neighbors[i]
                
                dx = (other_plane.x - plane.x) / self.width 
                dy = (other_plane.y - plane.y) / self.height 
                dv = (other_plane.speed - plane.speed) / (plane.max_speed - plane.min_speed) 
                dhead = other_plane.heading - plane.heading 
                closing_speed_norm, tcpa_norm, dcpa_norm = self._compute_conflict_features(plane, other_plane)
                
                obs_list.extend([
                    dx, 
                    dy, 
                    dv, 
                    math.cos(dhead), 
                    math.sin(dhead), 
                    self._encode_plane_type(other_plane.type),
                    self._encode_destination(other_plane.destination_id),
                    closing_speed_norm,
                    tcpa_norm,
                    dcpa_norm,
                    1.0 # indicator that this slot contains an active plane
                ])
            else:
                # Pad with empty data if there are fewer active planes than slots
                obs_list.extend([-1.0] * self.neighbor_feature_dim)
        # ---------------------------------------

        return np.array(obs_list, dtype=np.float32)

if __name__ == "__main__":
    env = AirTrafficEnv(max_planes=6, spawn_planes=4, enable_acceleration=False, enable_wind=False)
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
