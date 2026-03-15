# Gym Air Traffic

A custom Multi-Agent Reinforcement Learning environment compatible with the PettingZoo Parallel API.

Gym Air Traffic is a 2D simulation where an agent must guide multiple aircraft to their specific landing zones while avoiding collisions, managing speed, compensating for dynamic wind conditions, and adhering to strict runway traffic control.

## Features

* Multi-Agent Control: Built on the PettingZoo ParallelEnv.
* World Model & MPC Ready: Exposes global absolute states and static environment variables alongside standard ego-centric observations.
* Configurable Difficulty: Customize the maximum number of planes, and toggle acceleration or wind dynamics to scale the complexity of the task.
* Diverse Traffic:
* Red Jets: Must land on the Red Runway (East-facing).
* Blue Jets: Must land on the Blue Runway (East-facing).
* Helicopters: Must land on the Helipad (Omnidirectional). Limited to a maximum of 1 per episode.


* Physics-Based Movement: Aircraft have inertia, turning rates, and optional acceleration and wind drift mechanics.
* Strict Airspace Control: Features approach gates and a runway locking mechanism. Runways are exclusively locked to the first aircraft that correctly aligns with the approach gate, penalizing other aircraft that attempt to cut in line.
* One-Time Spawning: Aircraft all spawn at the beginning of the episode. Once an aircraft lands, crashes, or flies out of bounds, it is permanently disabled for the remainder of the episode.
* Strict Termination Masking: Unspawned and destroyed aircraft continuously report a terminal state.

## Installation

### Prerequisites

* Python 3.8+
* gymnasium, pettingzoo, numpy, pygame, imageio

### Install via pip (Local)

```bash
git clone https://github.com/TristanDonze/gym-air-traffic.git
cd gym-air-traffic
pip install -e .

```

### Install directly from GitHub

```bash
pip install git+https://github.com/TristanDonze/gym-air-traffic.git

```

## Environment Details

### Action Space

The action space is continuous and defined per agent. The dimension depends on the `enable_acceleration` parameter in the environment constructor.

If `enable_acceleration=True` (Shape: `(2,)`):
| Index | Name | Range | Description |
| --- | --- | --- | --- |
| 0 | Steering | [-1.0, 1.0] | Relative turn command. Multiplied by the aircraft's internal turn rate. |
| 1 | Throttle | [-1.0, 1.0] | Relative acceleration command. Multiplied by the aircraft's internal acceleration rate. |

If `enable_acceleration=False` (Shape: `(1,)`):
Only the Steering command (Index 0) is available. The aircraft will maintain its initial spawn speed.

### Observation Spaces

The environment provides two distinct ways to observe the state, depending on your training architecture.

#### 1. Ego-Centric Observation (RL Standard)

Its size is calculated dynamically based on the `max_planes` and `enable_wind` parameters: `base_features + ((max_planes - 1) * 6)`.
`base_features` is 14 if `enable_wind=True`, and 12 if `enable_wind=False`.

**Part 1: Ego State (Indices 0 to 9 or 11)**

| Index | Feature | Description |
| --- | --- | --- |
| 0 | x | X position (normalized by screen width). |
| 1 | y | Y position (normalized by screen height). |
| 2 | speed | Current speed (normalized between min and max speed). |
| 3 | cos(heading) | Cosine of the current heading angle. |
| 4 | sin(heading) | Sine of the current heading angle. |
| 5 | cos(rel_heading) | Cosine of the angle between current heading and target zone. |
| 6 | sin(rel_heading) | Sine of the angle between current heading and target zone. |
| 7 | longitudinal_norm | Longitudinal distance projected onto the landing zone's angle (normalized). |
| 8 | lateral_norm | Lateral (cross-track) distance projected onto the landing zone's angle (normalized). |
| 9 | cos(target_angle) | Cosine of the target landing zone angle. |
| 10 | sin(target_angle) | Sine of the target landing zone angle. |
| 11 | type_id | 0.0 (Red Jet), 0.5 (Blue Jet), 1.0 (Helicopter). |
| 12 | wind_x | Optional: X component of the global wind vector (normalized). |
| 13 | wind_y | Optional: Y component of the global wind vector (normalized). |

**Part 2: Radar State**
Each other aircraft occupies a block of 6 values:

| Offset | Feature | Description |
| --- | --- | --- |
| +0 | dx | Relative X distance to the other plane (normalized). |
| +1 | dy | Relative Y distance to the other plane (normalized). |
| +2 | dv | Relative speed difference (normalized). |
| +3 | cos(dhead) | Cosine of the relative heading difference. |
| +4 | sin(dhead) | Sine of the relative heading difference. |
| +5 | is_active | 1.0 if the slot contains an active plane, -1.0 if inactive. |

#### 2. Global State (World Model / MPC)

Accessible via `env.get_mpc_state()` and `env.get_mpc_zones()`. This provides raw, unnormalized global coordinates suitable for planning algorithms.

* **`get_mpc_state()`**: Returns a flat array containing the absolute state of every agent slot `[x, y, speed, heading, active_status, destination_id]`, followed by the global `[wind_x, wind_y]` if enabled.
* **`get_mpc_zones()`**: Returns a dictionary mapping zone IDs to their static properties `[x, y, angle, is_helipad]`.

### Rewards

Rewards are distributed individually to each agent based on movement, strict traffic control, and terminal conditions.

**Dense Rewards (per step):**

* Time Penalty: `-0.01`. Applied to every active aircraft at every step.
* Distance Reduction: `+0.1 * (distance_before - distance_after)`. Rewards the agent for moving closer to its specific landing zone.
* Cross-Track Penalty: `-0.05 * (abs(lateral_dist) / 100.0)`. Applied when approaching the runway from the front to discourage flying too far off the center line.
* Approach Funnel Alignment: `+0.15 * alignment_bonus`. Applied when the agent is in front of the runway and its lateral distance is less than 50 pixels.
* Approach Gate Violation: `-2.0`. Applied continuously if the aircraft flies into the approach gate of the wrong landing zone.
* Traffic Cut-In Penalty: `-5.0`. Applied if an aircraft enters its correct approach gate, but the runway is already locked by another landing aircraft.
* Off-Course Maneuvering: `-0.5`. Applied continuously if the aircraft flies past the runway but has not locked the approach gate yet.

**Sparse / Event Rewards:**

* Approach Gate Success: `+150.0`. Granted once per episode when the aircraft successfully passes through its correct approach gate while the runway is clear. This locks the runway for this agent.

**Terminal Rewards:**

* Landing Success: `+500.0`. The agent reached its zone with strict alignment and speed below the landing limit.
* Landing Speed Penalty: `-50.0`. The agent reached the correct zone and alignment, but was flying too fast.
* True Missed Approach: `-200.0`. The agent passed the approach gate but failed to land, flying past the runway. This terminates the episode for the agent.
* Out of Bounds: `-200.0`. The agent left the screen boundaries.
* Collision: `-500.0`. The agent collided with another aircraft. Applies to both involved agents.

## Usage Example

```python
import os
import numpy as np
from gym_air_traffic.envs.air_traffic_env import AirTrafficEnv

def main():
    env = AirTrafficEnv(render_mode="rgb_array", max_planes=3, enable_acceleration=False, enable_wind=False)
    print(f"Observation dimensions: {env.obs_dim}, Action dimensions: {env.action_dim}")
    
    frames = []
    observations, infos = env.reset()
    i = 0
    actual_return = 0.0
    
    print("Environment reset. Starting simulation...")

    actived = {f"plane_{idx}": False for idx in range(env.max_planes)}

    try:
        while env.steps < 1000:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            step_reward = sum(rewards.values()) if rewards else 0.0
            actual_return += step_reward
            
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            for obs_id, obs in observations.items():
                if not actived[obs_id] and obs.flatten()[0] != -1.0:
                    print(f"Agent has just become active at step {i}:\n{obs_id}: Shape {obs.shape}, Sample values: {obs.flatten()[:5]}")
                    actived[obs_id] = True
            
            for obs_id, obs in observations.items():
                if actived[obs_id] and obs.flatten()[0] == -1.0:
                    print(f"{obs_id} has just become inactive at step {i}:\n: Shape {obs.shape}, Sample values: {obs.flatten()[:5]}")
                    actived[obs_id] = False
            i += 1
            
        print("Episode finished.")

    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    
    print(f"End of simulation. Captured {len(frames)} frames.")
    env.save_video("videos", frames, filename="simulation.mp4", fps=30)
    env.close()

if __name__ == "__main__":
    main()