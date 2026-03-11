# Gym Air Traffic

A custom Multi-Agent Reinforcement Learning environment compatible with the PettingZoo Parallel API.

Gym Air Traffic is a 2D simulation where an agent must guide multiple aircraft (Jets and Helicopters) to their specific landing zones while avoiding collisions, managing speed, and compensating for dynamic wind conditions.

## Features

* Multi-Agent Control: Built on the PettingZoo ParallelEnv.
* Configurable Difficulty: Customize the maximum number of planes, and toggle acceleration or wind dynamics to scale the complexity of the task.
* Diverse Traffic:
    * Red Jets: Must land on the Red Runway (East-facing).
    * Blue Jets: Must land on the Blue Runway (East-facing).
    * Helicopters: Must land on the Helipad (Omnidirectional). Limited to a maximum of 1 per episode.


* Physics-Based Movement: Aircraft have inertia, turning rates, and optional acceleration and wind drift mechanics.
* One-Time Spawning: Aircraft all spawn at the beginning of the episode (checking for safe distances). Once an aircraft lands, crashes, or flies out of bounds, it is permanently disabled for the remainder of the episode.
* Strict Termination Masking: Unspawned and destroyed aircraft continuously report a terminal state. This ensures compatibility with PettingZoo wrappers like `black_death_v3` to mask inactive agents during training.
* Ego-Centric Observations: Each agent observes the environment from its own perspective. Inactive or dead agents are masked to maintain tensor stability.

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

### Observation Space

The observation space is ego-centric. Its size is calculated dynamically based on the `max_planes` and `enable_wind` parameters: `base_features + ((max_planes - 1) * 6)`.

* `base_features` is 14 if `enable_wind=True`, and 12 if `enable_wind=False`.
* If a plane slot is inactive (has not spawned yet, or has already terminated), its entire observation vector is filled with `-1.0`.

**Part 1: Ego State (Indices 0 to 9 or 11)**
Represents the current agent's state and target:

| Index | Feature | Description |
| --- | --- | --- |
| 0 | x | X position (normalized by screen width). |
| 1 | y | Y position (normalized by screen height). |
| 2 | speed | Current speed (normalized between min and max speed). |
| 3 | cos(heading) | Cosine of the current heading angle. |
| 4 | sin(heading) | Sine of the current heading angle. |
| 5 | cos(rel_heading) | Cosine of the angle between current heading and target zone. |
| 6 | sin(rel_heading) | Sine of the angle between current heading and target zone. |
| 7 | dx_target | Relative X distance to the landing zone (normalized). |
| 8 | dy_target | Relative Y distance to the landing zone (normalized). |
| 9 | cos(target_angle) | Cosine of the target landing zone angle. |
| 10 | sin(target_angle) | Sine of the target landing zone angle. |
| 11 | type_id | 0.0 (Red Jet), 0.5 (Blue Jet), 1.0 (Helicopter). |
| 12 | wind_x | Optional: X component of the global wind vector (normalized). |
| 13 | wind_y | Optional: Y component of the global wind vector (normalized). |

**Part 2: Radar State**
The remaining values represent the relative states of the other potential aircraft in the environment. Each other aircraft occupies a block of 6 values:

| Offset | Feature | Description |
| --- | --- | --- |
| +0 | dx | Relative X distance to the other plane (normalized). |
| +1 | dy | Relative Y distance to the other plane (normalized). |
| +2 | dv | Relative speed difference (normalized). |
| +3 | cos(dhead) | Cosine of the relative heading difference. |
| +4 | sin(dhead) | Sine of the relative heading difference. |
| +5 | is_active | 1.0 if the slot contains an active plane, -1.0 if inactive. |

### Rewards

The environment uses a combination of sparse terminal rewards and dense shaping rewards. Rewards are distributed individually to each agent.

### Rewards

The environment uses a combination of sparse terminal rewards and dense shaping rewards. Rewards are distributed individually to each agent.

**Dense Rewards (per step):**

* Time Penalty: `-0.15`. Applied to every active aircraft at every step to encourage fast task completion and prevent infinite hovering.
* Distance Reduction: `+0.1 * (distance_before - distance_after)`. Rewards the agent for moving closer to its specific landing zone.
* Approach Funnel Alignment: `+0.2 * alignment_bonus`. Applied when the agent is generally in front of the runway (dot product > 0.9). Rewards the agent linearly for minimizing its lateral distance to the runway's extended center line.
* Runway Heading Alignment: `+0.1 * heading_bonus`. Applied in the approach funnel, linearly rewarding the agent for matching the exact angle of the runway.

**Terminal Rewards:**

* Landing Success: `+500.0`. The agent reached its zone with strict alignment (within 20 pixels, < 0.15 radians off-axis) and speed below the landing limit.
* Landing Speed Penalty: `-50.0`. The agent reached the correct zone and alignment, but was flying too fast (only applies if acceleration is enabled).
* Out of Bounds: `-200.0`. The agent left the screen boundaries.
* Collision: `-500.0`. The agent collided with another aircraft (distance < 30 pixels). Applies to both involved agents.

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

            # if i % 100 == 0:
            #     print(f"Step {i}, Return: {actual_return:.2f}, Active agents: {len(env.nb_active_agents)}")
            #     for obs_id, obs in observations.items():
            #         print(f"  Agent {obs_id}: Shape {obs.shape}, Sample values: {obs.flatten()[:5]}")
            
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
```