Here is a complete, updated `README.md` file tailored specifically to the latest version of your environment. It accurately reflects the new observation shapes, advanced radar features (like TCPA/DCPA), the multi-runway approach gates, the L1 Manhattan distance progress rewards, and the updated scaling of all penalties and bonuses.

---

# Gym Air Traffic

A custom Multi-Agent Reinforcement Learning (MARL) environment built on the **PettingZoo Parallel API**.

Gym Air Traffic is a 2D physics-based simulation where an AI must guide multiple aircraft to their specific landing zones safely. Agents must learn complex airspace management, including speed control, collision avoidance, approach gate navigation, and precise runway alignment, all while compensating for dynamic wind conditions.

## Features

* **Multi-Agent Control:** Fully compatible with PettingZoo `ParallelEnv` and SuperSuit wrappers for Independent PPO (Parameter Sharing).
* **Advanced "Radar" Observations:** The observation space automatically calculates and sorts neighbor aircraft by distance, and provides advanced conflict metrics like Time to Closest Point of Approach (TCPA) and Distance at Closest Point of Approach (DCPA).
* **Approach Gates & Airspace Rules:** Aircraft must pass through an invisible "approach gate" 100-200 pixels in front of their designated runway. Trespassing in the wrong runway's gate or skipping the gate entirely results in strict penalties.
* **Physics-Based Movement & Smoothness:** Aircraft have inertia, turning rates, and optional acceleration. The environment actively penalizes jerky steering, forcing agents to fly smooth, realistic flight paths.
* **Curriculum-Ready Spawning:** Aircraft spawn at safe distances in designated edge slots to prevent instant, unavoidable collisions upon reset.
* **Diverse Traffic Types:** * **Red Jets:** Must land on the top Red Runway (East-facing).
* **Blue Jets:** Must land on the bottom Blue Runway (East-facing).
* *(Helicopters and Helipads are supported by the physics engine but are currently disabled in the standard spawn loop to focus on runway glide-slope training).*



## Installation

### Prerequisites

* Python 3.8+
* `gymnasium`, `pettingzoo`, `numpy`, `pygame`, `imageio`

### Install via pip (Local)

```bash
git clone https://github.com/TristanDonze/gym-air-traffic.git
cd gym-air-traffic
pip install -e .

```

## Environment Details

### Action Space

The action space is continuous and defined per agent. The dimension depends on the `enable_acceleration` parameter in the constructor.

**If `enable_acceleration=True` (Shape: `(2,)`):**
| Index | Name | Range | Description |
| --- | --- | --- | --- |
| 0 | Steering | `[-1.0, 1.0]` | Relative turn command. Interpolated by the aircraft's internal turn rate. |
| 1 | Throttle | `[-1.0, 1.0]` | Relative acceleration command. Allows the plane to slow down for landing. |

**If `enable_acceleration=False` (Shape: `(1,)`):**
Only the Steering command (Index 0) is available. The aircraft will maintain its initial spawn speed forever.

### Observation Space

The observation space is ego-centric. Its size is calculated dynamically based on `max_planes` and `include_wind_in_obs`.
**Total Shape:** `base_features + ((max_planes - 1) * 11)`

#### Part 1: Ego State (Indices 0 to 12 or 14)

Represents the current agent's state relative to its own specific landing zone:

| Index | Feature | Description |
| --- | --- | --- |
| 0 | `x` | X position (normalized by screen width). |
| 1 | `y` | Y position (normalized by screen height). |
| 2 | `speed` | Current speed (normalized between min and max speed). |
| 3 | `cos(heading)` | Cosine of the current heading angle. |
| 4 | `sin(heading)` | Sine of the current heading angle. |
| 5 | `cos(rel_heading)` | Cosine of the angle between current heading and target zone. |
| 6 | `sin(rel_heading)` | Sine of the angle between current heading and target zone. |
| 7 | `long_dist` | Longitudinal progress distance along the runway's axis (normalized). |
| 8 | `lat_dist` | Lateral cross-track distance from the runway's centerline (normalized). |
| 9 | `cos(target_angle)` | Cosine of the target landing zone angle. |
| 10 | `sin(target_angle)` | Sine of the target landing zone angle. |
| 11 | `type_val` | Plane identifier: 0.0 (Red Jet), 0.5 (Blue Jet). |
| 12 | `passed_gate` | Boolean flag (1.0 or 0.0) indicating if the plane successfully flew through its approach gate. |
| 13 | `wind_x` | *(Optional)* X component of the global wind vector (normalized). |
| 14 | `wind_y` | *(Optional)* Y component of the global wind vector (normalized). |

#### Part 2: Radar State (Distance-Sorted Neighbors)

The remaining values represent the relative states of other aircraft. **Neighbors are strictly sorted by Euclidean distance** so the closest threat is always in the first slot. Each neighbor occupies a block of 11 values:

| Offset | Feature | Description |
| --- | --- | --- |
| +0 | `dx` | Relative X distance to the other plane (normalized). |
| +1 | `dy` | Relative Y distance to the other plane (normalized). |
| +2 | `dv` | Relative speed difference (normalized). |
| +3 | `cos(dhead)` | Cosine of the relative heading difference. |
| +4 | `sin(dhead)` | Sine of the relative heading difference. |
| +5 | `type` | The plane type of the neighbor. |
| +6 | `destination` | The destination ID of the neighbor. |
| +7 | `closing_speed` | How fast the planes are approaching each other (normalized). |
| +8 | `tcpa` | Time to Closest Point of Approach (normalized up to 60 steps). |
| +9 | `dcpa` | Distance at Closest Point of Approach (normalized). |
| +10 | `is_active` | 1.0 if the slot contains an active plane, -1.0 if empty padding. |

### Rewards System

The environment utilizes a heavily shaped dense reward system paired with terminal condition penalties to guide the MARL training process smoothly.

**Dense (Continuous) Rewards:**

* **Time Penalty:** `-0.08` per step to encourage prompt landings.
* **L1 Progress Reward:** Agents are rewarded for closing the longitudinal and lateral gaps to the runway independently. Scaled to `~0.08` per step and clipped at `0.75` to prevent exploit loops.
* **Steering Smoothness:** `-0.015 * steering_effort`. Penalizes jerky left/right movements.
* **Wrong Approach Gate:** `-0.35` per step if flying through an approach gate belonging to a different runway.
* **Glide Slope Alignment:** Grants up to `+0.20` per step for keeping the nose perfectly parallel with the runway once aligned with the centerline.
* **Repulsion Zone:** If two planes get within 150 pixels of each other, they suffer a scaling penalty (up to `-0.6` per step) pushing them to actively maintain airspace separation before a fatal collision occurs.

**Sparse (Event) Rewards & Terminations:**

* **Gate Passed:** `+18.0` (One-time bonus) for perfectly threading the invisible approach box in front of the assigned runway.
* **Successful Touchdown:** `+45.0`. The plane crossed the exact center perpendicular "finish line" of the runway.
* **Hard Landing:** `-15.0`. The plane landed perfectly, but its speed exceeded the safe landing limit (`2.0`).
* **Skipped Gate:** `-15.0`. The plane touched down, but never flew through the mandatory approach gate. (Terminal)
* **Missed Approach:** `-35.0`. The plane flew completely past the back of the runway without touching down. (Terminal)
* **Collision:** `-45.0`. Hit another aircraft (< 30 pixels). Applies to both agents. (Terminal)
* **Out of Bounds:** `-30.0`. The plane flew off the edge of the radar map. (Terminal)

## Usage Example

```python
import os
import numpy as np
from gym_air_traffic.envs.air_traffic_env import AirTrafficEnv

def main():
    # Initialize environment with 2 planes
    env = AirTrafficEnv(
        render_mode="rgb_array", 
        max_planes=2, 
        spawn_planes=2, 
        enable_acceleration=True, 
        enable_wind=False
    )
    
    frames = []
    observations, infos = env.reset()
    
    print(f"Observation dimension: {env.obs_dim}, Action dimension: {env.action_dim}")
    print("Environment reset. Starting simulation...")

    try:
        while env.steps < 1000:
            # Sample random actions for all active agents
            actions = {}
            for agent in env.agents:
                if observations[agent][0] != -1.0: # Check if agent is active
                    actions[agent] = env.action_space(agent).sample()
            
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            # End episode if all agents have landed, crashed, or run out of time
            if all(terminations.values()) or all(truncations.values()):
                print(f"Episode finished at step {env.steps}.")
                break

    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    
    env.save_video("videos", frames, filename="random_agent.mp4", fps=30)
    env.close()

if __name__ == "__main__":
    main()

```
