# Gym Air Traffic

A custom Reinforcement Learning environment compatible with [Farama Gymnasium](https://gymnasium.farama.org/).

**Gym Air Traffic** is a 2D simulation where an agent must guide multiple aircraft (Jets and Helicopters) to their specific landing zones while avoiding collisions, managing speed, and compensating for dynamic wind conditions.

## Features

* **Multi-Agent Control:** Control up to 10 aircraft simultaneously.
* **Diverse Traffic:**
* 🔴 **Red Jets:** Must land on the Red Runway (East-facing).
* 🔵 **Blue Jets:** Must land on the Blue Runway (East-facing).
* 🚁 **Helicopters:** Must land on the Helipad (Omnidirectional).


* **Physics-Based Movement:** Aircraft have inertia, turning rates, and acceleration/deceleration mechanics.
* **Dynamic Environment:** Randomized spawn points and changing wind vectors that affect flight trajectories.
* **Continuous Action Space:** Precise control over heading and throttle.

## Installation

### Prerequisites

* Python 3.8+
* `gymnasium`, `numpy`, `pygame`, `imageio`

### Install via pip (Local)

To install the environment in editable mode (useful for development):

```bash
git clone https://github.com/TristanDonze/gym-air-traffic.git
cd gym-air-traffic
pip install -e .

```

### Install directly from GitHub

To install it directly into your python environment without cloning manually:

```bash
pip install git+https://github.com/TristanDonze/gym-air-traffic.git

```

> **Note:** To save videos (MP4), ensure you have `ffmpeg` installed on your system or install imageio with ffmpeg:
> `pip install "imageio[ffmpeg]"`

## Environment Details

### Action Space

The action space is **Continuous**.
`Box(-pi, pi, (Max_Planes, 2), float32)`

The agent provides a matrix of shape `(N, 2)` where `N` is the maximum number of planes (default: 10). For each aircraft slot $i$:

| Index | Name | Range | Description |
| --- | --- | --- | --- |
| 0 | **Heading** | $[-\pi, \pi]$ | The target angle the aircraft should turn towards. |
| 1 | **Throttle** | $[-1, 1]$ | Acceleration command. $-1$ is max braking, $1$ is max thrust. |

### Observation Space
The observation space is a matrix of shape `(N, 11)`.
`Box(-inf, inf, (Max_Planes, 11), float32)`

Each row represents the state of a potential aircraft slot:

| Index | Feature | Description |
| :--- | :--- | :--- |
| 0 | `x` | X position (normalized by screen width). |
| 1 | `y` | Y position (normalized by screen height). |
| 2 | `speed` | Current speed (normalized). |
| 3 | `cos(heading)` | Cosine of the current heading angle. |
| 4 | `sin(heading)` | Sine of the current heading angle. |
| 5 | `target_x` | X coordinate of the assigned landing zone (normalized). |
| 6 | `target_y` | Y coordinate of the assigned landing zone (normalized). |
| 7 | `type_id` | `0.0` (Red Jet), `0.5` (Blue Jet), `1.0` (Helicopter). |
| 8 | `wind_x` | X component of the global wind vector (normalized). |
| 9 | `wind_y` | Y component of the global wind vector (normalized). |
| 10 | `is_active` | `1.0` if the slot contains a plane, `0.0` otherwise. |

### Rewards

The goal is to maximize the score by landing planes quickly and safely.

* **Landing Success:** `+150` (Correct zone, correct alignment, speed < limit).
* **Crash (Mid-air):** `-100` (Collision between two aircraft).
* **Crash (Landing):** `-50` (Landing speed too high or bad alignment).
* **Out of Bounds:** `-50` (Leaving the screen area).
* **Time Penalty:** `-1.0` per step (Encourages efficiency).

## Usage Example

Here is a basic script to run the environment with random actions and save a video of the episode.

```python
import gymnasium as gym
import gym_air_traffic
import numpy as np
import os

def main():
    # Create the environment with rgb_array mode to capture frames
    env = gym.make("AirTraffic-v0", render_mode="rgb_array")
    
    frames = []
    observation, info = env.reset()
    print("Environment reset. Starting simulation...")

    try:
        while True:
            # Sample random actions
            action = env.action_space.sample()
            
            # Step the environment
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Render the current frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            # Print reward for debugging
            # print(reward)

            if terminated or truncated:
                print("Episode finished.")
                break

    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    
    print(f"End of simulation. Captured {len(frames)} frames.")
    
    # Save the episode as an MP4 video
    # Note: Requires 'imageio[ffmpeg]' installed
    env.unwrapped.save_video("videos", frames, filename="simulation.mp4", fps=30)
    
    env.close()

if __name__ == "__main__":
    main()

```