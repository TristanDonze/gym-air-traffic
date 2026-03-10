import os
import numpy as np
from gym_air_traffic.envs.air_traffic_env import AirTrafficEnv

def main():
    env = AirTrafficEnv(render_mode="rgb_array")
    
    frames = []
    observations, infos = env.reset()
    i = 0
    actual_return = 0.0
    
    print("Environment reset. Starting simulation...")

    try:
        while env.steps < 1000:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            step_reward = sum(rewards.values()) if rewards else 0.0
            actual_return += step_reward
            
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            if i % 100 == 0:
                print(f"Step {i}, Return: {actual_return:.2f}, Active agents: {len(env.agents)}")
                for obs_id, obs in observations.items():
                    print(f"  Agent {obs_id}: Shape {obs.shape}, Sample values: {obs.flatten()[:5]}")
            
            i += 1
            
        print("Episode finished.")

    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    
    print(f"End of simulation. Captured {len(frames)} frames.")
    env.save_video("videos", frames, filename="simulation.mp4", fps=30)
    env.close()

if __name__ == "__main__":
    main()