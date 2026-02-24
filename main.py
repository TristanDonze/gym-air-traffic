import gymnasium as gym
import gym_air_traffic
import numpy as np
import os

def main():
    env = gym.make("AirTraffic-v0", render_mode="rgb_array")
    
    frames = []
    observation, info = env.reset()
    print("Environment reset. Starting simulation...")

    try:
        while True:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            print(reward)

            if terminated or truncated:
                print("Episode finished.")
                break

    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    
    print(f"End of simulation. Captured {len(frames)} frames.")
    env.unwrapped.save_video("videos", frames, filename="simulation.mp4", fps=30)
    env.close()

if __name__ == "__main__":
    main()