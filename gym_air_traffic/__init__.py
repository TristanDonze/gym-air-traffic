from gymnasium.envs.registration import register

register(
    id="AirTraffic-v0",
    entry_point="gym_air_traffic.envs:AirTrafficEnv",
    max_episode_steps=1000
)