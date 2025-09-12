from __future__ import annotations

try:
    import gymnasium as gym
    GYMN = "gymnasium"
except Exception:
    import gym
    GYMN = "gym"

def make_taxi_env(seed: int = 0):
    env = gym.make("Taxi-v3")
    try:
        env.reset(seed=seed)
    except TypeError:
        # older gym
        env.seed(seed)
    return env