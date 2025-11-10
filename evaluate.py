import os
import time
import numpy as np
from stable_baselines3 import PPO
from rl.real_env import RealMetricsEnv

PROM_URL = os.getenv("PROM_URL", "http://localhost:9090")
STEP_SECONDS = int(os.getenv("STEP_SECONDS", 3))
STEPS = int(os.getenv("STEPS", 100))  # how many steps to simulate
MODEL_PATH = "rl/artifacts/ppo_edgecloud_finetuned"

def main():
    env = RealMetricsEnv(PROM_URL, step_seconds=STEP_SECONDS)

    # load the trained model
    model = PPO.load(MODEL_PATH, env=env, device="cpu")

    obs, _ = env.reset()
    for step in range(STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        print(f"[step {step}] action={action}, reward={reward:.3f}, obs={obs}")

        if done or truncated:
            obs, _ = env.reset()
        time.sleep(STEP_SECONDS)

if __name__ == "__main__":
    main()