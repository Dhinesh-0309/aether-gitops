# rl/evaluate.py
import numpy as np
from stable_baselines3 import PPO
from env_edge_cloud import EdgeCloudEnv

MODEL_PATH = "artifacts/ppo_edgecloud.zip"

def evaluate(n_episodes=50):
    env = EdgeCloudEnv()
    model = PPO.load(MODEL_PATH, env=env)
    sla_violations, cloud_time, switches = 0, 0, 0
    total_steps = 0
    prev_action = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, _, _ = env.step(action)
            done = terminated
            cpu, mem = obs[0]*100, obs[1]*100
            if action == 0 and (cpu > 95 or mem > 92):
                sla_violations += 1
            if action == 1:
                cloud_time += 1
            if action != prev_action:
                switches += 1
            prev_action = action
            total_steps += 1

    print(f"SLA Violations: {sla_violations/total_steps:.2%}")
    print(f"Cloud Usage: {cloud_time/total_steps:.2%}")
    print(f"Switches per 1000 steps: {switches/total_steps*1000:.1f}")

if __name__ == "__main__":
    evaluate()