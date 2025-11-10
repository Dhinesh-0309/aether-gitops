import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from env_edge_cloud import EdgeCloudEnv

TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", 50000))
LOG_DIR = os.path.join("rl", "logs", "ppo")
MODEL_PATH = os.path.join("artifacts", "ppo_edgecloud.zip")

def make_env():
    return EdgeCloudEnv()

def train():
    env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", env, verbose=1,
                tensorboard_log=LOG_DIR,
                device="cpu",
                learning_rate=3e-4,
                n_steps=512,
                batch_size=64,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2)

    logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    model.learn(total_timesteps=TOTAL_STEPS)
    model.save(MODEL_PATH)
    print(f"[ok] Model trained and saved at {MODEL_PATH}")

if __name__ == "__main__":
    train()