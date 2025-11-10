# rl/train_surrogate.py
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from stable_baselines3 import PPO
from env_edge_cloud import EdgeCloudEnv

MODEL_PATH = "artifacts/ppo_edgecloud.zip"
SURROGATE_PATH = "artifacts/surrogate.pkl"

def collect_rollouts(n_steps=5000):
    env = EdgeCloudEnv()
    model = PPO.load(MODEL_PATH, env=env)
    obs, _ = env.reset()
    X, y = [], []
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        X.append(obs)
        y.append(action)
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            obs, _ = env.reset()
    return np.array(X), np.array(y)

def main():
    X, y = collect_rollouts()
    clf = LogisticRegression(max_iter=500).fit(X, y)
    joblib.dump(clf, SURROGATE_PATH)
    print(f"[surrogate] saved to {SURROGATE_PATH}")

if __name__ == "__main__":
    main()