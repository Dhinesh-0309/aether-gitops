# rl/finetune_online.py
import os, time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from rl.real_env import RealMetricsEnv

ART_DIR = os.environ.get("ART_DIR", "artifacts")
BASE = os.path.join(ART_DIR, "ppo_edgecloud.zip")
OUT = os.path.join(ART_DIR, "ppo_edgecloud_finetuned.zip")
PROM = os.environ.get("PROM_URL", "http://localhost:9090")

STEP_SECONDS = int(os.environ.get("STEP_SECONDS", "3"))
TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", "30000"))

def main():
    print("[finetune] starting online finetune... (PROM=%s)" % PROM)
    env = RealMetricsEnv(prom_url=PROM, step_seconds=STEP_SECONDS, max_steps=999999)
    env = Monitor(env)

    if os.path.exists(BASE):
        print("[finetune] loading base model:", BASE)
        model = PPO.load(BASE, env=env, device="cpu")
    else:
        print("[finetune] base not found, training fresh")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-5)

    logdir = os.path.join("rl", "logs", "ppo_finetune")
    os.makedirs(logdir, exist_ok=True)
    new_logger = configure(logdir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    print(f"[finetune] learning {TOTAL_STEPS} steps (step={STEP_SECONDS}s)...")
    model.learn(total_timesteps=TOTAL_STEPS)
    model.save(OUT)
    print("[finetune] saved:", OUT)

if __name__ == "__main__":
    main()