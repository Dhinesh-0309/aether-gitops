import os, time, redis, psutil
import numpy as np
from stable_baselines3 import PPO
from collections import deque

MODEL_PATH = os.path.join("artifacts", "ppo_edgecloud.zip")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
STEP_SEC = int(os.environ.get("STEP_SECONDS", "3"))

# --- system caps ---
MAX_CPU = 100.0
MAX_MEM = 100.0
MAX_LOAD1 = 32.0   # tuned for 8 cores
MAX_IO = 500.0

# --- cooldowns ---
# ---- cooldowns / provisioning (DEMO values) ----
CLOUD_MIN_SEC = int(os.environ.get("CLOUD_MIN_SEC", "10"))   # was 120+ in real
EDGE_MIN_SEC = int(os.environ.get("EDGE_MIN_SEC", "5"))      # was 60+ in real
PROVISION_DELAY_SEC = int(os.environ.get("PROVISION_DELAY_SEC", "3"))  # simulate pod start

def norm(x, cap): return float(max(0.0, min(1.0, x / cap)))

class MetricsCollector:
    def __init__(self): self.prev = None

    def read(self):
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        try:
            load1 = os.getloadavg()[0]
        except Exception:
            load1 = cpu / 10.0

        io = psutil.disk_io_counters()
        net = psutil.net_io_counters()
        disk_mb = (io.read_bytes + io.write_bytes) / (1024 * 1024)
        net_mb = (net.bytes_sent + net.bytes_recv) / (1024 * 1024)

        cur = np.array([cpu, mem, load1,
                        min(disk_mb, MAX_IO),
                        min(net_mb, MAX_IO)], dtype=np.float32)

        if self.prev is None:
            self.prev = cur.copy()

        obs = np.concatenate([cur, self.prev]).astype(np.float32)
        self.prev = cur.copy()

        obs_norm = np.array([
            norm(obs[0], MAX_CPU), norm(obs[1], MAX_MEM), norm(obs[2], MAX_LOAD1),
            norm(obs[3], MAX_IO), norm(obs[4], MAX_IO),
            norm(obs[5], MAX_CPU), norm(obs[6], MAX_MEM), norm(obs[7], MAX_LOAD1),
            norm(obs[8], MAX_IO), norm(obs[9], MAX_IO)
        ], dtype=np.float32)

        return cur, obs_norm

def main():
    print("[AETHER-DRL] starting controller…")
    rds = redis.from_url(REDIS_URL)
    try:
        rds.ping(); print("[ok] Redis connected")
    except Exception as e:
        print("[error] Redis:", e); return

    try:
        model = PPO.load(MODEL_PATH, device="cpu")
        print(f"[ok] PPO loaded: {MODEL_PATH}")
    except Exception as e:
        print("[error] loading model:", e); return

    mc = MetricsCollector()
    last_decision = "EDGE"
    last_switch_time = time.time()
    cloud_provisioned = False
    provision_start = None

    votes = deque(maxlen=3)

    while True:
        try:
            raw, obs = mc.read()
            action, _ = model.predict(obs, deterministic=True)
            suggest = "EDGE" if int(action) == 0 else "CLOUD"
            votes.append(suggest)

            now = time.time()
            decision = last_decision

            # only switch if all votes agree
            if len(votes) == 3 and all(v == suggest for v in votes):
                if suggest != last_decision:
                    time_since = now - last_switch_time
                    if suggest == "CLOUD":
                        if (not cloud_provisioned and
                            (provision_start is None or now - provision_start >= PROVISION_DELAY_SEC)):
                            decision = "CLOUD"
                            cloud_provisioned = True
                            last_switch_time = now
                    else:  # EDGE
                        if time_since >= CLOUD_MIN_SEC:
                            decision = "EDGE"
                            cloud_provisioned = False
                            last_switch_time = now

            last_decision = decision
            rds.set("aether:target", decision.upper())

            cpu, mem, load1 = raw[0], raw[1], raw[2]
            print(f"[drl] cpu={cpu:.1f} mem={mem:.1f} load1={load1:.2f} "
                  f"-> {decision} (suggest={suggest}, vote={list(votes)})")

            time.sleep(STEP_SEC)
        except KeyboardInterrupt:
            print("\n[exit] stopping controller…"); break
        except Exception as e:
            print("[loop] error:", e); time.sleep(2)

if __name__ == "__main__":
    main()