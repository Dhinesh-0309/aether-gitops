# controller/controller.py
import os, time, redis, psutil
import numpy as np
from collections import deque

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))

STEP_SECONDS = int(os.environ.get("STEP_SECONDS", "3"))
COOLDOWN = 30  # seconds before switching back
K = 3          # consecutive predictions to confirm
EMERGENCY_CPU = 98.0
CPU_THRESHOLD = 80.0
MEM_THRESHOLD = 80.0

def get_system_metrics():
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    load1 = os.getloadavg()[0]
    disk = psutil.disk_io_counters().read_bytes / (1024 * 1024)
    net = psutil.net_io_counters().bytes_sent / (1024 * 1024)
    return np.array([cpu, mem, load1, disk, net], dtype=np.float32)

def main():
    print("[AETHER-BASELINE] starting…")
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.ping(); print("[ok] Redis")
    except Exception as e:
        print(f"[error] Redis connection failed: {e}")
        return

    window, current_target, cooldown_until = deque(maxlen=K), "EDGE", 0

    while True:
        try:
            cpu, mem, load1, disk, net = get_system_metrics()
            tnow = time.time()
            decision = current_target

            # --- simple heuristic rules ---
            if cpu > EMERGENCY_CPU or (cpu > CPU_THRESHOLD or mem > MEM_THRESHOLD):
                window.append("CLOUD")
            else:
                window.append("EDGE")

            if len(window) == K and tnow >= cooldown_until:
                if all(a == "CLOUD" for a in window):
                    decision = "CLOUD"
                elif all(a == "EDGE" for a in window):
                    decision = "EDGE"
                if decision != current_target:
                    current_target = decision
                    cooldown_until = tnow + COOLDOWN

            print(f"[baseline] cpu={cpu:.1f} mem={mem:.1f} load1={load1:.2f} -> {decision}")
            r.set("aether:target", decision.lower())

            time.sleep(STEP_SECONDS)
        except KeyboardInterrupt:
            print("\n[exit] stopping baseline controller…")
            break
        except Exception as e:
            print(f"[loop] error: {e}")
            time.sleep(STEP_SECONDS)

if __name__ == "__main__":
    main()