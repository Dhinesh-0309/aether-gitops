import os
import time
import psutil
import yaml
import numpy as np
from github import Github
from dotenv import load_dotenv
from stable_baselines3 import PPO
from collections import deque

# ---- Load env + config ----
load_dotenv()
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "gitops_config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

GITHUB_TOKEN = os.getenv(config["github"]["token_env"])
REPO_NAME = config["github"]["repo"]
BRANCH = config["github"]["branch"]
MANIFEST_PATH = config["github"]["manifest_path"]

STEP_SEC = config["controller"]["step_seconds"]
VOTES_N = config["controller"]["votes_to_switch"]
CLOUD_MIN_SEC = config["controller"]["cloud_min_sec"]
EDGE_MIN_SEC = config["controller"]["edge_min_sec"]
PROVISION_DELAY_SEC = config["controller"]["provision_delay_sec"]

MODEL_PATH = os.path.join("artifacts", "ppo_edgecloud.zip")

# ---- System caps ----
MAX_CPU = 100.0
MAX_MEM = 100.0
MAX_LOAD1 = 32.0
MAX_IO = 500.0

def norm(x, cap): return float(max(0.0, min(1.0, x / cap)))

# ---- Metrics Collector ----
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

        cur = np.array([cpu, mem, load1, min(disk_mb, MAX_IO), min(net_mb, MAX_IO)], dtype=np.float32)
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

# ---- GitOps helper ----
class GitOpsUpdater:
    def __init__(self, token, repo_name):
        self.gh = Github(token)
        self.repo = self.gh.get_repo(repo_name)

    def update_node_selector(self, branch, path, new_target):
        file = self.repo.get_contents(path, ref=branch)
        content = file.decoded_content.decode()
        data = yaml.safe_load(content)

        # Modify nodeSelector
        tmpl = data.get("spec", {}).get("template", {}).get("spec", {})
        if "nodeSelector" not in tmpl:
            tmpl["nodeSelector"] = {}
        tmpl["nodeSelector"]["location"] = new_target

        new_yaml = yaml.dump(data, sort_keys=False)
        if new_yaml != content:
            commit_msg = f"[AETHER] Switch deployment to {new_target.upper()}"
            self.repo.update_file(file.path, commit_msg, new_yaml, file.sha, branch=branch)
            print(f"[gitops] ✅ updated manifest to {new_target.upper()}")
        else:
            print("[gitops] no changes detected")

# ---- Main controller ----
def main():
    print("[AETHER-GITOPS] starting controller…")

    try:
        model = PPO.load(MODEL_PATH, device="cpu")
        print(f"[ok] PPO loaded from {MODEL_PATH}")
    except Exception as e:
        print("[error] loading model:", e)
        return

    gitops = GitOpsUpdater(GITHUB_TOKEN, REPO_NAME)
    mc = MetricsCollector()

    last_decision = "edge"
    last_switch_time = time.time()
    votes = deque(maxlen=VOTES_N)
    provision_start = None
    cloud_provisioned = False

    while True:
        try:
            raw, obs = mc.read()
            action, _ = model.predict(obs, deterministic=True)
            suggest = "edge" if int(action) == 0 else "cloud"
            votes.append(suggest)

            cpu, mem, load1 = raw[0], raw[1], raw[2]
            now = time.time()

            if len(votes) == VOTES_N and all(v == suggest for v in votes):
                if suggest != last_decision:
                    time_since = now - last_switch_time
                    if suggest == "cloud":
                        if not cloud_provisioned:
                            if (provision_start is None or now - provision_start >= PROVISION_DELAY_SEC):
                                gitops.update_node_selector(BRANCH, MANIFEST_PATH, "cloud")
                                last_decision = "cloud"
                                last_switch_time = now
                                cloud_provisioned = True
                    else:
                        if time_since >= CLOUD_MIN_SEC:
                            gitops.update_node_selector(BRANCH, MANIFEST_PATH, "edge")
                            last_decision = "edge"
                            last_switch_time = now
                            cloud_provisioned = False

            print(f"[drl] cpu={cpu:.1f} mem={mem:.1f} load1={load1:.2f} -> {last_decision.upper()} (suggest={suggest}, votes={list(votes)})")

            time.sleep(STEP_SEC)
        except KeyboardInterrupt:
            print("\n[exit] stopping controller…")
            break
        except Exception as e:
            print("[loop] error:", e)
            time.sleep(2)

if __name__ == "__main__":
    main()