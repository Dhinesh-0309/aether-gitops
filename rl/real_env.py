# rl/real_env.py
import os, time, requests, numpy as np
import gymnasium as gym
from gymnasium import spaces


MAX_CPU = 100.0
MAX_MEM = 100.0
MAX_LOAD1 = 30.0     
MAX_IO = 1e3

def _norm(x, cap):
    return float(max(0.0, min(1.0, x / cap)))

class RealMetricsEnv(gym.Env):
    """
    Wraps real Prometheus metrics (sampled online) into a Gym environment
    Observation: 10-D same as synthetic env (current + previous).
    Action: Discrete(2) advisory (0=edge,1=cloud)
    """
    metadata = {"render.modes": []}

    def __init__(self, prom_url=None, step_seconds=3, max_steps=600):
        super().__init__()
        self.prom = prom_url or os.environ.get("PROM_URL", "http://localhost:9090")
        self.step_seconds = int(step_seconds)
        self.max_steps = int(max_steps)
        self.steps = 0

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.prev = None

        # prom queries (node exporter)
        self.q_cpu = '100 - (avg by(instance)(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)'
        self.q_mem = '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100'
        self.q_load1 = 'node_load1'
        self.q_disk_r = 'sum by(instance)(irate(node_disk_read_bytes_total[1m]))'
        self.q_net_tx = 'sum by(instance)(irate(node_network_transmit_bytes_total{device!~"lo"}[1m]))'

    def _q(self, expr):
        try:
            r = requests.get(f"{self.prom}/api/v1/query", params={"query": expr}, timeout=5)
            r.raise_for_status()
            data = r.json().get("data", {}).get("result", [])
            return float(data[0]["value"][1]) if data else 0.0
        except Exception:
            return 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        cpu = self._q(self.q_cpu)
        mem = self._q(self.q_mem)
        load = self._q(self.q_load1)
        disk = self._q(self.q_disk_r) / 1024.0  # convert bytes/s -> KB/s approx
        net = self._q(self.q_net_tx) / 1024.0
        cur = np.array([cpu, mem, load, disk, net], dtype=np.float32)
        self.prev = cur.copy()
        self.steps = 0
        return self._obs_norm(np.concatenate([cur, self.prev])), {}

    def step(self, action):
        time.sleep(self.step_seconds)
        self.steps += 1
        cpu = self._q(self.q_cpu)
        mem = self._q(self.q_mem)
        load = self._q(self.q_load1)
        disk = self._q(self.q_disk_r) / 1024.0
        net = self._q(self.q_net_tx) / 1024.0
        cur = np.array([cpu, mem, load, disk, net], dtype=np.float32)
        raw_state = np.concatenate([cur, self.prev])
        self.prev = cur.copy()

        # reward: mimic synthetic reward but use real value thresholds
        overloaded = (cpu >= 85.0) or (mem >= 88.0) or (load >= 3 * os.cpu_count())
        reward = 0.0
        if action == 0:
            reward = 1.0 if not overloaded else -4.0
        else:
            reward = 2.0 if overloaded else -0.5

        # small hysteresis encouragement if previous action info available in info (not enforced here)
        terminated = self.steps >= self.max_steps
        obs = self._obs_norm(raw_state)
        info = {"raw": cur.tolist(), "overloaded": overloaded}
        return obs, float(reward), terminated, False, info

    def _obs_norm(self, state_raw):
        cur = state_raw[:5]
        prev = state_raw[5:]
        normed = np.array([
            _norm(cur[0], MAX_CPU),
            _norm(cur[1], MAX_MEM),
            _norm(cur[2], MAX_LOAD1),
            _norm(cur[3], MAX_IO),
            _norm(cur[4], MAX_IO),
            _norm(prev[0], MAX_CPU),
            _norm(prev[1], MAX_MEM),
            _norm(prev[2], MAX_LOAD1),
            _norm(prev[3], MAX_IO),
            _norm(prev[4], MAX_IO),
        ], dtype=np.float32)
        return normed

    def render(self):
        print("RealMetricsEnv step", self.steps)