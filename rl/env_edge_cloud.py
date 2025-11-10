# rl/env_edge_cloud.py
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

class EdgeCloudEnv(gym.Env):
    """
    Synthetic Edge-Cloud RL environment
    - Observation: 10D (current metrics + previous metrics)
    - Action: 0 = EDGE, 1 = CLOUD
    - Reward: encourages EDGE under normal load, CLOUD under overload
    """

    def __init__(self):
        super().__init__()

        # Observation: [cpu, mem, load1, disk_io, net_io, prev_cpu, prev_mem, prev_load1, prev_disk, prev_net]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )

        # Action: 0 = EDGE, 1 = CLOUD
        self.action_space = spaces.Discrete(2)

        # State tracking
        self.state = None
        self.prev = None
        self.steps = 0
        self.max_steps = 300

    def _generate_metrics(self, prev=None):
        """
        Generate synthetic system metrics with bursts and recovery.
        Returns normalized values in 0-1 scale.
        """
        if prev is None:
            cpu = random.uniform(5, 30)
            mem = random.uniform(40, 70)
            load1 = random.uniform(1, 5)
            disk_io = random.uniform(0, 10)
            net_io = random.uniform(0, 10)
        else:
            cpu, mem, load1, disk_io, net_io = prev

            if random.random() < 0.25:  # burst
                cpu = min(100, cpu + random.uniform(20, 50))
                mem = min(100, mem + random.uniform(5, 15))
                load1 = min(50, load1 + random.uniform(2, 5))
            else:  # recovery
                cpu = max(0, cpu - random.uniform(5, 10))
                mem = max(0, mem - random.uniform(2, 5))
                load1 = max(0, load1 - random.uniform(0.5, 1.0))

            # background IO noise
            disk_io = min(100, max(0, disk_io + random.uniform(-2, 2)))
            net_io = min(100, max(0, net_io + random.uniform(-2, 2)))

        # normalize
        return np.array([
            cpu / 100.0,
            mem / 100.0,
            min(load1 / 50.0, 1.0),
            min(disk_io / 100.0, 1.0),
            min(net_io / 100.0, 1.0),
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        cur = self._generate_metrics(None)
        self.prev = self._generate_metrics(None)
        self.state = np.concatenate([cur, self.prev]).astype(np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        # Current and previous metrics (normalized)
        cur = self._generate_metrics(self.state[:5] * np.array([100, 100, 50, 100, 100]))
        prev = self.state[:5]  # previous step is last current
        self.state = np.concatenate([cur, prev]).astype(np.float32)

        # Denormalized for reward logic
        cpu = cur[0] * 100
        mem = cur[1] * 100
        load1 = cur[2] * 50

        # ---- Reward function ----
        overloaded = (cpu > 85.0) or (mem > 80.0) or (load1 > 24.0)

        if action == 0:  # EDGE
            if not overloaded:
                reward = +2.0   # ✅ good: efficient & cheap
            else:
                reward = -5.0   # ❌ edge overloaded
        else:  # CLOUD
            if overloaded:
                reward = +3.0   # ✅ good: safe offload
            else:
                reward = -2.0   # ❌ wasted cost

        self.steps += 1
        terminated = self.steps >= self.max_steps
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def render(self):
        cpu = self.state[0] * 100
        mem = self.state[1] * 100
        load1 = self.state[2] * 50
        print(f"[render] step={self.steps}, cpu={cpu:.1f}, mem={mem:.1f}, load1={load1:.2f}")