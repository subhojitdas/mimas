from src.bandit.multi_armed_bandit import BanditAlgorithm
import numpy as np

class BernoulliBandit:

    def __init__(self, probs, seed=None):
        self.probs = np.asarray(probs, dtype=float)
        assert self.probs.ndim == 1 and np.all(self.probs >= 0.0 and self.probs <= 1.0)
        self.K = self.probs.shape[0]
        self.rng = np.random.default_rng(seed=seed)
        self.t = 0

    def reset(self):
        self.t = 0

    def step(self, a):
        self.t += 1
        return float(self.rng.random() < self.probs[a])

    @property
    def optimal_mean(self):
        return float(np.max(self.probs))

    def optimal_arm(self):
        return int(np.argmax(self.probs))


class BaseAgent:
    def __init__(self, K, seed=None):
        self.K = K
        self.rng = np.random.default_rng(seed=seed)
        self.count = np.zeros(K, dtype=int)
        self.values = np.zeros(K, dtype=float)
        self.t = 0

    def select_action(self) -> int:
        raise NotImplementedError

    def update(self, a, r):
        self.t += 1
        self.count[a] += 1
        n = self.count[a]
        # this time we got reward `r` , so this is the incremental update on the previous avg reward
        self.values[a] += (r - self.values[a]) / n
    

