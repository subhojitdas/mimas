import numpy as np
from dataclasses import dataclass

from src.bandit.multi_armed_bandit import BanditAlgorithm


class GaussianBandit:
    def __init__(self, means, stds, seed=None):
        self.means = np.asarray(means, dtype=float)
        self.stds = np.asarray(stds, dtype=float)
        assert self.means.ndim == 1 and self.stds.ndim == 1
        assert self.means.shape[0] == self.stds.shape[0]
        assert np.all(self.stds) > 0
        self.K = self.means.shape[0]
        self.rng = np.random.default_rng(seed=seed)
        self.t = 0

    def reset(self):
        self.t = 0

    def step(self, a):
        self.t += 1
        return float(self.rng.normal(self.means[a], self.stds[a]))

    def optimal_mean(self):
        return float(np.max(self.means))

    def optimal_arm(self):
        return int(np.argmax(self.means))



