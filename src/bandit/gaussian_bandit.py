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


class BaseAgent:
    def __init__(self, K, seed=None):
        self.K = K
        self.rng = np.random.default_rng(seed)
        self.count = np.zeros(K, dtype=int)
        self.values = np.zeros(K, dtype=float)  # running mean
        self.t = 0

    def select_action(self) -> int:
        raise NotImplementedError

    def update(self, a, r: float):
        self.t += 1
        self.count[a] += 1
        n = self.count[a]
        self.values[a] += (r - self.values[a]) / n  # incremental mean


class EpsilonGreedy(BaseAgent):
    def __init__(self, K, eps=0.1, seed=None):
        super().__init__(K, seed)
        assert 0.0 <= eps <= 1.0
        self.eps = eps

    def select_action(self) -> int:
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.K))  # explore
        return int(np.argmax(self.values))         # exploit


class UCBV(BaseAgent):
    """
    UCB-V for unbounded/gaussian-like rewards.
    Uses empirical variance per arm.
    """
    def __init__(self, K, seed=None):
        super().__init__(K, seed)
        self.sumsq = np.zeros(K, dtype=float)  # sum of squares per arm

    def select_action(self) -> int:
        # Pull each arm once to initialize
        for a in range(self.K):
            if self.count[a] == 0:
                return a

        t = np.max(1, self.t)
        n = self.count.astype(float)

        # empirical variance: var = E[x^2] - (E[x])^2
        ex2 = self.sumsq / n
        mu  = self.values
        var = np.maximum(1e-12, ex2 - mu**2)  # keep positive

        bonus = np.sqrt((2.0 * var * np.log(t)) / n) + (3.0 * np.log(t)) / n
        scores = mu + bonus
        return int(np.argmax(scores))

    def update(self, a, r: float):
        super().update(a, r)
        self.sumsq[a] += r * r


#Gaussian Thompson Sampling (Normal prior, known variance)
class GuassianThompson(BaseAgent):
    def __init__(self, K, obs_var=1.0, mu0=0.0, tau0_sq=1.0, seed=None):
        super().__init__(K, seed)
        self.obs_var = float(obs_var)
        self.mu0 = float(mu0)
        self.tau0_sq = float(tau0_sq)
        self.sums = np.zeros(K, dtype=float)

    def select_action(self) -> int:
        for a in range(self.K):
            if self.count[a] == 0:
                return a
        n = self.count.astype(float)
        tau_n_sq = 1.0 / (1.0 / self.tau0_sq + n / self.obs_var)
        mu_n = tau_n_sq * (self.mu0 / self.tau0_sq + self.sums / self.obs_var)

        samples = self.rng.normal(mu_n, np.sqrt(tau_n_sq))
        return int(np.argmax(samples))

    def update(self, a, r: float):
        super().update(a, r)
        self.sums[a] += r
