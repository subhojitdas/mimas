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

    @property
    def optimal_mean(self):
        return float(np.max(self.means))

    @property
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

        t = max(1, self.t)
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
class GaussianThompson(BaseAgent):

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


@dataclass
class RunResult:
    actions: np.ndarray
    rewards: np.ndarray
    cum_rewards: np.ndarray
    instantaneous_regret: np.ndarray
    cumulative_regret: np.ndarray
    final_estimates: np.ndarray
    counts: np.ndarray

def run_bandit(env, agent, T: int) -> RunResult:
    actions = np.zeros(T, dtype=int)
    rewards = np.zeros(T, dtype=float)
    inst_regret = np.zeros(T, dtype=float)

    env.reset()
    for t in range(T):
        a = agent.select_action()
        r = env.step(a)
        agent.update(a, r)

        actions[t] = a
        rewards[t] = r
        # pseudo-regret uses true means (env is known only to evaluator)
        inst_regret[t] = env.optimal_mean - env.means[a]

    return RunResult(
        actions=actions,
        rewards=rewards,
        cum_rewards=np.cumsum(rewards),
        instantaneous_regret=inst_regret,
        cumulative_regret=np.cumsum(inst_regret),
        final_estimates=agent.values.copy(),
        counts=agent.count.copy(),
    )


if __name__ == "__main__":
    means = np.array([0.0, 0.5, 1.0, 1.1, 0.8])
    stds  = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # same noise for clarity
    env = GaussianBandit(means, stds, seed=42)
    T = 10_000

    # ε-greedy
    eg = EpsilonGreedy(K=env.K, eps=0.1, seed=0)
    eg_res = run_bandit(env, eg, T)
    print("ε-greedy:", eg_res.cumulative_regret[-1], eg_res.counts)

    # UCB-V
    env = GaussianBandit(means, stds, seed=42)
    ucbv = UCBV(K=env.K, seed=0)
    ucbv_res = run_bandit(env, ucbv, T)
    print("UCB-V:", ucbv_res.cumulative_regret[-1], ucbv_res.counts)

    # Gaussian Thompson (assume known variance = 1.0)
    env = GaussianBandit(means, stds, seed=42)
    ts = GaussianThompson(K=env.K, obs_var=1.0, mu0=0.0, tau0_sq=10.0, seed=0)
    ts_res = run_bandit(env, ts, T)
    print("Gaussian TS:", ts_res.cumulative_regret[-1], ts_res.counts)