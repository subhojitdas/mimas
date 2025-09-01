from mpmath import sqrtm

from src.bandit.multi_armed_bandit import BanditAlgorithm
import numpy as np

class BernoulliBandit:

    def __init__(self, probs, seed=None):
        self.probs = np.asarray(probs, dtype=float)
        assert self.probs.ndim == 1 and np.all((self.probs >= 0.0) & (self.probs <= 1.0))
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


class EpsilonGreedy(BaseAgent):
    def __init__(self, K, eps=0.1, seed=None):
        super().__init__(K, seed)
        assert 0.0 <= eps <= 1.0
        self.eps = eps

    def select_action(self):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.K)) # explore
        return int(np.argmax(self.values)) # exploit


class UCB1(BaseAgent):
    """
    # Cons of eps-greedy
        Exploration is uninformed — it pulls every arm equally, even clearly suboptimal ones.
        Fixed ε means you keep exploring forever (even though you know what's best).
        Decayed ε can solve this, but tuning the decay is non-trivial
    This is problem is solved by UCB1.
    Instead of random exploration like ε-greedy, UCB1 assumes that uncertain arms are better than
     what we currently know — so explore them until their uncertainty is reduced
     UCB1 balance exploration and expoitation automatically
    """
    def __init__(self, K, c=2.0, seed=None):
        super().__init__(K, seed)
        """
        c is the exploration coefficient; classic UCB1 uses sqrt((2 ln t)/n_a).
        You can tune c (multiplying the sqrt term) if desired.
        """
        self.c = c

    def select_action(self) -> int:
        # Warm-up: pull each arm once
        for a in range(self.K):
            if self.count[a] == 0:
                return a

        # After warm-up
        t = max(self.t, 1)
        bonus = np.sqrt((self.c * np.log(t)) / self.count)
        scores = self.values + bonus
        return int(np.argmax(scores))


class ThompsonSampling(BaseAgent):
    def __init__(self, K, seed=None):
        super().__init__(K, seed)
        self.alpha = np.ones(K, dtype=float)
        self.beta = np.ones(K, dtype=float)

    def select_action(self):
        # Warm-up: pull each arm once
        for a in range(self.K):
            if self.count[a] == 0:
                return a
        # Thompson draw
        theta = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(theta))

    def update(self, a, r):
        super().update(a, r)
        #update posterior
        if r == 1.0:
            self.alpha[a] += 1.0
        else:
            self.beta[a] += 1.0

from dataclasses import dataclass

@dataclass
class RunResult:
    actions: np.ndarray
    rewards: np.ndarray
    cum_rewards: np.ndarray
    instantaneous_regret: np.ndarray
    cumulative_regret: np.ndarray
    final_estimates: np.ndarray
    counts: np.ndarray


def run_bandit(env: BernoulliBandit, agent: BaseAgent, T: int) -> RunResult:
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
        inst_regret[t] = env.optimal_mean - env.probs[a]

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
    # Define a 5-armed bandit with unknown (to the agent) success probs
    true_probs = np.array([0.92, 0.25, 0.20, 0.90, 0.91])
    env = BernoulliBandit(true_probs, seed=42)

    T = 100_000

    # ε-greedy
    eg_agent = EpsilonGreedy(K=env.K, eps=0.1, seed=0)
    eg_result = run_bandit(env, eg_agent, T)
    print("ε-greedy:")
    print("  total reward:", eg_result.cum_rewards[-1])
    print("  cumulative regret:", eg_result.cumulative_regret[-1])
    print("  pulls per arm:", eg_result.counts)
    print("  value estimates:", np.round(eg_result.final_estimates, 3))

    # UCB1
    env = BernoulliBandit(true_probs, seed=42)  # fresh env for fair comparison
    ucb_agent = UCB1(K=env.K, c=2.0, seed=0)
    ucb_result = run_bandit(env, ucb_agent, T)
    print("\nUCB1:")
    print("  total reward:", ucb_result.cum_rewards[-1])
    print("  cumulative regret:", ucb_result.cumulative_regret[-1])
    print("  pulls per arm:", ucb_result.counts)
    print("  value estimates:", np.round(ucb_result.final_estimates, 3))

    # Thompson Sampling
    env = BernoulliBandit(true_probs, seed=42)  # fresh env for fair comparison
    ts_agent = ThompsonSampling(K=env.K, seed=0)
    ts_result = run_bandit(env, ts_agent, T)
    print("\nThompson Sampling:")
    print("  total reward:", ts_result.cum_rewards[-1])
    print("  cumulative regret:", ts_result.cumulative_regret[-1])
    print("  pulls per arm:", ts_result.counts)
    print("  value estimates:", np.round(ts_result.final_estimates, 3))





