from mpmath import sqrtm

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

    def select_action(self):
        # ensure every arm is pulled at least once
        for a in range(self.K):
            if self.count[a] == 0:
                return a

        # UCB1 score
        t = max(1, self.t)
        bonus = np.sqrt(self.c * np.log(t) / self.count[a])
        scores = self.values + bonus
        return np.argmax(scores)

