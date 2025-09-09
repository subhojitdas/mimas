from __future__ import annotations
import numpy as np

# ----- MDP definition: 4x4 Gridworld -----
ACTIONS = ['U', 'R', 'D', 'L']
A2DELTA = {'U': (-1, 0), 'R': (0, 1), 'D': (1, 0), 'L': (0, -1)}


class Gridworld:
    def __init__(self, gamma=1.0):
        self.n = 5
        self.S = [(i, j) for i in range(self.n) for j in range(self.n)]
        self.s_idx = {s: k for k, s in enumerate(self.S)}
        self.gamma = gamma
        self.terminals = {(0, 0), (self.n - 1, self.n - 1)}
        self.nS = len(self.S)
        self.nA = len(ACTIONS)

    def is_terminal(self, s):
        return s in self.terminals

    def step(self, s, a):
        """Deterministic transition. Returns (s_next, reward)."""
        if self.is_terminal(s):
            return s, 0.0
        di, dj = A2DELTA[a]
        i, j = s
        ni, nj = i + di, j + dj
        # bounce off walls
        if not (0 <= ni < self.n and 0 <= nj < self.n):
            ni, nj = i, j
        s_next = (ni, nj)
        reward = 0.0 if self.is_terminal(s_next) else -1.0
        return s_next, reward

    def all_q_values(self, V):
        """Compute Q(s,a) for all s,a given V."""
        Q = np.zeros((self.nS, self.nA))
        for si, s in enumerate(self.S):
            for ai, a in enumerate(ACTIONS):
                s_next, r = self.step(s, a)
                Q[si, ai] = r + self.gamma * V[self.s_idx[s_next]]
        return Q


# ----- Policy Evaluation -----
def evaluate_policy(env: Gridworld, policy, theta=1e-9, max_iters=10_000):
    """
    policy: shape (nS, nA) stochastic policy (rows sum to 1).
    Returns V (nS,)
    """
    V = np.zeros(env.nS, dtype=float)
    for _ in range(max_iters):
        delta = 0.0
        for si, s in enumerate(env.S):
            if env.is_terminal(s):
                continue
            v_old = V[si]
            v = 0.0
            for ai, a in enumerate(ACTIONS):
                if policy[si, ai] == 0.0:
                    continue
                s_next, r = env.step(s, a)
                v += policy[si, ai] * (r + env.gamma * V[env.s_idx[s_next]])
            V[si] = v
            delta = max(delta, abs(v - v_old))
        if delta < theta:
            break
    return V


# ----- Policy Improvement -----
def improve_policy(env: Gridworld, V, policy):
    """
    Greedy improvement with tie-breaking to keep it stable & deterministic.
    Returns (new_policy, policy_stable).
    """
    Q = env.all_q_values(V)
    best_actions = np.argmax(Q, axis=1)  # greedy
    new_policy = np.zeros_like(policy)
    for si in range(env.nS):
        # If terminal, leave as uniform (won't be used)
        if env.is_terminal(env.S[si]):
            new_policy[si] = policy[si]
        else:
            new_policy[si, best_actions[si]] = 1.0
    policy_stable = np.all(np.argmax(policy, axis=1) == np.argmax(new_policy, axis=1))
    return new_policy, policy_stable


# ----- Full Policy Iteration -----
def policy_iteration(env: Gridworld, theta=1e-9, max_eval_iters=10_000, max_pi_iters=1_000):
    # start from uniform random policy
    policy = np.ones((env.nS, env.nA), dtype=float) / env.nA
    for _ in range(max_pi_iters):
        V = evaluate_policy(env, policy, theta=theta, max_iters=max_eval_iters)
        policy, stable = improve_policy(env, V, policy)
        if stable:
            return V, policy
    return V, policy


# ----- Pretty printing helpers -----
ARROWS = np.array(['↑', '→', '↓', '←'])


def show_policy(env, policy):
    greedy = np.argmax(policy, axis=1)
    grid = greedy.reshape(env.n, env.n)
    out = []
    for i in range(env.n):
        row = []
        for j in range(env.n):
            s = (i, j)
            if env.is_terminal(s):
                row.append('T')
            else:
                row.append(ARROWS[grid[i, j]])
        out.append(' '.join(row))
    return '\n'.join(out)


def show_values(env, V, prec=2):
    grid = V.reshape(env.n, env.n)
    return '\n'.join(' '.join(f"{grid[i, j]: .{prec}f}" for j in range(env.n)) for i in range(env.n))


# ----- Run -----
if __name__ == "__main__":
    env = Gridworld(gamma=1.0)
    V, policy = policy_iteration(env, theta=1e-9)
    print("Optimal policy (greedy arrows, T = terminal):")
    print(show_policy(env, policy))
    print("\nOptimal state values:")
    print(show_values(env, V, prec=3))
