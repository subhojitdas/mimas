from __future__ import annotations
import numpy as np


try:
    import gymnasium as gym
    GYMN = "gymnasium"
except Exception:
    import gym
    GYMN = "gym"


def make_taxi_env(seed: int = 0):
    env = gym.make("Taxi-v3")
    try:
        env.reset(seed=seed)
    except TypeError:
        # older gym
        env.seed(seed)
    return env


def greedy_policy_from_V(P, nS, nA, V, gamma=0.99):
    policy = np.zeros((nS, nA), dtype=float)
    for s in range(nS):
        best = -np.inf
        best_a = 0
        for a in range(nA):
            q = 0.0
            for (p, s2, r, done) in P[s][a]:
                q += p * (r + gamma * (0.0 if done else V[s2]))
            if q > best:
                best = q
                best_a = a
        policy[s, best_a] = 1.0
    return policy

def value_iteration_tabular(P, nS, nA, gamma=0.99, theta=1e-10, max_iters=1_000_000, in_place=False):
    V = np.zeros(nS, dtype=float)
    deltas = []
    for it in range(max_iters):
        delta = 0.0
        if in_place:
            # Gauss–Seidel (updates V immediately)
            for s in range(nS):
                v_old = V[s]
                best = -np.inf
                for a in range(nA):
                    q = 0.0
                    for (p, s2, r, done) in P[s][a]:
                        q += p * (r + gamma * (0.0 if done else V[s2]))
                    if q > best: best = q
                V[s] = best
                delta = max(delta, abs(V[s] - v_old))
        else:
            # Synchronous
            V_new = np.copy(V)
            for s in range(nS):
                best = -np.inf
                for a in range(nA):
                    q = 0.0
                    for (p, s2, r, done) in P[s][a]:
                        q += p * (r + gamma * (0.0 if done else V[s2]))
                    if q > best: best = q
                V_new[s] = best
            delta = np.max(np.abs(V_new - V))
            V = V_new
        deltas.append(delta)
        if delta < theta:
            break
    policy = greedy_policy_from_V(P, nS, nA, V, gamma=gamma)
    return V, policy, deltas