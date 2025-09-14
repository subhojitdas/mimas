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


def evaluate_policy_tabular(P, nS, nA, policy, gamma=0.99, theta=1e-10, max_iters=10_000):
    """Iterative policy evaluation for tabular env.P."""
    V = np.zeros(nS, dtype=float)
    deltas = []
    for it in range(max_iters):
        delta = 0.0
        for s in range(nS):
            v_old = V[s]
            v = 0.0
            for a in range(nA):
                pi = policy[s, a]
                if pi == 0.0:
                    continue
                q = 0.0
                for (p, s2, r, done) in P[s][a]:
                    q += p * (r + gamma * (0.0 if done else V[s2]))
                v += pi * q
            V[s] = v
            delta = max(delta, abs(v - v_old))
        deltas.append(delta)
        if delta < theta:
            break
    return V, deltas


def policy_iteration_tabular(P, nS, nA, gamma=0.99, theta=1e-10, max_eval_iters=10_000, max_pi_iters=1_000):
    policy = np.ones((nS, nA), dtype=float) / nA  # start uniform
    total_eval_sweeps = 0
    eval_deltas_per_outer = []
    for k in range(max_pi_iters):
        V, eval_deltas = evaluate_policy_tabular(P, nS, nA, policy, gamma=gamma, theta=theta, max_iters=max_eval_iters)
        total_eval_sweeps += len(eval_deltas)
        eval_deltas_per_outer.append(eval_deltas)
        # Greedy improvement
        new_policy = greedy_policy_from_V(P, nS, nA, V, gamma=gamma)
        stable = np.all(np.argmax(policy, axis=1) == np.argmax(new_policy, axis=1))
        policy = new_policy
        if stable:
            return V, policy, total_eval_sweeps, eval_deltas_per_outer, k + 1
    return V, policy, total_eval_sweeps, eval_deltas_per_outer, max_pi_iters


if __name__ == "__main__":
    env = make_taxi_env(seed=0)
    # Unwrap the environment to get the environment with P attribute
    while not hasattr(env, 'P') and hasattr(env, 'env'):
        env = env.env
    # Gymnasium: env.P directly available; older gym too.
    P = env.P
    nS = env.observation_space.n
    nA = env.action_space.n

    gamma = 0.99
    theta = 1e-10

    # Value Iteration
    V_vi, pi_vi, deltas_vi = value_iteration_tabular(P, nS, nA, gamma=gamma, theta=theta, in_place=False)
    # Policy Iteration
    V_pi, pi_pi, eval_sweeps, eval_deltas_per_outer, n_improvements = policy_iteration_tabular(
        P, nS, nA, gamma=gamma, theta=theta
    )

    # Metrics
    print(f"Environment: Taxi-v3 via {GYMN}")
    print(f"States: {nS}, Actions: {nA}, gamma={gamma}")
    print("\n--- Value Iteration ---")
    print(f"sweeps (VI): {len(deltas_vi)}")
    print(f"final Δ (VI): {deltas_vi[-1]:.3e}")

    print("\n--- Policy Iteration ---")
    print(f"policy improvements: {n_improvements}")
    print(f"total evaluation sweeps: {eval_sweeps}")
    print(f"last eval Δ: {eval_deltas_per_outer[-1][-1]:.3e}")

    # Compare solutions
    max_abs_diff = np.max(np.abs(V_vi - V_pi))
    print("\n--- Comparison ---")
    print(f"‖V_VI - V_PI‖_∞ = {max_abs_diff:.6e}")
    same_greedy = np.mean(np.argmax(pi_vi, axis=1) == np.argmax(pi_pi, axis=1))
    print(f"Fraction of states with identical greedy action: {same_greedy:.4f}")

    # Peek a few states
    ACTION_NAMES = ["S", "N", "E", "W", "PU", "DO"]
    print("\nSample greedy actions for first 20 states (VI vs PI):")
    for s in range(20):
        a_vi = ACTION_NAMES[int(np.argmax(pi_vi[s]))]
        a_pi = ACTION_NAMES[int(np.argmax(pi_pi[s]))]
        print(f"s={s:3d}: VI={a_vi}  PI={a_pi}  V_VI={V_vi[s]:7.3f}  V_PI={V_pi[s]:7.3f}")
