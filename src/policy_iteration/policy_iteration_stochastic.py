from __future__ import annotations
import numpy as np

ACTIONS = ['U','R','D','L']
A2DELTA = {'U':(-1,0), 'R':(0,1), 'D':(1,0), 'L':(0,-1)}
LEFT_OF  = {'U':'L', 'L':'D', 'D':'R', 'R':'U'}
RIGHT_OF = {'U':'R', 'R':'D', 'D':'L', 'L':'U'}

class GridworldStochastic:
    """
    4x4 gridworld with stochastic moves:
      - intended action with prob p_intend
      - lateral (left/right of intended) with probs p_left, p_right
    Rewards: step_cost for non-terminal transitions, 0 at terminals.
    """
    def __init__(self, n=4, gamma=1.0, p_intend=0.8, p_left=0.1, p_right=0.1, step_cost=-1.0):
        assert abs(p_intend + p_left + p_right - 1.0) < 1e-12, "probs must sum to 1"
        self.n = n
        self.S = [(i,j) for i in range(n) for j in range(n)]
        self.s_idx = {s:k for k,s in enumerate(self.S)}
        self.gamma = gamma
        self.terminals = {(0,0), (n-1,n-1)}
        self.nS = len(self.S)
        self.nA = len(ACTIONS)
        self.p_intend, self.p_left, self.p_right = p_intend, p_left, p_right
        self.step_cost = step_cost

        # Precompute transitions P(s'|s,a) with rewards
        # transitions[si][ai] = list of (prob, sj, reward)
        self.transitions = self._build_transitions()

    def is_terminal(self, s):
        return s in self.terminals

    def _move(self, s, a):
        """One-step deterministic move; walls keep you in place."""
        if self.is_terminal(s):
            return s  # absorbing
        di, dj = A2DELTA[a]
        i, j = s
        ni, nj = i + di, j + dj
        if not (0 <= ni < self.n and 0 <= nj < self.n):
            return s
        return (ni, nj)

    def _build_transitions(self):
        trans = [[[] for _ in range(self.nA)] for _ in range(self.nS)]
        for si, s in enumerate(self.S):
            for ai, a in enumerate(ACTIONS):
                if self.is_terminal(s):
                    # Absorbing with zero reward
                    sj = self.s_idx[s]
                    trans[si][ai] = [(1.0, sj, 0.0)]
                    continue
                # Stochastic action outcomes
                choices = [
                    (self.p_intend, a),
                    (self.p_left,   LEFT_OF[a]),
                    (self.p_right,  RIGHT_OF[a])
                ]
                # Aggregate probs for identical s' (can happen due to walls)
                probs = {}
                for p, aa in choices:
                    s_next = self._move(s, aa)
                    probs[s_next] = probs.get(s_next, 0.0) + p
                # Build list with rewards
                entries = []
                for s_next, p in probs.items():
                    r = 0.0 if self.is_terminal(s_next) else self.step_cost
                    entries.append((p, self.s_idx[s_next], r))
                trans[si][ai] = entries
        return trans

    def all_q_values(self, V):
        """Q(s,a) = sum_{s'} P(s'|s,a) [ r + gamma * V(s') ]"""
        Q = np.zeros((self.nS, self.nA), dtype=float)
        for si in range(self.nS):
            for ai in range(self.nA):
                val = 0.0
                for p, sj, r in self.transitions[si][ai]:
                    val += p * (r + self.gamma * V[sj])
                Q[si, ai] = val
        return Q

def evaluate_policy(env: GridworldStochastic, policy, theta=1e-9, max_iters=10_000):
    """
    Iterative policy evaluation for stochastic dynamics.
    policy: (nS, nA) with rows summing to 1.
    """
    V = np.zeros(env.nS, dtype=float)
    for _ in range(max_iters):
        delta = 0.0
        for si in range(env.nS):
            s = env.S[si]
            if env.is_terminal(s):
                continue
            v_old = V[si]
            v = 0.0
            for ai in range(env.nA):
                pi = policy[si, ai]
                if pi == 0.0:
                    continue
                # expected return under (s, a)
                qa = 0.0
                for p, sj, r in env.transitions[si][ai]:
                    qa += p * (r + env.gamma * V[sj])
                v += pi * qa
            V[si] = v
            delta = max(delta, abs(v - v_old))
        if delta < theta:
            break
    return V

def improve_policy(env: GridworldStochastic, V, policy):
    Q = env.all_q_values(V)
    best_actions = np.argmax(Q, axis=1)
    new_policy = np.zeros_like(policy)
    for si in range(env.nS):
        s = env.S[si]
        if env.is_terminal(s):
            new_policy[si] = policy[si]  # irrelevant, keep as-is
        else:
            new_policy[si, best_actions[si]] = 1.0
    stable = np.all(np.argmax(policy, axis=1) == np.argmax(new_policy, axis=1))
    return new_policy, stable

def policy_iteration(env: GridworldStochastic, theta=1e-9, max_eval_iters=10_000, max_pi_iters=1_000):
    policy = np.ones((env.nS, env.nA), dtype=float) / env.nA  # uniform start
    for _ in range(max_pi_iters):
        V = evaluate_policy(env, policy, theta=theta, max_iters=max_eval_iters)
        policy, stable = improve_policy(env, V, policy)
        if stable:
            return V, policy
    return V, policy

ARROWS = np.array(['↑','→','↓','←'])

def show_policy(env, policy):
    greedy = np.argmax(policy, axis=1).reshape(env.n, env.n)
    out = []
    for i in range(env.n):
        row = []
        for j in range(env.n):
            s = (i,j)
            row.append('T' if env.is_terminal(s) else ARROWS[greedy[i,j]])
        out.append(' '.join(row))
    return '\n'.join(out)

def show_values(env, V, prec=2):
    grid = V.reshape(env.n, env.n)
    return '\n'.join(' '.join(f"{grid[i,j]: .{prec}f}" for j in range(env.n)) for i in range(env.n))

if __name__ == "__main__":
    # Example: 4x4, stochastic with 0.8/0.1/0.1 slip, episodic
    env = GridworldStochastic(n=4, gamma=1.0, p_intend=0.8, p_left=0.1, p_right=0.1, step_cost=-1.0)
    V, policy = policy_iteration(env, theta=1e-9)
    print("Optimal (stochastic) policy:")
    print(show_policy(env, policy))
    print("\nOptimal values:")
    print(show_values(env, V, prec=6))
