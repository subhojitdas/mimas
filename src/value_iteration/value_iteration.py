from __future__ import annotations
import numpy as np

ACTIONS = ['U','R','D','L']
A2DELTA = {'U':(-1,0), 'R':(0,1), 'D':(1,0), 'L':(0,-1)}
LEFT_OF  = {'U':'L', 'L':'D', 'D':'R', 'R':'U'}
RIGHT_OF = {'U':'R', 'R':'D', 'D':'L', 'L':'U'}

class GridworldStochastic:
    """
    n x n gridworld with stochastic moves:
      intended with p_intend, lateral with p_left/p_right.
    Terminals at (0,0) and (n-1,n-1). Terminals return 0 reward and are absorbing.
    Non-terminal step reward = step_cost.
    """
    def __init__(self, n=4, gamma=1.0, p_intend=0.8, p_left=0.1, p_right=0.1, step_cost=-1.0):
        assert abs(p_intend + p_left + p_right - 1.0) < 1e-12
        self.n = n
        self.S = [(i,j) for i in range(n) for j in range(n)]
        self.s_idx = {s:k for k,s in enumerate(self.S)}
        self.gamma = gamma
        self.terminals = {(0,0), (n-1,n-1)}
        self.nS = len(self.S)
        self.nA = len(ACTIONS)
        self.p_intend, self.p_left, self.p_right = p_intend, p_left, p_right
        self.step_cost = step_cost
        self.transitions = self._build_transitions()

    def is_terminal(self, s): return s in self.terminals

    def _move(self, s, a):
        if self.is_terminal(s):
            return s
        di, dj = A2DELTA[a]
        i, j = s
        ni, nj = i + di, j + dj
        if not (0 <= ni < self.n and 0 <= nj < self.n):
            return s
        return (ni, nj)

    def _build_transitions(self):
        # transitions[si][ai] = list of (prob, sj, reward)
        trans = [[[] for _ in range(self.nA)] for _ in range(self.nS)]
        for si, s in enumerate(self.S):
            for ai, a in enumerate(ACTIONS):
                if self.is_terminal(s):
                    trans[si][ai] = [(1.0, self.s_idx[s], 0.0)]
                    continue
                choices = [
                    (self.p_intend, a),
                    (self.p_left,   LEFT_OF[a]),
                    (self.p_right,  RIGHT_OF[a]),
                ]
                agg = {}
                for p, aa in choices:
                    s_next = self._move(s, aa)
                    agg[s_next] = agg.get(s_next, 0.0) + p
                entries = []
                for s_next, p in agg.items():
                    r = 0.0 if self.is_terminal(s_next) else self.step_cost
                    entries.append((p, self.s_idx[s_next], r))
                trans[si][ai] = entries
        return trans

    def all_q_values(self, V):
        Q = np.zeros((self.nS, self.nA), dtype=float)
        for si in range(self.nS):
            for ai in range(self.nA):
                val = 0.0
                for p, sj, r in self.transitions[si][ai]:
                    val += p * (r + self.gamma * V[sj])
                Q[si, ai] = val
        return Q

# ---------- Value Iteration ----------
def value_iteration(env: GridworldStochastic, theta=1e-9, max_iters=1_000_000, in_place=False):
    """
    Runs VI until max-norm change < theta.
    If in_place=True, uses Gauss-Seidel (updates V[s] immediately) which often converges faster.
    Returns V*, and greedy policy derived from it.
    """
    V = np.zeros(env.nS, dtype=float)
    for _ in range(max_iters):
        delta = 0.0
        if in_place:
            # Gauss-Seidel style
            for si in range(env.nS):
                s = env.S[si]
                if env.is_terminal(s):
                    continue
                v_old = V[si]
                # Bellman optimality backup
                best = -np.inf
                for ai in range(env.nA):
                    q = 0.0
                    for p, sj, r in env.transitions[si][ai]:
                        q += p * (r + env.gamma * V[sj])
                    if q > best: best = q
                V[si] = best
                delta = max(delta, abs(V[si] - v_old))
        else:
            # Synchronous backup
            V_new = np.copy(V)
            for si in range(env.nS):
                s = env.S[si]
                if env.is_terminal(s):
                    continue
                best = -np.inf
                for ai in range(env.nA):
                    q = 0.0
                    for p, sj, r in env.transitions[si][ai]:
                        q += p * (r + env.gamma * V[sj])
                    if q > best: best = q
                V_new[si] = best
            delta = np.max(np.abs(V_new - V))
            V = V_new
        if delta < theta:
            break
    # Greedy policy extraction
    policy = extract_greedy_policy(env, V)
    return V, policy

def extract_greedy_policy(env: GridworldStochastic, V):
    Q = env.all_q_values(V)
    greedy = np.argmax(Q, axis=1)
    policy = np.zeros((env.nS, env.nA), dtype=float)
    for si in range(env.nS):
        s = env.S[si]
        if env.is_terminal(s):
            policy[si] = np.ones(env.nA)/env.nA  # arbitrary
        else:
            policy[si, greedy[si]] = 1.0
    return policy

# ---------- Pretty printers ----------
ARROWS = np.array(['↑','→','↓','←'])
def show_policy(env, policy):
    greedy = np.argmax(policy, axis=1).reshape(env.n, env.n)
    rows = []
    for i in range(env.n):
        row = []
        for j in range(env.n):
            s = (i,j)
            row.append('T' if env.is_terminal(s) else ARROWS[greedy[i,j]])
        rows.append(' '.join(row))
    return '\n'.join(rows)

def show_values(env, V, prec=3):
    G = V.reshape(env.n, env.n)
    return '\n'.join(' '.join(f"{G[i,j]: .{prec}f}" for j in range(env.n)) for i in range(env.n))

# ---------- Demo ----------
if __name__ == "__main__":
    env = GridworldStochastic(n=4, gamma=1.0, p_intend=0.8, p_left=0.1, p_right=0.1, step_cost=-1.0)
    V, pi = value_iteration(env, theta=1e-9, in_place=False)  # try in_place=True for faster convergence
    print("Optimal policy from VI:")
    print(show_policy(env, pi))
    print("\nOptimal values from VI:")
    print(show_values(env, V, prec=6))
