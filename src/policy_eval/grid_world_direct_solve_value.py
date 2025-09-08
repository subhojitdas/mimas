import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

# P[s][a] = list of (prob, next_state, reward, done)
Transitions = Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]]

A_UP, A_RIGHT, A_DOWN, A_LEFT = 0, 1, 2, 3
ACTIONS = (A_UP, A_RIGHT, A_DOWN, A_LEFT)

@dataclass
class Gridworld:
    n_rows: int = 4
    n_cols: int = 4
    terminals: Tuple[int, ...] = (0, 15)        # top-left and bottom-right as terminals
    step_reward: float = -1.0                   # -1 per step encourages faster termination
    terminal_reward: float = 0.0                # reward when entering/being in terminal
    stay_on_wall: bool = True                   # bumping into wall keeps you in place

    def build(self) -> Tuple[int, int, Transitions]:
        nS = self.n_rows * self.n_cols
        nA = len(ACTIONS)
        P: Transitions = {s: {a: [] for a in ACTIONS} for s in range(nS)}

        def to_rc(s):
            return divmod(s, self.n_cols)

        def to_s(r, c):
            return r * self.n_cols + c

        for s in range(nS):
            r, c = to_rc(s)
            is_terminal = s in self.terminals

            for a in ACTIONS:
                if is_terminal:
                    # Terminal is absorbing
                    P[s][a] = [(1.0, s, self.terminal_reward, True)]
                    continue

                nr, nc = r, c
                if a == A_UP:    nr = max(r - 1, 0)
                if a == A_RIGHT: nc = min(c + 1, self.n_cols - 1)
                if a == A_DOWN:  nr = min(r + 1, self.n_rows - 1)
                if a == A_LEFT:  nc = max(c - 1, 0)

                ns = to_s(nr, nc)
                if not self.stay_on_wall:
                    # optional variant: if you try to go out of bounds, you "lose" the move
                    # Here we already clipped, so 'stay_on_wall' and this variant act the same.
                    pass

                done = ns in self.terminals
                rew = self.terminal_reward if done else self.step_reward
                P[s][a] = [(1.0, ns, rew, done)]
        return nS, nA, P


def direct_solve_value(P: Transitions, policy: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """
    Build P_pi and r_pi and solve (I - γ P_pi) V = r_pi.
    """
    nS, nA = policy.shape
    P_pi = np.zeros((nS, nS), dtype=np.float64)
    r_pi = np.zeros(nS, dtype=np.float64)

    for s in range(nS):
        for a in range(nA):
            pa = policy[s, a]
            if pa == 0.0:
                continue
            for (p, ns, r, done) in P[s][a]:
                P_pi[s, ns] += pa * p * (0.0 if done else 1.0)  # no future value if episode ends
                r_pi[s]     += pa * p * r

    I = np.eye(nS)
    # Solve (I - γP)V = r
    A = I - gamma * P_pi
    V = np.linalg.solve(A, r_pi)
    return V


if __name__ == "__main__":
    env = Gridworld(n_rows=4, n_cols=4, terminals=(0, 15), step_reward=-1.0, terminal_reward=0.0)
    nS, nA, P = env.build()

    # Uniform random policy: pi(a|s)=1/nA except at terminals (ignored by transitions anyway)
    policy = np.full((nS, nA), 1.0 / nA, dtype=np.float64)

    gamma = 0.99
    V_direct = direct_solve_value(P, policy, gamma=gamma)

    # Pretty print as 4x4 grid
    def grid(v):
        return "\n".join(" ".join(f"{v[r*4+c]:7.6f}" for c in range(4)) for r in range(4))

    print("\nDirect Solve values (should closely match):")
    print(grid(V_direct))
    # print("\nMax abs diff:", np.max(np.abs(V_iter - V_direct)))