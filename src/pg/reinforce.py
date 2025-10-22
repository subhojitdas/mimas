# reinforce_cartpole.py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ----------------------------
# Policy network
# ----------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # logits for a categorical distribution
        return self.net(x)

    def action_and_logprob(self, obs: np.ndarray):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)  # [1, obs_dim]
        logits = self(obs_t)                                            # [1, act_dim]
        dist = Categorical(logits=logits)
        action = dist.sample()                                          # [1]
        logp = dist.log_prob(action)                                    # [1]
        return int(action.item()), logp.squeeze(0)

# ----------------------------
# Returns computation
# ----------------------------
def compute_returns(rewards, gamma: float):
    """
    rewards: list[float] length T
    returns: torch.Tensor shape [T]
    """
    G = 0.0
    out = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)

# ----------------------------
# Training loop (REINFORCE)
# ----------------------------
def train(
    env_id="CartPole-v1",
    seed=0,
    gamma=0.99,
    lr=1e-2,
    max_episodes=1000,
    return_normalize=True,
    entropy_bonus=0.0,  # keep 0.0 for "pure" REINFORCE; can set small like 0.01
    device="cpu",
):
    env = gym.make(env_id)
    env.reset(seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNet(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    reward_ma = None

    for ep in range(1, max_episodes + 1):
        obs, _ = env.reset()
        done = False

        logps = []
        entropies = []
        rewards = []

        while not done:
            action, logp = policy.action_and_logprob(obs)

            # get entropy for optional exploration regularizer
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                dist = Categorical(logits=policy(obs_t))
                ent = dist.entropy().squeeze(0)

            next_obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            logps.append(logp)
            entropies.append(ent)
            rewards.append(float(r))

            obs = next_obs

        # Compute returns G_t
        returns = compute_returns(rewards, gamma).to(device)

        # Optional: normalize returns to reduce variance (common in practice)
        if return_normalize:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        logps_t = torch.stack(logps).to(device)          # [T]
        ent_t = torch.stack(entropies).to(device)        # [T]

        # REINFORCE loss: -E[ logπ(a_t|s_t) * G_t ]
        loss = -(logps_t * returns).sum()

        # Optional entropy bonus (helps exploration; keep 0.0 for pure REINFORCE)
        if entropy_bonus != 0.0:
            loss -= entropy_bonus * ent_t.sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        ep_return = sum(rewards)
        reward_ma = ep_return if reward_ma is None else 0.95 * reward_ma + 0.05 * ep_return

        if ep % 10 == 0:
            print(
                f"ep={ep:4d}  return={ep_return:6.1f}  ma={reward_ma:6.1f}  "
                f"T={len(rewards):3d}  loss={loss.item():8.3f}"
            )

        # CartPole is "solved" around ~475+ over 100 episodes in Gymnasium terms;
        # this is just a convenient early stop.
        if reward_ma is not None and reward_ma >= 475:
            print(f"Solved! moving_avg_return={reward_ma:.1f} at episode {ep}")
            break

    env.close()
    return policy

if __name__ == "__main__":
    train()
