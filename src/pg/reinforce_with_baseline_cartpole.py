# reinforce_with_baseline_cartpole.py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ----------------------------
# Actor (policy) network
# ----------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)  # logits


# ----------------------------
# Critic (value baseline)
# ----------------------------
class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)  # [B]


def compute_returns(rewards, gamma: float):
    G = 0.0
    out = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)  # [T]


def train(
    env_id="CartPole-v1",
    seed=0,
    gamma=0.99,
    actor_lr=1e-2,
    critic_lr=1e-2,
    max_episodes=2000,
    entropy_bonus=0.0,      # keep 0.0 initially
    value_coef=0.5,         # weight for critic loss
    normalize_adv=True,     # common variance reduction
    device="cpu",
):
    env = gym.make(env_id)
    env.reset(seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    actor = PolicyNet(obs_dim, act_dim).to(device)
    critic = ValueNet(obs_dim).to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=critic_lr)

    reward_ma = None

    for ep in range(1, max_episodes + 1):
        obs, _ = env.reset()
        done = False

        obs_list = []
        act_list = []
        rew_list = []

        ep_return = 0.0

        # ---------- Rollout one episode ----------
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # [1, obs_dim]
            logits = actor(obs_t)
            dist = Categorical(logits=logits)
            action_t = dist.sample()                 # [1]
            action = int(action_t.item())

            next_obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs_list.append(obs)
            act_list.append(action)
            rew_list.append(float(r))
            ep_return += float(r)

            obs = next_obs

        # ---------- Prepare tensors ----------
        obs_batch = torch.as_tensor(np.array(obs_list), dtype=torch.float32, device=device)  # [T, obs_dim]
        act_batch = torch.as_tensor(np.array(act_list), dtype=torch.int64, device=device)   # [T]
        returns = compute_returns(rew_list, gamma).to(device)                               # [T]

        # ---------- Critic predictions ----------
        values = critic(obs_batch)  # [T]

        # ---------- Advantages ----------
        # IMPORTANT: detach baseline in actor loss so actor doesn't backprop into critic
        adv = returns - values.detach()

        if normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---------- Actor loss ----------
        logits = actor(obs_batch)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(act_batch)  # [T]
        entropy = dist.entropy()         # [T]

        actor_loss = -(logp * adv).sum()
        if entropy_bonus != 0.0:
            actor_loss -= entropy_bonus * entropy.sum()

        # ---------- Critic loss ----------
        # Fit V(s) to returns (regression)
        critic_loss = 0.5 * ((values - returns) ** 2).sum()

        # ---------- Optimize (separately) ----------
        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_opt.step()

        critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_opt.step()

        # ---------- Logging ----------
        reward_ma = ep_return if reward_ma is None else 0.95 * reward_ma + 0.05 * ep_return
        if ep % 10 == 0:
            print(
                f"ep={ep:4d}  return={ep_return:6.1f}  ma={reward_ma:6.1f}  "
                f"T={len(rew_list):3d}  actor_loss={actor_loss.item():8.3f}  critic_loss={critic_loss.item():8.3f}"
            )

        if reward_ma is not None and reward_ma >= 475:
            print(f"Solved! moving_avg_return={reward_ma:.1f} at episode {ep}")
            break

    env.close()
    return actor, critic


if __name__ == "__main__":
    train()
