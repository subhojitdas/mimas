# a2c_cartpole.py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, act_dim)  # logits
        self.value_head = nn.Linear(hidden, 1)         # V(s)

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)  # [B]
        return logits, value


def train(
    env_id="CartPole-v1",
    seed=0,
    gamma=0.99,
    lr=3e-4,
    max_episodes=3000,
    entropy_bonus=0.01,
    value_coef=0.5,
    normalize_adv=True,
    device="cpu",
):
    env = gym.make(env_id)
    env.reset(seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = ActorCritic(obs_dim, act_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    reward_ma = None

    for ep in range(1, max_episodes + 1):
        obs, _ = env.reset()
        done = False

        obs_list, act_list, rew_list, done_list = [], [], [], []

        ep_return = 0.0

        # -------- Rollout one episode --------
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = model(obs_t)
            dist = Categorical(logits=logits)
            action_t = dist.sample()
            action = int(action_t.item())

            next_obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs_list.append(obs)
            act_list.append(action)
            rew_list.append(float(r))
            done_list.append(done)

            ep_return += float(r)
            obs = next_obs

        # -------- Make tensors --------
        obs_batch = torch.as_tensor(np.array(obs_list), dtype=torch.float32, device=device)   # [T, obs_dim]
        act_batch = torch.as_tensor(np.array(act_list), dtype=torch.int64, device=device)    # [T]
        rew_batch = torch.as_tensor(np.array(rew_list), dtype=torch.float32, device=device)  # [T]
        done_batch = torch.as_tensor(np.array(done_list), dtype=torch.float32, device=device)# [T] (1.0 if done)

        # Values for s_t
        logits, values = model(obs_batch)  # logits [T, A], values [T]

        # Bootstrap values for s_{t+1}
        # We need V(s_{t+1}) for each t. Easiest: build next_obs list and run model once.
        next_obs_list = obs_list[1:] + [obs]  # last next_obs is final obs after episode ends
        next_obs_batch = torch.as_tensor(np.array(next_obs_list), dtype=torch.float32, device=device)
        with torch.no_grad():
            _, next_values = model(next_obs_batch)  # [T]

        # -------- A2C bootstrapped target and advantage (1-step TD) --------
        # target_t = r_t + gamma * V(s_{t+1}) * (1 - done_t)
        targets = rew_batch + gamma * next_values * (1.0 - done_batch)

        advantages = targets - values  # NOT detached here yet; we'll detach for actor loss

        if normalize_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # -------- Losses --------
        dist = Categorical(logits=logits)
        logp = dist.log_prob(act_batch)       # [T]
        entropy = dist.entropy()              # [T]

        # Actor: maximize logπ * advantage (so minimize negative)
        actor_loss = -(logp * advantages.detach()).sum()

        # Critic: fit V(s) to bootstrapped targets
        critic_loss = 0.5 * ((values - targets.detach()) ** 2).sum()

        # Entropy bonus encourages exploration
        entropy_loss = -entropy_bonus * entropy.sum()

        loss = actor_loss + value_coef * critic_loss + entropy_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        reward_ma = ep_return if reward_ma is None else 0.95 * reward_ma + 0.05 * ep_return
        if ep % 10 == 0:
            print(
                f"ep={ep:4d}  return={ep_return:6.1f}  ma={reward_ma:6.1f}  T={len(rew_list):3d}  "
                f"actor={actor_loss.item():8.3f}  critic={critic_loss.item():8.3f}  ent={entropy.mean().item():6.3f}"
            )

        if reward_ma is not None and reward_ma >= 475:
            print(f"Solved! moving_avg_return={reward_ma:.1f} at episode {ep}")
            break

    env.close()
    return model


if __name__ == "__main__":
    train()
