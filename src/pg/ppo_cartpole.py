import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ----------------------------
# Actor-Critic model
# ----------------------------
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
        value = self.value_head(h).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value


# ----------------------------
# GAE computation
# ----------------------------
@torch.no_grad()
def compute_gae(rewards, values, dones, gamma: float, lam: float):
    """
    rewards: [T] float tensor
    values:  [T+1] float tensor (includes bootstrap value at T)
    dones:   [T] float tensor (1.0 if done else 0.0)
    returns: [T]
    adv:     [T]
    """
    T = rewards.shape[0]
    adv = torch.zeros(T, dtype=torch.float32, device=rewards.device)
    gae = 0.0
    for t in reversed(range(T)):
        # if done at t, no bootstrap beyond t
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    returns = adv + values[:-1]
    return returns, adv


# ----------------------------
# PPO training
# ----------------------------
def train_ppo(
    env_id="CartPole-v1",
    seed=0,
    total_steps=200_000,     # increase if needed
    rollout_len=2048,        # steps collected per iteration
    gamma=0.99,
    gae_lambda=0.95,
    lr=3e-4,
    clip_eps=0.2,
    update_epochs=10,
    minibatch_size=256,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    target_kl=0.02,          # early stop if policy changes too much (optional)
    device="cpu",
):
    # --- Env ---
    env = gym.make(env_id)
    env.reset(seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # --- Model/Optim ---
    model = ActorCritic(obs_dim, act_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    obs, _ = env.reset()

    # For logging
    ep_return = 0.0
    ep_len = 0
    returns_window = []

    steps_done = 0
    iter_idx = 0

    while steps_done < total_steps:
        iter_idx += 1

        # ----------------------------
        # 1) Collect rollout (on-policy)
        # ----------------------------
        obs_buf = torch.zeros((rollout_len, obs_dim), dtype=torch.float32, device=device)
        act_buf = torch.zeros((rollout_len,), dtype=torch.int64, device=device)
        logp_old_buf = torch.zeros((rollout_len,), dtype=torch.float32, device=device)
        rew_buf = torch.zeros((rollout_len,), dtype=torch.float32, device=device)
        done_buf = torch.zeros((rollout_len,), dtype=torch.float32, device=device)
        val_buf = torch.zeros((rollout_len,), dtype=torch.float32, device=device)

        for t in range(rollout_len):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            obs_buf[t] = obs_t

            action, logp, value = model.act(obs_t.unsqueeze(0))
            action = action.squeeze(0)
            logp = logp.squeeze(0)
            value = value.squeeze(0)

            act_buf[t] = action
            logp_old_buf[t] = logp
            val_buf[t] = value

            next_obs, r, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated

            rew_buf[t] = float(r)
            done_buf[t] = 1.0 if done else 0.0

            ep_return += float(r)
            ep_len += 1

            obs = next_obs
            steps_done += 1

            if done:
                returns_window.append(ep_return)
                ep_return = 0.0
                ep_len = 0
                obs, _ = env.reset()

            if steps_done >= total_steps:
                # still finish this rollout buffer element-wise, but break if desired
                pass

        # Bootstrap value V(s_T) for GAE
        with torch.no_grad():
            obs_T = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, v_T = model(obs_T)
            v_T = v_T.squeeze(0)
        values_ext = torch.cat([val_buf, v_T.unsqueeze(0)], dim=0)  # [T+1]

        # ----------------------------
        # 2) Compute returns/advantages (GAE)
        # ----------------------------
        returns, adv = compute_gae(rew_buf, values_ext, done_buf, gamma, gae_lambda)

        # Normalize advantages (very standard)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ----------------------------
        # 3) PPO updates (multiple epochs over rollout)
        # ----------------------------
        n = rollout_len
        idxs = np.arange(n)

        approx_kl = 0.0
        clipfrac_accum = 0.0
        n_minibatches = 0

        for epoch in range(update_epochs):
            np.random.shuffle(idxs)

            for start in range(0, n, minibatch_size):
                mb_idx = idxs[start:start + minibatch_size]
                mb_idx = torch.as_tensor(mb_idx, dtype=torch.int64, device=device)

                obs_mb = obs_buf[mb_idx]
                act_mb = act_buf[mb_idx]
                logp_old_mb = logp_old_buf[mb_idx]
                adv_mb = adv[mb_idx]
                ret_mb = returns[mb_idx]

                logits, value = model(obs_mb)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act_mb)
                entropy = dist.entropy().mean()

                # Ratio r_t(theta)
                ratio = torch.exp(logp - logp_old_mb)

                # Clipped surrogate objective
                unclipped = ratio * adv_mb
                clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Value function loss (optionally you can clip value too; keeping simple)
                value_loss = 0.5 * (ret_mb - value).pow(2).mean()

                # Total loss
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()

                # Diagnostics
                with torch.no_grad():
                    # approx KL: E[logp_old - logp]
                    approx_kl_batch = (logp_old_mb - logp).mean().item()
                    approx_kl += approx_kl_batch

                    clipped_mask = (torch.abs(ratio - 1.0) > clip_eps).float()
                    clipfrac_accum += clipped_mask.mean().item()
                    n_minibatches += 1

            # Optional early stopping if KL too high
            if target_kl is not None and n_minibatches > 0:
                mean_kl = approx_kl / n_minibatches
                if mean_kl > 1.5 * target_kl:
                    break

        mean_kl = approx_kl / max(n_minibatches, 1)
        clipfrac = clipfrac_accum / max(n_minibatches, 1)

        # ----------------------------
        # Logging
        # ----------------------------
        if len(returns_window) > 0:
            last_50 = returns_window[-50:]
            avg_ret = float(np.mean(last_50))
        else:
            avg_ret = float("nan")

        print(
            f"iter={iter_idx:3d} steps={steps_done:7d}/{total_steps} "
            f"avg_return(last50)={avg_ret:7.2f}  "
            f"kl={mean_kl:8.5f}  clipfrac={clipfrac:6.3f}"
        )

        # CartPole usually considered solved around 475+ avg over 100 episodes
        if len(returns_window) >= 100 and np.mean(returns_window[-100:]) >= 475:
            print(f"Solved! avg_return(last100)={np.mean(returns_window[-100:]):.1f}")
            break

    env.close()
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ppo(device=device)
