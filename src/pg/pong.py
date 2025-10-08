import numpy as np
from torch import nn, optim
import torch
import gymnasium as gym


def preprocess(obs):
    """
    210x160x3 uint 8 Pong frame into 80*80 float32 vector.
    """
    # obs: (210,160,3)
    obs = obs[35:195]          # crop
    obs = obs[::2, ::2, 0]     # downsample by 2, take R channel -> (80,80)
    obs[(obs == 144) | (obs == 109)] = 0  # erase background (type 1)
    obs[obs != 0] = 1          # paddles, ball = 1
    return obs.astype(np.float32).ravel()  # (80*80,)


class PolicyNet(nn.Module):

    def __init__(self, input_dim=80*80, hidden_dim=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # (batch, 80*80)
        return self.net(x)


class REINFORCEPongAgent:

    def __init__(self, gamma=0.99, lr=1e-3, device=None):
        self.gamma = gamma
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNet().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Storage for one episode
        self.log_probs = []
        self.rewards = []

    def select_action(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        logits = self.policy(x.unsqueeze(0))  # (1,1)
        prob_up = logits.squeeze(0).squeeze(0)  # scalar in [0,1]

        m = torch.distributions.Bernoulli(prob_up)
        action_sample = m.sample()
        log_prob = m.log_prob(action_sample)

        env_action = 2 if action_sample.item() == 1.0 else 3

        self.log_probs.append(log_prob)
        return env_action

    def finish_episode(self):
        # Compute discounted returns G_t
        returns = []
        R = 0.0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.append(R)
        returns = list(reversed(returns))
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.stack(self.log_probs)  # (T,)
        loss = -torch.sum(log_probs * returns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs.clear()
        self.rewards.clear()

        return loss.item()


# --------- Training loop ----------
def train_pong_pg(
    env_name="ale_py:ALE/Pong-v5",
    num_episodes=1000,
    render=False,
    save_every=100,
    model_path="pong_pg.pt",
):
    # Some gym versions require this wrapper to get proper atari preprocessing; for
    # now we do our own frame processing and just disable frame skip.
    env = gym.make(env_name)
    agent = REINFORCEPongAgent()

    running_reward = None

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
        prev_x = None
        episode_reward = 0.0

        done = False
        while not done:
            if render:
                env.render()

            cur_x = preprocess(obs)
            # Use frame difference to emphasize motion
            x = cur_x - prev_x if prev_x is not None else np.zeros_like(cur_x)
            prev_x = cur_x

            action = agent.select_action(x)
            step_out = env.step(action)
            # Gym vs Gymnasium step signature
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                obs, reward, done, info = step_out

            agent.rewards.append(reward)
            episode_reward += reward

        loss = agent.finish_episode()
        running_reward = (
            episode_reward
            if running_reward is None
            else running_reward * 0.99 + episode_reward * 0.01
        )

        print(
            f"Episode {episode:4d} | "
            f"Reward: {episode_reward:6.1f} | "
            f"Running reward: {running_reward:6.1f} | "
            f"Loss: {loss:8.4f}"
        )

        if episode % save_every == 0:
            torch.save(agent.policy.state_dict(), model_path)
            print(f"Saved model to {model_path}")

    env.close()
    torch.save(agent.policy.state_dict(), model_path)
    print("Training finished, final model saved.")


if __name__ == "__main__":
    # You can tweak num_episodes; Pong often needs a few thousand episodes to get good.
    train_pong_pg(num_episodes=1000, render=False)