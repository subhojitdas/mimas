import numpy as np
from torch import nn, optim
import torch


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