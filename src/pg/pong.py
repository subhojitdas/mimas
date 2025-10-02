import numpy as np
from torch import nn


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
