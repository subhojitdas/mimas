

# SARSA is an on-policy algorithm to learn action-value fn **Q(S,A)**

import numpy as np
import random


class GridworldEnv:
    """
    Simple 4x4 gridworld:
    - Start: (0, 0)
    - Goal:  (3, 3)
    - Reward: -1 per step, 0 at goal
    - Episode ends when we reach the goal.
    """
    def __init__(self, size=4):
        self.size = size
        self.start_state = (0, 0)
        self.goal_state = (size - 1, size - 1)
        self.state = self.start_state

    def reset(self):
        self.state = self.start_state
        return self._state_to_index(self.state)

    def step(self, action):
        """
        action: 0=up, 1=right, 2=down, 3=left
        returns: next_state_idx, reward, done
        """
        x, y = self.state

        if action == 0:   # up
            x = max(x - 1, 0)
        elif action == 1: # right
            y = min(y + 1, self.size - 1)
        elif action == 2: # down
            x = min(x + 1, self.size - 1)
        elif action == 3: # left
            y = max(y - 1, 0)

        self.state = (x, y)

        if self.state == self.goal_state:
            reward = 0.0
            done = True
        else:
            reward = -1.0
            done = False

        return self._state_to_index(self.state), reward, done

    def _state_to_index(self, state):
        x, y = state
        return x * self.size + y  # flatten 2D -> 1D index

    def _index_to_state(self, idx):
        return divmod(idx, self.size)


def epsilon_greedy_action(Q, state, epsilon):
    """
    Choose action using epsilon-greedy policy from Q[s, :]
    """
    if random.random() < epsilon:
        # explore
        return random.randint(0, Q.shape[1] - 1)
    else:
        # exploit
        return int(np.argmax(Q[state]))


def sarsa(
    num_episodes=500,
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
):
    env = GridworldEnv(size=4)

    n_states = env.size * env.size
    n_actions = 4

    # Q-table: shape (n_states, n_actions)
    Q = np.zeros((n_states, n_actions))

    epsilon = epsilon_start

    for episode in range(num_episodes):
        state = env.reset()

        # select initial action using current policy
        action = epsilon_greedy_action(Q, state, epsilon)

        done = False
        total_reward = 0

        while not done:
            next_state, reward, done = env.step(action)
            total_reward += reward

            # choose next action a' from next_state
            next_action = epsilon_greedy_action(Q, next_state, epsilon)

            # SARSA update
            td_target = reward + gamma * Q[next_state, next_action] * (0.0 if done else 1.0)
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            action = next_action

        # decay epsilon after each episode
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, total_reward={total_reward}, epsilon={epsilon:.3f}")

    return Q, env