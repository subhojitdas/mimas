import numpy as np
import matplotlib.pyplot as plt


def run_epsilon_greedy_bandit(
        k: int = 10,
        eps: float = 0.1,
        n_steps: int = 1000,
        n_runs: int = 2000,
        init_q: float = 0.0,
        reward_std: float = 1.0,
):
    rewards = np.zeros((n_runs, n_steps))

    for run in range(n_runs):
        # True action values q* ~ N(0, 1) for this bandit instance
        q_true = np.random.normal(loc=0.0, scale=20.0, size=k)

        # Estimated action values
        q_est = np.full(k, init_q, dtype=np.float64)
        action_counts = np.zeros(k, dtype=np.int32)

        for t in range(n_steps):
            # ε-greedy action selection
            if np.random.rand() < eps:
                action = np.random.randint(k)  # explore
            else:
                action = np.argmax(q_est)  # exploit

            reward = np.random.normal(q_true[action], reward_std)
            rewards[run, t] = reward

            action_counts[action] += 1
            alpha = 1.0 / action_counts[action]  # step-size = 1 / N(a)
            q_est[action] += alpha * (reward - q_est[action])

    # Average reward over runs
    avg_rewards = rewards.mean(axis=0)
    return avg_rewards


if __name__ == "__main__":
    k = 10
    n_steps = 1000
    n_runs = 2000

    epsilons = [0.0, 0.01, 0.1] # diff eps values

    plt.figure(figsize=(8, 5))

    for eps in epsilons:
        avg_rewards = run_epsilon_greedy_bandit(
            k=k,
            eps=eps,
            n_steps=n_steps,
            n_runs=n_runs,
        )
        plt.plot(avg_rewards, label=f"ε = {eps}")

    plt.xlabel("Step")
    plt.ylabel("Average reward")
    plt.title(f"{k}-armed bandit: ε-greedy, average reward vs. step")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

