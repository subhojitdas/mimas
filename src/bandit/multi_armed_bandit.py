"""
Multi-Armed Bandit Problem Implementation from Scratch

This module implements the classic multi-armed bandit problem, which is one of the
fundamental problems in reinforcement learning. In this problem:

- We have N "arms" (actions), each with its own reward distribution
- At each time step, we pull one arm and observe a reward
- The goal is to maximize the total reward over time

This implementation includes several strategies:
1. Random selection (baseline)
2. Epsilon-Greedy
3. Upper Confidence Bound (UCB)
4. Thompson Sampling (Bayesian approach)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class BanditArm:
    def __init__(self, true_mean: float, variance: float = 1.0):
        self.true_mean = true_mean
        self.variance = variance
    
    def pull(self) -> float:
        return np.random.normal(self.true_mean, np.sqrt(self.variance))


class MultiArmedBandit:
    def __init__(self, n_arms: int, means: List[float] = None, variances: List[float] = None):
        self.n_arms = n_arms
        self.means = means
        if variances is None:
            self.variances = [1.0] * n_arms
        else:
            self.variances = variances
            
        self.arms = [BanditArm(mean, var) for mean, var in zip(self.means, self.variances)]
        
    def get_reward(self, arm_index: int) -> float:
        return self.arms[arm_index].pull()
    
    def best_arm(self) -> int:
        return np.argmax(self.means)


class BanditAlgorithm:
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)  # Number of times each arm was pulled
        self.values = np.zeros(n_arms)  # Estimated value of each arm
        self.name = "Base Algorithm"
    
    def select_arm(self) -> int:
        raise NotImplementedError
    
    def update(self, arm_index: int, reward: float) -> None:
        self.counts[arm_index] += 1
        
        n = self.counts[arm_index]
        value = self.values[arm_index]
        self.values[arm_index] = ((n - 1) / n) * value + (1 / n) * reward


class RandomAlgorithm(BanditAlgorithm):
    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        self.name = "Random"
    
    def select_arm(self) -> int:
        return np.random.randint(self.n_arms)


class EpsilonGreedy(BanditAlgorithm):
    """
    Epsilon-Greedy algorithm.
    
    With probability epsilon, explore (choose a random arm)
    With probability 1-epsilon, exploit (choose the best arm according to current estimates)
    """
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        super().__init__(n_arms)
        self.epsilon = epsilon
        self.name = f"ε-Greedy (ε={epsilon})"
    
    def select_arm(self) -> int:
        """
        Select an arm using epsilon-greedy strategy.
        
        Returns:
            Index of the selected arm
        """
        # Explore: with probability epsilon, choose a random arm
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        
        # Exploit: choose the arm with the highest estimated value
        # If there are multiple arms with the same value, choose randomly among them
        best_value = np.max(self.values)
        best_arms = np.where(self.values == best_value)[0]
        
        return np.random.choice(best_arms)


class UCB(BanditAlgorithm):
    """
    Upper Confidence Bound algorithm.
    
    Selects the arm that has the highest upper confidence bound.
    This balances exploration and exploitation by considering both 
    the estimated value and the uncertainty of that estimate.
    """
    def __init__(self, n_arms: int, c: float = 2.0):
        super().__init__(n_arms)
        self.c = c  # Exploration parameter
        self.t = 0  # Total number of trials
        self.name = f"UCB (c={c})"
    
    def select_arm(self) -> int:
        """
        Select an arm using UCB strategy.
        
        Returns:
            Index of the selected arm
        """
        # First, make sure all arms have been tried at least once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        
        # Calculate the UCB values for each arm
        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            # UCB formula: Q(a) + c * sqrt(log(t) / N(a))
            # where Q(a) is the estimated value, t is total trials,
            # N(a) is the number of times arm a was pulled, and c is exploration parameter
            bonus = self.c * np.sqrt(np.log(self.t) / self.counts[arm])
            ucb_values[arm] = self.values[arm] + bonus
        
        # Choose the arm with the highest UCB value
        # If there are multiple arms with the same value, choose randomly among them
        best_value = np.max(ucb_values)
        best_arms = np.where(ucb_values == best_value)[0]
        
        return np.random.choice(best_arms)
    
    def update(self, arm_index: int, reward: float) -> None:
        """
        Update the algorithm's internal estimates based on the observed reward.
        
        Args:
            arm_index: The arm that was pulled
            reward: The observed reward
        """
        super().update(arm_index, reward)
        self.t += 1


class ThompsonSampling(BanditAlgorithm):
    """
    Thompson Sampling algorithm (Bayesian approach).
    
    Uses a beta distribution to model the uncertainty about each arm's reward probability.
    For each arm, we sample from its beta distribution and choose the arm with the highest sample.
    """
    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        # For each arm, we maintain alpha and beta parameters of a Beta distribution
        # These represent our belief about the reward distribution
        self.alpha = np.ones(n_arms)  # Prior success count (add 1 for prior)
        self.beta = np.ones(n_arms)   # Prior failure count (add 1 for prior)
        self.name = "Thompson Sampling"
    
    def select_arm(self) -> int:
        """
        Select an arm using Thompson Sampling.
        
        Returns:
            Index of the selected arm
        """
        # Sample a value from each arm's beta distribution
        samples = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            samples[arm] = np.random.beta(self.alpha[arm], self.beta[arm])
        
        # Choose the arm with the highest sample
        return np.argmax(samples)
    
    def update(self, arm_index: int, reward: float) -> None:
        """
        Update the algorithm's internal estimates based on the observed reward.
        
        Args:
            arm_index: The arm that was pulled
            reward: The observed reward
        """
        super().update(arm_index, reward)
        
        # For Thompson Sampling with normal rewards, we need to adapt the updating rule
        # We'll use a simple approach: reward > 0 is a success, reward <= 0 is a failure
        # In a real-world scenario, you might want a more sophisticated update rule
        if reward > 0:
            self.alpha[arm_index] += 1
        else:
            self.beta[arm_index] += 1


def run_experiment(
    bandit: MultiArmedBandit, 
    algorithms: List[BanditAlgorithm], 
    n_steps: int = 1000,
    n_experiments: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a bandit experiment.
    
    Args:
        bandit: The multi-armed bandit environment
        algorithms: List of algorithms to test
        n_steps: Number of steps to run for each experiment
        n_experiments: Number of experiments to average over
        
    Returns:
        Tuple of (average_rewards, optimal_arm_percentages)
    """
    # Initialize arrays to store results
    n_algorithms = len(algorithms)
    all_rewards = np.zeros((n_algorithms, n_experiments, n_steps))
    all_optimal_arm = np.zeros((n_algorithms, n_experiments, n_steps))
    
    # Get the index of the optimal arm
    optimal_arm = bandit.best_arm()
    
    # Run the experiments
    for exp in range(n_experiments):
        # Reset all algorithms for this experiment
        for alg in algorithms:
            alg.counts = np.zeros(bandit.n_arms)
            alg.values = np.zeros(bandit.n_arms)
            if isinstance(alg, UCB):
                alg.t = 0
            if isinstance(alg, ThompsonSampling):
                alg.alpha = np.ones(bandit.n_arms)
                alg.beta = np.ones(bandit.n_arms)
        
        # Run a single experiment for all algorithms
        for step in range(n_steps):
            for i, alg in enumerate(algorithms):
                # Select an arm using this algorithm
                arm = alg.select_arm()
                
                # Get a reward from the environment
                reward = bandit.get_reward(arm)
                
                # Update the algorithm's estimates
                alg.update(arm, reward)
                
                # Store the results
                all_rewards[i, exp, step] = reward
                all_optimal_arm[i, exp, step] = 1 if arm == optimal_arm else 0
    
    # Calculate the average reward across experiments
    average_rewards = np.mean(all_rewards, axis=1)
    
    # Calculate the percentage of times the optimal arm was chosen
    optimal_arm_percentages = np.mean(all_optimal_arm, axis=1)
    
    return average_rewards, optimal_arm_percentages


def plot_results(
    average_rewards: np.ndarray,
    optimal_arm_percentages: np.ndarray,
    algorithms: List[BanditAlgorithm],
    n_steps: int
):
    """
    Plot the results of the experiment.
    
    Args:
        average_rewards: Average rewards for each algorithm at each step
        optimal_arm_percentages: Percentage of times the optimal arm was chosen
        algorithms: List of algorithms tested
        n_steps: Number of steps in the experiment
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot the average reward
    for i, alg in enumerate(algorithms):
        # Calculate the cumulative average reward
        cumulative_avg = np.cumsum(average_rewards[i]) / np.arange(1, n_steps + 1)
        ax1.plot(cumulative_avg, label=alg.name)
    
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Average Reward over Time')
    ax1.legend()
    ax1.grid(True)
    
    # Plot the percentage of optimal arm pulls
    for i, alg in enumerate(algorithms):
        # Use a moving average to smooth the plot
        window_size = min(100, n_steps // 10)
        smoothed = np.convolve(optimal_arm_percentages[i], np.ones(window_size)/window_size, mode='valid')
        ax2.plot(smoothed, label=alg.name)
    
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('% Optimal Arm Pulls')
    ax2.set_title('Optimal Arm Selection Rate')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('bandit_results.png')
    plt.show()


def main():
    """Run a demonstration of the multi-armed bandit problem"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create a bandit with 10 arms
    n_arms = 10
    # Fixed set of means for reproducibility
    means = np.array([-0.25, 1.0, -0.5, 1.5, 0.0, 0.75, -1.0, 0.5, -1.5, 2.0])
    bandit = MultiArmedBandit(n_arms, means=means)
    
    print(f"True arm means: {bandit.means}")
    print(f"Best arm: {bandit.best_arm()} with mean reward {bandit.means[bandit.best_arm()]}")
    
    # Create algorithms to test
    algorithms = [
        RandomAlgorithm(n_arms),
        EpsilonGreedy(n_arms, epsilon=0.1),
        EpsilonGreedy(n_arms, epsilon=0.01),
        UCB(n_arms, c=2.0),
        ThompsonSampling(n_arms)
    ]
    
    # Run the experiment
    n_steps = 1000
    n_experiments = 100
    print(f"Running {n_experiments} experiments with {n_steps} steps each...")
    avg_rewards, optimal_percentages = run_experiment(
        bandit, algorithms, n_steps=n_steps, n_experiments=n_experiments
    )
    
    # Plot the results
    plot_results(avg_rewards, optimal_percentages, algorithms, n_steps)
    
    # Print the final percentage of optimal arm pulls
    print("\nFinal percentage of optimal arm pulls:")
    for i, alg in enumerate(algorithms):
        print(f"{alg.name}: {optimal_percentages[i, -1]*100:.1f}%")
    
    # Print the final average cumulative reward
    print("\nFinal average cumulative reward:")
    for i, alg in enumerate(algorithms):
        cumulative_avg = np.sum(avg_rewards[i]) / n_steps
        print(f"{alg.name}: {cumulative_avg:.3f}")


if __name__ == "__main__":
    main()
