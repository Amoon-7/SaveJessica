import numpy as np
import matplotlib.pyplot as plt

class KalmanThompsonBandit:
    def __init__(self, n_arms, q=0.01, r=0.1):
        self.n_arms = n_arms
        self.q = q
        self.r = r
        
        # Initialize beliefs
        self.mu = np.zeros(n_arms)
        self.P = np.ones(n_arms)
    
    def select_arm(self):
        # Thompson sampling step
        samples = np.random.normal(self.mu, np.sqrt(self.P))
        return np.argmax(samples)
    
    def update(self, arm, reward):
        # Predict
        mu_pred = self.mu[arm]
        P_pred = self.P[arm] + self.q
        
        # Kalman gain
        K = P_pred / (P_pred + self.r)
        
        # Update
        self.mu[arm] = mu_pred + K * (reward - mu_pred)
        self.P[arm] = (1 - K) * P_pred
        
        # Predict forward for unplayed arms
        for i in range(self.n_arms):
            if i != arm:
                self.P[i] += self.q  # increase uncertainty over time

if __name__ == "__main__":

    np.random.seed(42)

    # Example usage
    n_arms = 3
    true_means_f = lambda t: np.array([np.sin(t/2), np.cos(t/25), 0.5*np.sin(t/15)])
    bandit = KalmanThompsonBandit(n_arms)
    
    estimated_means = []
    uncertainties = []
    # Simulate some pulls and updates
    for t in range(100):
        arm = bandit.select_arm()
        # Simulate a drifting true mean
        true_means = true_means_f(t)
        reward = np.random.normal(true_means[arm], 0.1)
        bandit.update(arm, reward)
        estimated_means.append(bandit.mu.copy())
        uncertainties.append(bandit.P.copy())
    
    # Plot true means vs estimated means
    time_steps = np.arange(100)
    true_means_over_time = np.array([true_means_f(t) for t in time_steps])
    estimated_means = np.array(estimated_means)
    uncertainties = np.array(uncertainties)
    plt.figure(figsize=(12, 8))
    for i in range(n_arms):
        plt.plot(time_steps, true_means_over_time[:, i], label=f'True Mean Arm {i}', color=f'C{i}', linestyle='--')
        plt.plot(time_steps, estimated_means[:, i], label=f'Estimated Mean Arm {i}', color=f'C{i}')
        plt.fill_between(time_steps,
                         estimated_means[:, i] - np.sqrt(uncertainties[:, i]),
                         estimated_means[:, i] + np.sqrt(uncertainties[:, i]),
                         alpha=0.2, color=f'C{i}')
    plt.xlabel('Time Step')
    plt.ylabel('Mean Reward')
    plt.title('Kalman-Thompson Bandit: True vs Estimated Means')
    plt.legend()
    plt.show()

    plt.savefig("plots/kalman_thompson_bandit.png")