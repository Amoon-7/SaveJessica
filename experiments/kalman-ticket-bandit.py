import numpy as np
import matplotlib.pyplot as plt

class KalmanTicketBandit:
    def __init__(self, n_arms, q=0.01, r_min=1e-3):
        self.n_arms = n_arms
        self.q = q
        self.r_min = r_min
        
        self.mu = np.full(n_arms, 0.5)  # start with uniform probability
        self.P = np.ones(n_arms) * 0.1  # initial uncertainty
    
    def select_action(self):
        # Sample from posterior (Thompson Sampling)
        samples = np.random.normal(self.mu, np.sqrt(self.P))
        samples = np.clip(samples, 0, 1)
        
        # Choose ticket number based on sampled mean
        n_tickets = np.clip((samples * 3).astype(int) + 1, 1, 3)
        
        # Expected reward per arm
        expected_rewards = n_tickets * samples
        
        # Pick best arm
        arm = np.argmax(expected_rewards)
        n = n_tickets[arm]
        
        return arm, n, samples
    
    def update(self, arm, n_tickets, reward):
        # Prediction
        mu_pred = self.mu[arm]
        P_pred = self.P[arm] + self.q
        
        # Approximate observation noise (Binomial variance)
        R_t = max(n_tickets * mu_pred * (1 - mu_pred), self.r_min)
        C_t = n_tickets
        
        # Kalman gain
        K = (P_pred * C_t) / (C_t**2 * P_pred + R_t)
        
        # Update
        self.mu[arm] = mu_pred + K * (reward - C_t * mu_pred)
        self.P[arm] = (1 - K * C_t) * P_pred
        
        # Clip probabilities
        self.mu[arm] = np.clip(self.mu[arm], 0, 1)
        
        # Predict forward for unplayed arms
        for i in range(self.n_arms):
            if i != arm:
                self.P[i] += self.q

if __name__ == "__main__":

    np.random.seed(42)

    bandit = KalmanTicketBandit(n_arms=3, q=0.02)
    # true_p = np.array([0.1, 0.2, 0.3])  # evolving rates (can drift)
    true_p_f = lambda t: (np.array([np.sin(t/2), np.cos(t/25), 0.5*np.sin(t/15)]) + 1) / 2
    T = 100

    # --- Data recording ---
    estimates = np.zeros((T, 3))
    uncertainties = np.zeros((T, 3))
    true_ps = np.zeros((T, 3))
    chosen_arms = np.zeros(T, dtype=int)
    chosen_tickets = np.zeros(T, dtype=int)
    rewards = np.zeros(T)

    for t in range(T):
        # Simulate slow drift
        # true_p += np.random.normal(0, 0.05, size=3)
        # true_p = np.clip(true_p, 0, 1)
        true_p = true_p_f(t)

        # Agent picks arm and tickets
        arm, n, samples = bandit.select_action()
        
        # Generate reward (Binomial)
        reward = np.random.binomial(n, true_p[arm])
        
        # Update
        bandit.update(arm, n, reward)

        # log
        estimates[t] = bandit.mu
        uncertainties[t] = bandit.P
        true_ps[t] = true_p
        chosen_arms[t] = arm
        chosen_tickets[t] = n
        rewards[t] = reward
        
    # --- Plotting ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # 1. True vs estimated probabilities
    for i in range(3):
        axs[0].plot(true_ps[:, i], '--', label=f"True p (arm {i})", color=f"C{i}")
        axs[0].plot(estimates[:, i], label=f"Estimate (arm {i})", color=f"C{i}")
        axs[0].fill_between(range(T),
                            estimates[:, i] - np.sqrt(uncertainties[:, i]),
                            estimates[:, i] + np.sqrt(uncertainties[:, i]),
                            alpha=0.2, color=f"C{i}")
    axs[0].set_ylabel("Probability")
    axs[0].set_title("True vs Estimated Success Probabilities")
    axs[0].legend(loc="best")
    axs[0].grid(True)

    # 2. Chosen arm and ticket count
    axs[1].scatter(range(T), chosen_arms, c=chosen_tickets, cmap='viridis', s=50)
    axs[1].set_ylabel("Chosen Arm")
    axs[1].set_title("Chosen Arm (color = number of tickets)")
    axs[1].grid(True)

    # 3. Rewards
    axs[2].plot(rewards, color='black')
    axs[2].set_ylabel("Reward")
    axs[2].set_xlabel("Time Step")
    axs[2].set_title("Observed Reward per Step")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    plt.savefig("plots/kalman_ticket_bandit.png")
