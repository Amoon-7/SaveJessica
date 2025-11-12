import numpy as np
import matplotlib.pyplot as plt

class DecayingBetaBandit:
    def __init__(self, n_arms, decay=0.02, exploration_constant=2.0, min_alpha_beta=0.0):
        self.n_arms = n_arms
        self.decay = decay
        self.alpha = np.ones(n_arms)  # prior successes
        self.beta = np.ones(n_arms)   # prior failures
        self.c = exploration_constant
        self.min_alpha_beta = min_alpha_beta

    def mean_estimates(self):
        return self.alpha / (self.alpha + self.beta)

    def variance_estimates(self):
        s = self.alpha + self.beta
        return (self.alpha * self.beta) / (s*s*(s+1))

    def select_action(self):
        samples = np.random.beta(self.alpha, self.beta)                         # Thompson Sampling
        exploration_bonus = self.c * np.sqrt(self.variance_estimates() + 1e-6)  # UCB-like bonus
        n_tickets = np.clip((self.mean_estimates() * 3).astype(int) + 1, 1, 3)  # Simple ticket rule
        
        arm = np.argmax(samples + exploration_bonus)
        return arm, n_tickets[arm], samples

    def update(self, arm, reward):
        # decay old evidence
        self.alpha = np.maximum(self.min_alpha_beta, self.alpha * (1 - self.decay))
        self.beta = np.maximum(self.min_alpha_beta, self.beta * (1 - self.decay))
        
        success = 1 if reward > 0 else 0
        
        self.alpha[arm] += success
        self.beta[arm] += (1 - success)

if __name__ == "__main__":
    # --- Simulation setup ---
    np.random.seed(42)
    bandit = DecayingBetaBandit(n_arms=3, decay=0.02)
    true_p = np.array([0.3, 0.5, 0.7])
    # true_p_f = lambda t: (np.array([np.sin(t/2), np.cos(t/25), 0.5*np.sin(t/15)]) + 1) / 2
    T = 200

    # --- Logging ---
    estimated_means = np.zeros((T, 3))
    uncertainties = np.zeros((T, 3))
    true_ps = np.zeros((T, 3))
    chosen_arms = np.zeros(T, dtype=int)
    chosen_tickets = np.zeros(T, dtype=int)
    rewards = np.zeros(T)

    # --- Simulation loop ---
    for t in range(T):
        # Slowly drift true probabilities
        true_p += np.random.normal(0, 0.01, size=3)
        true_p = np.clip(true_p, 0, 1)

        # true_p = true_p_f(t)

        # Agent chooses arm and ticket count
        arm, n_tickets, _ = bandit.select_action()
        # arm, n_tickets, = (0, 1)
        success = np.random.rand() < true_p[arm]
        reward = n_tickets if success else 0

        # Update beliefs
        bandit.update(arm, reward)

        # Log data
        estimated_means[t] = bandit.mean_estimates()
        uncertainties[t] = bandit.variance_estimates()
        true_ps[t] = true_p
        chosen_arms[t] = arm
        chosen_tickets[t] = n_tickets
        rewards[t] = reward


    # --- Plot results ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # True vs Estimated p
    for i in range(3):
        axs[0].plot(true_ps[:, i], '--', label=f"True p (arm {i})", color=f"C{i}")
        axs[0].plot(estimated_means[:, i], label=f"Estimate (arm {i})", color=f"C{i}")
        axs[0].fill_between(
            np.arange(T),
            estimated_means[:, i] - np.sqrt(uncertainties[:, i]),
            estimated_means[:, i] + np.sqrt(uncertainties[:, i]),
            alpha=0.2, color=f'C{i}'
        ) 
    axs[0].set_ylabel("Success probability")
    axs[0].set_title("True vs Estimated Success Probabilities")
    axs[0].legend()
    axs[0].grid(True)

    # Chosen arm and ticket count
    scatter = axs[1].scatter(range(T), chosen_arms, c=chosen_tickets, cmap="viridis", s=50)
    axs[1].set_ylabel("Chosen Arm")
    axs[1].set_title("Chosen Arm (color = number of tickets)")
    axs[1].grid(True)
    cbar = plt.colorbar(scatter, ax=axs[1])
    cbar.set_label("Tickets")

    # Rewards
    axs[2].plot(rewards, color="black")
    axs[2].set_ylabel("Reward")
    axs[2].set_xlabel("Time Step")
    axs[2].set_title("Observed Reward per Step")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
