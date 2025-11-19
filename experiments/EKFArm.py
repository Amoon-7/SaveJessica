import numpy as np
import matplotlib.pyplot as plt

class EKFArm:
    """
    EKF estimator for a stationary phase φ in the model:

        x_t ~ Bernoulli(p_t)
        p_t = (1 + sin(ω t − φ)) / 2
        z_t = 2*x_t - 1     ∈ {-1, +1}

    Measurement model:   h(φ) = sin(ω t − φ)
    Jacobian:            dh/dφ = -cos(ω t − φ)

    R_t = min(R_min, 4*p_hat*(1-p_hat))
    """

    def __init__(self, omega, phi0=0.0, P0=1.0, Q=1e-2, Q_decay=0.95, min_Q=5e-5):
        self.omega = omega
        self.phi = phi0         # state = phase estimate
        self.P = P0             # variance of φ
        self.Q = Q              # process noise
        self.Q_decay = Q_decay  # process noise decay
        self.min_Q = min_Q
        self.last_innov = 0.0

    def update(self, t, x):
        """
        EKF update using Bernoulli→z transformation.
        """
        # Convert observation x ∈ {0,1} to z ∈ {-1,+1}
        z = 2*x - 1

        # --------- Measurement prediction ---------
        h = np.sin(self.omega * t - self.phi)       # predicted z
        H = -np.cos(self.omega * t - self.phi)      # Jacobian dh/dφ

        # predicted probability p_hat ∈ [0,1]
        p_hat = 0.5 * (1.0 + h)

        # --------- R_t calculation ---------
        # Bernoulli variance mapped to z-space:
        alpha = np.clip(0.1 + 0.9 * np.exp(-abs(self.last_innov)), 0.2, 1.0)
        R_t = max(1e-2, alpha * 4 * p_hat * (1 - p_hat))

        # --------- Kalman computations ---------
        # annealed process noise
        self.P += self.Q
        self.Q *= max(self.Q_decay, self.min_Q)

        # Kalman gain
        S = H * self.P * H + R_t
        K = 0.0 if S <= 1e-12 else (self.P * H) / S

        # innovation
        innov = z - h
        self.last_innov = innov

        # state update
        self.phi = self.phi + K * innov

        # covariance update with minimum P
        # P_min doesn't let the gain die out
        # P_factor makes update more conservative
        P_min = 1e-3
        self.P = max((1 - 1.0 * K * H) * self.P, P_min)

    def predict_z(self, t):
        return np.sin(self.omega * t - self.phi)

    def predict_p(self, t):
        z = self.predict_z(t)
        return 0.5 * (1 + z)
    
    def sample_p(self, t):
        sampled_phi = np.random.normal(self.phi, np.sqrt(self.P))
        z_sampled = np.sin(self.omega * t - sampled_phi)
        p_sampled = 0.5 * (1 + z_sampled)
        return p_sampled

    def predictive_variance(self, t):
        arg = self.omega * t - self.phi
        p = 0.5 * (1 + np.sin(arg))
        epistemic = 0.25 * np.cos(arg)**2 * self.P
        aleatoric = p * (1 - p)
        return epistemic + 0.1*aleatoric

    def get_phase(self):
        return self.phi, 1.0  # amplitude is fixed to 1
    

# ------------------------------
# Simulation
# ------------------------------
def simulate(omega, true_phi, n_steps=200):
    ekf = EKFArm(omega=omega, phi0=0.0)
    phi_est = []
    pred_var = []
    phi_true_series = []

    for t in range(n_steps):
        p = 0.5 * (1 + np.sin(omega * t - true_phi))
        x = np.random.rand() < p
        ekf.update(t, x)

        phi_est.append(ekf.phi)
        pred_var.append(ekf.predictive_variance(t))
        phi_true_series.append(true_phi)

    return np.array(phi_est), np.array(pred_var), np.array(phi_true_series)


# ------------------------------
# Plot test for a given frequency
# ------------------------------
def test_case(omega, true_phi):
    phi_est, pred_var, phi_true = simulate(omega, true_phi)

    n = len(phi_est)
    t = np.arange(n)

    # Reconstruct predicted + true sine waves
    true_wave = np.sin(omega * t - true_phi)
    pred_wave = np.sin(omega * t - phi_est[t])

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: Phase tracking
    axes[0].plot(t, phi_est, label="Estimated φ")
    axes[0].plot(t, phi_true, label="True φ")
    axes[0].set_title(f"Phase tracking — ω={omega}, φ={true_phi}")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("phase")
    axes[0].legend()

    # Right: Reconstructed sine
    axes[1].plot(t, true_wave, label="True sin wave")
    axes[1].plot(t, pred_wave, label="Predicted sin wave")
    axes[1].fill_between(t, pred_wave-1.96*np.sqrt(pred_var), pred_wave+1.96*np.sqrt(pred_var),
                         color="green", alpha=0.2)
    axes[1].set_title("Reconstructed sine wave")
    axes[1].set_xlabel("t")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    a=0.8
    test_case(omega=2*np.pi/10, true_phi=a*np.pi)
    test_case(omega=2*np.pi/20, true_phi=a*np.pi)
    test_case(omega=2*np.pi/200, true_phi=a*np.pi)