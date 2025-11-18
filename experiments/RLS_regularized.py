import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RLSArmRegularized:
    """ 
    This version is an improved RLS arm with enhanced numerical stability.
    Key improvements:
    - Regularization added to covariance matrix P in order to prevent singularity
    - Stable theta sampling using Cholesky decomposition
    These enhancements ensure mainly stable performance. They don't improve the results too much.
    """
    def __init__(self, omega, forgetting=1.0, delta=1, regularization=1e-6):
        # theta = [alpha, beta] for z ~ alpha*sin + beta*cos
        self.omega = omega
        self.theta = np.zeros(2)
        self.P = np.eye(2) * delta
        self.lambda_ = forgetting
        self.n = 0
        self.sse = 0.0  
        self.regularization = regularization  # For numerical stability
        
        self.residual_history = []
        self.max_history = 50
    
    def feature(self, t):
        return np.array([np.sin(self.omega * t), np.cos(self.omega * t)])
    
    def update(self, t, x):
        # x in {0,1}
        z = 2*x - 1
        xvec = self.feature(t)
        Px = self.P.dot(xvec)
        denom = self.lambda_ + xvec.dot(Px)
        K = Px / denom
        resid = z - xvec.dot(self.theta)
        self.theta = self.theta + K * resid
        
        self.P = (self.P - np.outer(K, xvec).dot(self.P)) / self.lambda_
        self.P += self.regularization * np.eye(2)  
        
        self.n += 1
        self.sse += resid**2
        
        # Track residuals
        self.residual_history.append(resid**2)
        if len(self.residual_history) > self.max_history:
            self.residual_history.pop(0)
    
    def predict_z(self, t):
        xvec = self.feature(t)
        z_pred = xvec.dot(self.theta)
        # Clip to valid range for z ∈ [-1, 1]
        return np.clip(z_pred, -0.99, 0.99)
    
    def predict_p(self, t):
        zhat = self.predict_z(t)
        # Map to probability space [0, 1]
        return 0.5 * (1.0 + zhat)
    
    def predict_p_variance(self, t, include_observation_noise=False):
        xvec = self.feature(t)
        theta_var = xvec @ self.P @ xvec
        
        # Include observation noise scaled to p
        sigma2 = self.est_noise_var() if include_observation_noise else 0.0
        total_var = 0.25 * (theta_var + sigma2)  # scale to p
        return total_var
    
    # mainly for phase analysis
    def get_phase(self):
        alpha, beta = self.theta
        # z = alpha*sin + beta*cos = R sin(omega t - phi)
        phi = np.arctan2(-beta, alpha)
        R = np.sqrt(alpha*alpha + beta*beta)
        return phi, R
    
    def est_noise_var(self):
        if len(self.residual_history) > 5:
            return np.median(self.residual_history)
        elif self.n > 2:
            return self.sse / max(1, self.n - 2)
        else:
            return 1.0
    
    def get_stable_covariance(self):
        sigma2 = self.est_noise_var()
        cov = sigma2 * self.P
        cov_stable = cov + 1e-6 * np.eye(2)
        return cov_stable
    
    def sample_theta(self):
        """ Sample from posterior distribution of theta """
        try:
            cov = self.get_stable_covariance()
            L = np.linalg.cholesky(cov)
            theta_sample = self.theta + L @ np.random.randn(2)
            return theta_sample
        except np.linalg.LinAlgError:
            # Fallback to mean if sampling fails
            return self.theta.copy()


MA_WINDOW = 4
def moving_average(a, n=MA_WINDOW):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n-1:] / n
    return np.pad(ret, (n//2, n - n//2 - 1), mode='constant', constant_values=np.nan)


if __name__ == "__main__":
    df = pd.read_csv("../data/multiple_regularized_runs_exploration.csv")
    run = 2
    sampling_period = 1
    T = 1000
    planet = df[(df['planet'] == 0) & (df['run'] == run)][['trip_number', "survived"]].to_numpy()
    frequency = 2*np.pi/10

    arm = RLSArmRegularized(frequency, regularization=1e-6)
    ts = []
    preds = []
    vars = []
    
    for t, x in planet[::sampling_period]:
        ts.append(t)
        arm.update(t, float(x))
        preds.append(arm.predict_p(t))
        vars.append(arm.predict_p_variance(t))

    preds = np.array(preds)
    stds = np.sqrt(np.array(vars))

    plt.figure(figsize=(10,4))
    plt.plot(np.arange(T), moving_average(planet[:,1]), label="Moving average", alpha=0.7)
    plt.plot(ts, preds, label="RLS Improved Estimate", linewidth=2)
    plt.fill_between(ts, preds - 1.96*stds, preds + 1.96*stds,
                     color="orange", alpha=0.3, label="95% CI")
    plt.title("Improved RLS Estimation with Confidence Bounds")
    plt.xlabel("Time Step")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(-0.1, 1.1)
    plt.show()
