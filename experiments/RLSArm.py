import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RLSArm:
    def __init__(self, omega, forgetting=1.0, delta=1):
        # theta = [alpha, beta] for z ~ alpha*sin + beta*cos
        self.omega = omega
        self.theta = np.zeros(2)
        self.P = np.eye(2) * delta
        self.lambda_ = forgetting
        self.n = 0
        self.sse = 0.0  # for noise variance estimate
    
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
        self.n += 1
        self.sse += resid**2
    
    def predict_z(self, t):
        xvec = self.feature(t)
        return xvec.dot(self.theta)
    
    def predict_p(self, t):
        zhat = self.predict_z(t)
        return 0.5 * (1.0 + zhat)
    
    def predict_p_variance(self, t, include_observation_noise=False):
        xvec = self.feature(t)
        theta_var = xvec @ self.P @ xvec
        # include observation noise scaled to p
        sigma2 = self.est_noise_var() if include_observation_noise else 0.0  # variance of e
        total_var = 0.25 * (theta_var + sigma2)  # scale to p
        return total_var
    
    def get_phase(self):
        alpha, beta = self.theta
        # recall: z = alpha*sin + beta*cos = R sin(omega t - phi)
        phi = np.arctan2(-beta, alpha)
        R = np.sqrt(alpha*alpha + beta*beta)
        return phi, R
    
    def est_noise_var(self):
        if self.n > 2:
            return self.sse / max(1, self.n - 2)
        else:
            return 1.0


MA_WINDOW=4
def moving_average(a, n=MA_WINDOW):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n-1:] / n  # get valid averages
    # optionally, pad to keep same length as input
    return np.pad(ret, (n//2, n - n//2 - 1), mode='constant', constant_values=np.nan)


if __name__ == "__main__":
    df = pd.read_csv("../data/multiple_runs_exploration.csv")
    run = 2
    sampling_period=1
    T = 1000
    planet = df[(df['planet'] == 0) & (df['run'] == run)][['trip_number', "survived"]].to_numpy()
    frequency = 2*np.pi/10

    arm = RLSArm(frequency)
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
    plt.plot(np.arange(T), moving_average(planet[:,1]), label="Moving average")
    plt.plot(ts, preds, label="Estimated signal")
    plt.fill_between(ts, preds - 1.96*stds, preds + 1.96*stds,
                     color="orange", alpha=0.3, label="95% CI")
    plt.title("Estimated sine wave with confidence bounds")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.legend()
    plt.grid()
    plt.show()