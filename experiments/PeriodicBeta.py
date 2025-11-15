import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from api_client import SphinxAPIClient
from data_collector import DataCollector
from experiments.kalman_ticket_bandit import KalmanTicketBandit
from experiments.beta_bandit import DecayingBetaBandit
import matplotlib.pyplot as plt
from visualizations import plot_moving_average
from scipy.fft import fft, fftfreq

class PeriodicBetaBandit:
    def __init__(self, n_arms, decay=0.02, exploration_constant=2.0, min_alpha_beta=0.0):
        self.n_arms = n_arms
        self.decay = decay
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
        self.c = exploration_constant
        self.min_alpha_beta = min_alpha_beta
        
        # Periodic parameters (from pre-exploration)
        self.period = np.zeros(n_arms)
        self.amplitude = np.zeros(n_arms)
        self.phase = np.zeros(n_arms)
        self.offset = np.ones(n_arms) * 0.5
        self.has_periodic_info = False
        
        # Current time step
        self.t = 0
    
    def load_periodic_params(self, exploration_df):
        print("\n" + "="*60)
        print("LEARNING PERIODIC PATTERNS")
        print("="*60)
        
        for arm in range(self.n_arms):
            arm_data = exploration_df[exploration_df['planet'] == arm]
            
            if len(arm_data) == 0:
                continue
            
            time_series = arm_data['survived'].values.astype(float)
            
            period = self._detect_period(time_series, min_period=50, max_period=300)
            self.period[arm] = period
            
            self.offset[arm] = np.mean(time_series)
            self.amplitude[arm] = (np.max(time_series) - np.min(time_series)) / 2
            
            self.phase[arm] = self._estimate_phase(time_series, period)
            
            planet_name = arm_data['planet_name'].iloc[0]
        
        self.has_periodic_info = True
        print("\n" + "="*60)
    
    def _detect_period(self, time_series, min_period=50, max_period=300):
        n = len(time_series)
        
        # FFT
        yf = fft(time_series - np.mean(time_series))
        xf = fftfreq(n, 1)[:n//2]
        amplitudes = 2.0/n * np.abs(yf[0:n//2])
        
        periods = np.zeros_like(xf)
        periods[1:] = 1.0 / xf[1:]
        
        mask = (periods >= min_period) & (periods <= max_period)
        
        if not np.any(mask):
            return 100.0  
        
        valid_amplitudes = amplitudes.copy()
        valid_amplitudes[~mask] = 0
        
        peak_idx = np.argmax(valid_amplitudes)
        peak_freq = xf[peak_idx]
        
        if peak_freq == 0:
            return 100.0
        
        return 1.0 / peak_freq
    
    def _estimate_phase(self, time_series, period):
        # can be useful in case we want to modelise the variation 
        pass
    
    def predict_current_rate(self, arm):
        # trying to figure out this 
        pass
    
    def mean_estimates(self):
        beta_means = self.alpha / (self.alpha + self.beta)
        
        if not self.has_periodic_info:
            return beta_means
        
        # Blend: start with more beta, transition to more periodic
        blend_weight = min(0.7, self.t / 500) 
        
        adjusted_means = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            periodic_pred = self.predict_current_rate(arm)
            adjusted_means[arm] = (1 - blend_weight) * beta_means[arm] + blend_weight * periodic_pred
        
        return adjusted_means
    
    def variance_estimates(self):
        s = self.alpha + self.beta
        return (self.alpha * self.beta) / (s*s*(s+1))
    
    def select_action(self):
        # Get periodic-adjusted means
        means = self.mean_estimates()
        
        # Thompson sampling with periodic adjustment
        samples = np.random.beta(self.alpha, self.beta)
        
        if self.has_periodic_info:
            periodic_preds = np.array([self.predict_current_rate(i) for i in range(self.n_arms)])
            blend = 0.5
            samples = (1 - blend) * samples + blend * periodic_preds
        
        exploration_bonus = self.c * np.sqrt(self.variance_estimates() + 1e-6)
        
        # Number of tickets based on predicted current rate
        n_tickets = np.clip((means * 3).astype(int) + 1, 1, 3)
        
        arm = np.argmax(samples + exploration_bonus)
        
        return int(arm), int(n_tickets[arm]), samples
    
    def update(self, arm, reward):
        # Decay old evidence
        self.alpha = np.maximum(self.min_alpha_beta, self.alpha * (1 - self.decay))
        self.beta = np.maximum(self.min_alpha_beta, self.beta * (1 - self.decay))
        
        success = 1 if reward > 0 else 0
        self.alpha[arm] += success
        self.beta[arm] += (1 - success)
        
        self.t += 1



