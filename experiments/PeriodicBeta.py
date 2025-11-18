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
    """
    This class is a first attempt to integrate the analysis of the pre exploration in the choice making.
    It modelize each planet as a sinus function.
    The results aren't good because the phase and other sinus parameters are fixed all along the process (after pre exploration)
    """
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
        print("LEARNING PERIODIC PATTERNS FROM CSV")
        print("="*60)
        
        if isinstance(exploration_df, str):
            exploration_df = pd.read_csv(exploration_df)
        if 'run' in exploration_df.columns:
            run_id = exploration_df['run'].iloc[0]
            exploration_df = exploration_df[exploration_df['run'] == run_id].copy()
            print(f"\nUsing only run {run_id} for period detection ({len(exploration_df)} trips)")
        
        for arm in range(self.n_arms):
            arm_data = exploration_df[exploration_df['planet'] == arm].copy()
            
            if len(arm_data) == 0:
                print(f"\nWarning: No data for planet {arm}")
                continue
            
            arm_data = arm_data.sort_values('trip_number').reset_index(drop=True)
            
            time_series = arm_data['survived'].astype(float).values
            
            print(f"\nPlanet {arm} - analyzing {len(time_series)} trips")
            
            # Detect period using FFT with KNOWN ranges
            if arm == 0:
                # Planet 0 should be ~10
                period = self._detect_period(time_series, min_period=5, max_period=15)
            elif arm == 1:
                # Planet 1 should be ~20
                period = self._detect_period(time_series, min_period=15, max_period=30)
            elif arm == 2:
                # Planet 2 should be ~200
                period = self._detect_period(time_series, min_period=150, max_period=250)
            
            self.period[arm] = period
            
            # Estimate offset (mean survival rate)
            self.offset[arm] = np.mean(time_series)
            
            # Estimate amplitude using the detected period
            self.amplitude[arm] = self._estimate_amplitude(time_series, period)
            
            # Estimate phase shift
            self.phase[arm] = self._estimate_phase(time_series, period, self.offset[arm], self.amplitude[arm])
            
            planet_name = arm_data['planet_name'].iloc[0]
            print(f"\nPlanet {arm} - {planet_name}:")
            print(f"  Period: {period:.1f} trips (expected: {[10, 20, 200][arm]})")
            print(f"  Amplitude: {self.amplitude[arm]:.3f}")
            print(f"  Offset (mean): {self.offset[arm]:.3f}")
            print(f"  Phase: {self.phase[arm]:.3f} rad ({np.degrees(self.phase[arm]):.1f}°)")
            
            x = np.arange(len(time_series))
            predicted = self.offset[arm] + self.amplitude[arm] * np.sin(
                2 * np.pi * x / period + self.phase[arm]
            )
            mse = np.mean((time_series - predicted) ** 2)
            r2 = 1 - (np.sum((time_series - predicted)**2) / np.sum((time_series - np.mean(time_series))**2))
            print(f"  Model fit: MSE={mse:.4f}, R²={r2:.4f}")
        
        self.has_periodic_info = True
        print("\n" + "="*60)

    # Estimate amplitude more accurately by fitting the sine wave
    def _estimate_amplitude(self, time_series, period):
        if period <= 0 or len(time_series) < period:
            return (np.max(time_series) - np.min(time_series)) / 2
        
        # Use multiple cycles if available
        n_cycles = int(len(time_series) / period)
        if n_cycles >= 2:
            cycle_amplitudes = []
            for i in range(n_cycles):
                start = int(i * period)
                end = int((i + 1) * period)
                cycle = time_series[start:end]
                if len(cycle) > 0:
                    cycle_amp = (np.max(cycle) - np.min(cycle)) / 2
                    cycle_amplitudes.append(cycle_amp)
            
            if cycle_amplitudes:
                return np.mean(cycle_amplitudes)
        
        return (np.max(time_series) - np.min(time_series)) / 2
    
    # Detect dominant period using FFT with specified range
    def _detect_period(self, time_series, min_period=5, max_period=250):
        n = len(time_series)
        
        print(f"  Searching for periods in range [{min_period}, {max_period}]")
        
        # Apply FFT
        yf = fft(time_series - np.mean(time_series))
        xf = fftfreq(n, 1)[:n//2]
        amplitudes = 2.0/n * np.abs(yf[0:n//2])
        
        periods = np.zeros_like(xf)
        periods[1:] = 1.0 / xf[1:]
        
        # Filter to valid period range
        mask = (periods >= min_period) & (periods <= max_period)
        
        if not np.any(mask):
            print(f"  Warning: No periods found in range, using midpoint")
            return (min_period + max_period) / 2
        
        valid_amplitudes = amplitudes.copy()
        valid_amplitudes[~mask] = 0
        
        # Find peak
        peak_idx = np.argmax(valid_amplitudes)
        peak_freq = xf[peak_idx]
        
        if peak_freq == 0:
            return (min_period + max_period) / 2
        
        detected_period = 1.0 / peak_freq
        
        top_indices = np.argsort(valid_amplitudes)[-5:][::-1]
        print(f"  Top 5 detected periods in range:")
        for i, idx in enumerate(top_indices, 1):
            if xf[idx] > 0:
                p = 1.0 / xf[idx]
                print(f"    {i}. Period={p:.1f} trips, Amplitude={amplitudes[idx]:.4f}")
        
        return detected_period
    
    def _estimate_phase(self, time_series, period, offset, amplitude):
        # can be useful in case we want to modelise the variation 
        if period <= 0:
            return 0.0
        n = len(time_series)
        x = np.arange(n)
        # Find phase that minimizes error
        best_phase = 0.0
        min_error = float('inf')
        for phase in np.linspace(0, 2 * np.pi, 100):
            predicted = offset + amplitude * np.sin(2 * np.pi * x / period + phase)
            error = np.mean((time_series - predicted) ** 2)
            if error < min_error:
                min_error = error
                best_phase = phase
        return best_phase
    
    def predict_current_rate(self, arm):
        if not self.has_periodic_info:
            return 0.5  
        predicted_rate = self.offset[arm] + self.amplitude[arm] * np.sin(
            2 * np.pi * self.t / self.period[arm] + self.phase[arm]
        )
        
        # Clamp to [0, 1]
        predicted_rate = np.clip(predicted_rate, 0.0, 1.0)
        
        return predicted_rate
    
    def mean_estimates(self):
        beta_means = self.alpha / (self.alpha + self.beta)
        
        if not self.has_periodic_info:
            return beta_means
        
        # Blend: start with more beta, transition to more periodic
        #blend_weight = min(0.7, self.t / 500) 
        blend_weight = 0.6   

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
        if self.has_periodic_info:
            periodic_preds = np.array([self.predict_current_rate(i) for i in range(self.n_arms)])
            samples = periodic_preds + np.random.normal(0, 0.1, size=self.n_arms)
        else:
            samples = np.random.beta(self.alpha, self.beta)

        
        exploration_bonus = self.c * np.sqrt(self.variance_estimates() + 1e-6)
        
        n_tickets = np.clip((means * 3).astype(int) + 1, 1, 3)
        
        scores = (samples + exploration_bonus) * n_tickets
        #arm = np.argmax(samples + exploration_bonus)
        arm = np.argmax(scores)
        return int(arm), int(n_tickets[arm]), samples
    
    def update(self, arm, reward):
        # Decay old evidence
        self.alpha = np.maximum(self.min_alpha_beta, self.alpha * (1 - self.decay))
        self.beta = np.maximum(self.min_alpha_beta, self.beta * (1 - self.decay))
        
        success = 1 if reward > 0 else 0
        self.alpha[arm] += success
        self.beta[arm] += (1 - success)
        
        self.t += 1





