"""
Strategy template for the Morty Express Challenge.

This file provides a template for implementing your own strategy
to maximize the number of Morties saved.

The challenge: The survival probability of each planet changes over time
(based on the number of trips taken). Your strategy should adapt to these
changing conditions.
"""

from abc import ABC, abstractmethod
from api_client import SphinxAPIClient
from data_collector import DataCollector
import pandas as pd
import numpy as np
from experiments.kalman_ticket_bandit import KalmanTicketBandit
from experiments.beta_bandit import DecayingBetaBandit
from experiments.RLSArm import RLSArm
import matplotlib.pyplot as plt
from visualizations import plot_moving_average
import os
from experiments.RLS_regularized import RLSArmRegularized

class MortyRescueStrategy(ABC):
    """Abstract base class for implementing rescue strategies."""
    
    def __init__(self, client: SphinxAPIClient):
        """
        Initialize the strategy.
        
        Args:
            client: SphinxAPIClient instance
        """
        self.client = client
        self.collector = DataCollector(client)
        self.exploration_data = []
    
    def explore_phase(self, trips_per_planet: int = 30) -> pd.DataFrame:
        """
        Initial exploration phase to understand planet behaviors.
        
        Args:
            trips_per_planet: Number of trips to send to each planet
            
        Returns:
            DataFrame with exploration data
        """
        print("\n=== EXPLORATION PHASE ===")
        df = self.collector.explore_all_planets(
            trips_per_planet=trips_per_planet,
            morty_count=1  # Send 1 Morty at a time during exploration
        )
        self.exploration_data = df
        return df
    
    def analyze_planets(self) -> dict:
        """
        Analyze planet data to determine characteristics.
        
        Returns:
            Dictionary with analysis results
        """
        if len(self.exploration_data) == 0:
            raise ValueError("No exploration data available. Run explore_phase() first.")
        
        return self.collector.analyze_risk_changes(self.exploration_data)
    
    @abstractmethod
    def execute_strategy(self):
        """
        Execute the main rescue strategy.
        Must be implemented by subclasses.
        """
        pass


class SimpleGreedyStrategy(MortyRescueStrategy):
    """
    Simple greedy strategy: always pick the planet with highest recent success.
    """
    
    def execute_strategy(self, morties_per_trip: int = 3):
        """
        Execute the greedy strategy.
        
        Args:
            morties_per_trip: Number of Morties to send per trip (1-3)
        """
        print("\n=== EXECUTING GREEDY STRATEGY ===")
        
        # Get current status
        status = self.client.get_status()
        morties_remaining = status['morties_in_citadel']
        
        print(f"Starting with {morties_remaining} Morties in Citadel")
        
        # Determine best planet from exploration
        best_planet, best_planet_name = self.collector.get_best_planet(
            self.exploration_data,
            consider_trend=True
        )
        
        print(f"Best planet identified: {best_planet_name}")
        print(f"Sending all remaining Morties to {best_planet_name}...")
        
        trips_made = 0
        
        while morties_remaining > 0:
            # Determine how many to send
            morties_to_send = min(morties_per_trip, morties_remaining)
            
            # Send Morties
            result = self.client.send_morties(best_planet, morties_to_send)
            
            morties_remaining = result['morties_in_citadel']
            trips_made += 1
            
            if trips_made % 50 == 0:
                print(f"  Progress: {trips_made} trips, "
                      f"{result['morties_on_planet_jessica']} saved, "
                      f"{morties_remaining} remaining")
        
        # Final status
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Total Steps: {final_status['steps_taken']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/1000)*100:.2f}%")


class AdaptiveStrategy(MortyRescueStrategy):
    """
    Adaptive strategy: continuously monitor and switch planets if needed.
    """
    
    def execute_strategy(
        self,
        morties_per_trip: int = 3,
        reevaluate_every: int = 50
    ):
        """
        Execute the adaptive strategy.
        
        Args:
            morties_per_trip: Number of Morties to send per trip (1-3)
            reevaluate_every: Re-evaluate best planet every N trips
        """
        print("\n=== EXECUTING ADAPTIVE STRATEGY ===")
        
        status = self.client.get_status()
        morties_remaining = status['morties_in_citadel']
        
        print(f"Starting with {morties_remaining} Morties in Citadel")
        
        # Initial best planet
        current_planet, current_planet_name = self.collector.get_best_planet(
            self.exploration_data,
            consider_trend=True
        )
        
        print(f"Starting with planet: {current_planet_name}")
        
        trips_since_evaluation = 0
        total_trips = 0
        recent_results = []
        
        while morties_remaining > 0:
            # Send Morties
            morties_to_send = min(morties_per_trip, morties_remaining)
            result = self.client.send_morties(current_planet, morties_to_send)
            
            # Track recent results
            recent_results.append({
                'planet': current_planet,
                'survived': result['survived']
            })
            
            morties_remaining = result['morties_in_citadel']
            trips_since_evaluation += 1
            total_trips += 1
            
            # Re-evaluate strategy periodically
            if trips_since_evaluation >= reevaluate_every and morties_remaining > 0:
                # Check if we should switch planets
                recent_success_rate = sum(
                    r['survived'] for r in recent_results[-reevaluate_every:]
                ) / min(len(recent_results), reevaluate_every)
                
                print(f"\n  Re-evaluating at trip {total_trips}...")
                print(f"  Current planet: {current_planet_name}")
                print(f"  Recent success rate: {recent_success_rate*100:.2f}%")
                
                # TODO: Implement logic to potentially switch planets
                # For now, we stick with the same planet
                
                trips_since_evaluation = 0
            
            if total_trips % 50 == 0:
                print(f"  Progress: {total_trips} trips, "
                      f"{result['morties_on_planet_jessica']} saved")
        
        # Final status
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Total Steps: {final_status['steps_taken']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/1000)*100:.2f}%")



class UCBStratery(MortyRescueStrategy):
    """
    Upper Confidence Bound (UCB) strategy for exploration-exploitation trade-off.
    """
    def __init__(self, client: SphinxAPIClient, exploration_constant: float = 2.0):
        """
        Initialize UCB strategy.
        
        Args:
            client: SphinxAPIClient instance
            exploration_constant: Controls exploration vs exploitation trade-off
        """
        super().__init__(client)
        self.c = exploration_constant
        
        # Track successes and attempts for each action
        self.successes = {
            (0, 1): 0, (0, 2): 0, (0, 3): 0,
            (1, 1): 0, (1, 2): 0, (1, 3): 0,
            (2, 1): 0, (2, 2): 0, (2, 3): 0,
        }
        self.attempts = {
            (0, 1): 0, (0, 2): 0, (0, 3): 0,
            (1, 1): 0, (1, 2): 0, (1, 3): 0,
            (2, 1): 0, (2, 2): 0, (2, 3): 0,
        }
        
        self.total_trips = 0
        self.planet_names = ['Gazorpazorp', 'Purge Planet', 'Planet Squanch']



    def compute_ucb_scores(self) -> float:
        """
        Compute the UCB value for a given action.
        
        Args:
            action: Tuple of (planet_index, morties_sent)
        """
        ucb_scores = {}
        for key in self.successes.keys():
            if self.attempts[key] == 0:
                # Unvisited actions get infinite score
                ucb_scores[key] = float('inf')
            else:
                # UCB1 formula
                mean_reward = self.successes[key] / self.attempts[key]
                exploration_bonus = np.sqrt(
                    (self.c * np.log(self.total_trips)) / self.attempts[key]
                )
                ucb_scores[key] = mean_reward + exploration_bonus

        return ucb_scores
        

    def update_statistics(self, planet: int, morty_count: int, 
                          survived: int, total: int):
        """Update statistics based on outcome."""
        key = (planet, morty_count)
        self.successes[key] += survived
        self.attempts[key] += total
        self.total_trips += 1


    def select_action(self, morties_remaining: int) -> tuple:
        """
        Select action with highest UCB score.
        
        Args:
            morties_remaining: Number of Morties left
            
        Returns:
            Tuple of (planet_id, morty_count)
        """
        ucb_scores = self.compute_ucb_scores()
        
        # Filter valid actions
        valid_actions = {
            k: v * k[1]  # Scale by number of Morties
            for k, v in ucb_scores.items()
            if k[1] <= morties_remaining
        }
        
        best_action = max(valid_actions.items(), key=lambda x: x[1])
        return best_action[0]

    def execute_strategy(self, verbose: bool = True, report_every: int = 50):
        """Execute UCB strategy."""
        print("\n=== EXECUTING UCB STRATEGY ===")
        
        status = self.client.get_status()
        morties_remaining = status['morties_in_citadel']
        
        while morties_remaining > 0:
            # Select action
            planet, morty_count = self.select_action(morties_remaining)
            
            # Execute
            result = self.client.send_morties(planet, morty_count)
            survived = morty_count if result['survived'] else 0
            
            # Update
            self.update_statistics(planet, morty_count, survived, morty_count)
            morties_remaining = result['morties_in_citadel']
            
            # Report
            if verbose and self.total_trips % report_every == 0:
                print(f"Trip {self.total_trips}: "
                      f"{self.planet_names[planet]} × {morty_count}, "
                      f"Saved: {result['morties_on_planet_jessica']}")
        
        # Final results
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/1000)*100:.2f}%")

class KalmanStrategy(MortyRescueStrategy):
    
    def __init__(self, client: SphinxAPIClient):
        super().__init__(client)
        self.kalman_bandit = KalmanTicketBandit(n_arms=3, q=0.01, r_min=0.001)

    def explore_phase(self, trips_per_planet=0):
        # No exploration needed for this strategy
        pass

    def execute_strategy(self):
        print("\n=== EXECUTING KALMAN STRATEGY ===")
        
        status = self.client.get_status()
        morties_remaining = status['morties_in_citadel']
        trips_made = 0

        estimates = []
        uncertainties = []

        planet = 0  # Fixed planet for this example
        morties_to_send = 1  # Send 1 Morty per trip

        while morties_remaining > 0:
            # planet, morties_to_send, _ = self.kalman_bandit.select_action()
            morties_to_send = min(morties_to_send, morties_remaining)
            
            # Send Morties
            result = self.client.send_morties(int(planet), int(morties_to_send))
            result["planet"] = int(planet)
            result["planet_name"] = self.client.get_planet_name(int(planet))
            self.collector.trips_data.append(result)

            # Update bandit state
            self.kalman_bandit.update(planet, morties_to_send, result["survived"])
            estimates.append(self.kalman_bandit.mu.copy())
            uncertainties.append(self.kalman_bandit.P.copy())

            morties_remaining = result['morties_in_citadel']
            trips_made += 1
            
            if trips_made % 50 == 0:
                print(f"  Progress: {trips_made} trips, "
                      f"{result['morties_on_planet_jessica']} saved, "
                      f"{morties_remaining} remaining")

        if os.path.exists("data") == False:
            os.makedirs("data")
        if os.path.exists("plots") == False:
            os.makedirs("plots")

        self.collector.save_data("data/kalman_real_0.csv")

        # Plot kalman estimates
        estimates = np.array(estimates)
        uncertainties = np.array(uncertainties)

        i = 0
        plt.plot(estimates[:, i], label=f"Estimate (arm {i})", color=f"C{i}")
        plt.fill_between(range(trips_made),
                            estimates[:, i] - np.sqrt(uncertainties[:, i]),
                            estimates[:, i] + np.sqrt(uncertainties[:, i]),
                            alpha=0.2, color=f"C{i}")
        plt.title("Estimated Success Probabilities")
        plt.legend(loc="best")
        plt.grid(True)

        plt.tight_layout()
        #plt.show()
        plt.savefig("plots/kalman_estimates_0.png")

        # Final status
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Total Steps: {final_status['steps_taken']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/1000)*100:.2f}%")

class DecayingBetaStrategy(MortyRescueStrategy):

    def __init__(self, client: SphinxAPIClient):
        super().__init__(client)
        self.beta_bandit = DecayingBetaBandit(
            n_arms=3, decay=0.02, exploration_constant=2.0, min_alpha_beta=0.0
        )

    def explore_phase(self, trips_per_planet=0):
        # No exploration needed for this strategy
        pass

    def execute_strategy(self):
        print("\n=== EXECUTING DECAYING-BETA STRATEGY ===")
        
        status = self.client.get_status()
        morties_remaining = status['morties_in_citadel']
        trips_made = 0

        while morties_remaining > 0:
            planet, morties_to_send, _ = self.beta_bandit.select_action()
            morties_to_send = min(morties_to_send, morties_remaining)
            
            # Send Morties
            result = self.client.send_morties(int(planet), int(morties_to_send))
            result["planet"] = int(planet)
            result["planet_name"] = self.client.get_planet_name(int(planet))
            self.collector.trips_data.append(result)

            # Update bandit state
            self.beta_bandit.update(planet, result["survived"])

            morties_remaining = result['morties_in_citadel']
            trips_made += 1
            
            if trips_made % 50 == 0:
                print(f"  Progress: {trips_made} trips, "
                      f"{result['morties_on_planet_jessica']} saved, "
                      f"{morties_remaining} remaining")

        # Final status
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Total Steps: {final_status['steps_taken']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/1000)*100:.2f}%")


class RLSStrategy(MortyRescueStrategy):

    def __init__(self, client: SphinxAPIClient, sampling_window=1, forgetting=0.9999, pessimism=0.3, tu=0.75, te=0.82):
        super().__init__(client)
        T = [10, 20, 200]
        # ESTIMATED_PHIS = np.array([-0.18064311, -0.06519789,  0.65443256])
        # ESTIMATED_THETAS = [np.array((np.cos(phi), -np.sin(phi))) for phi in ESTIMATED_PHIS]
        self.arms = [RLSArm(omega=2*np.pi/period, forgetting=forgetting) for period in T]
        self.sampling_window = sampling_window
        self.pessimism = pessimism
        # tu: threshold for uncertain exploitation (2 morties)
        # te: threshold for exploitation (3 morties)
        self.tu = tu
        self.te = te

    def explore_phase(self, trips_per_planet = 5, scaling = [1, 2, 8]):
        trips = [s*trips_per_planet for s in scaling]
        all_data = []
        for planet, n in enumerate(trips):
            df = self.collector.explore_planet(planet, n, morty_count=1)
            all_data.append(df)
            survived = df[["steps_taken", "survived"]].to_numpy()
            for t, s in survived:
                self.arms[planet].update(t, s)
        all_data = pd.concat(all_data, ignore_index=True)
        self.exploration_data = all_data
        return all_data
    
    def morty_activation(self, prob):
        if prob < self.tu:
            return 1
        elif prob < self.te:
            return 2
        else:
            return 3
    
    def execute_strategy(self):
        print("\n=== EXECUTING RLS STRATEGY ===")
        
        status = self.client.get_status()
        morties_remaining = status['morties_in_citadel']
        steps_taken = status['steps_taken']
        trips_made = 0

        choices = [[],[],[]]
        preds = [[arm.predict_p(steps_taken) for arm in self.arms]]
        vars = [[arm.predict_p_variance(steps_taken) for arm in self.arms]]
        new_sample = 0

        while morties_remaining > 0:
            # Thompson sampling
            if new_sample % self.sampling_window == 0:
                samples = []
                for arm in self.arms:
                    # approximate posterior: theta ~ N(theta, sigma2 * P)
                    sigma2 = arm.est_noise_var()
                    cov = sigma2 * arm.P
                    # print(cov)
                    # draw theta_sample
                    try:
                        theta_samp = np.random.multivariate_normal(arm.theta, cov)
                    except Exception:
                        theta_samp = arm.theta.copy()
                    # compute p_hat from sampled theta
                    xvec = arm.feature(steps_taken)
                    z_samp = xvec.dot(theta_samp)
                    p_samp = 0.5*(1.0 + z_samp)
                    # p_est = arm.predict_p(steps_taken)
                    # var_est = arm.predict_p_variance(steps_taken)
                    # print(var_est)
                    # p_samp = np.random.normal(p_est, np.sqrt(var_est))
                    samples.append(p_samp)
                planet = int(np.argmax(samples))

            new_sample += 1
            p_estimate = self.arms[planet].predict_p(steps_taken)
            sig_estimate = np.sqrt(self.arms[planet].predict_p_variance(steps_taken))
            # proportional to the estimate, pessimistic in terms of variance
            morties_to_send = self.morty_activation(p_estimate - self.pessimism * sig_estimate)
            morties_to_send = min(morties_to_send, morties_remaining)
            
            choices[planet].append((steps_taken, morties_to_send))

            # Send Morties
            result = self.client.send_morties(planet, int(morties_to_send))
            result["planet"] = planet
            result["planet_name"] = self.client.get_planet_name(planet)
            steps_taken = result["steps_taken"]
            self.collector.trips_data.append(result)

            # Update bandit state
            self.arms[planet].update(steps_taken-1, result["survived"])

            preds.append([arm.predict_p(steps_taken) for arm in self.arms])
            vars.append([arm.predict_p_variance(steps_taken) for arm in self.arms])

            morties_remaining = result['morties_in_citadel']
            trips_made += 1
            
            if trips_made % 50 == 0:
                print(f"  Progress: {trips_made} trips, "
                      f"{result['morties_on_planet_jessica']} saved, "
                      f"{morties_remaining} remaining")

        # Final status
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Total Steps: {final_status['steps_taken']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/1000)*100:.2f}%")

        preds = np.array(preds)
        stds = np.sqrt(np.array(vars))
        choices = [np.array(c) for c in choices]
        ts = np.arange(0, preds.shape[0])

        fig, axs = plt.subplots(3, 1, figsize=(10,12))

        for i in range(3):
            c = choices[i]
            axs[i].scatter(c[:,0], preds[c[:,0],i], label="Actions",
                           color=[f"C{color}" for color in c[:,1]], alpha=0.33*c[:,1])
            axs[i].plot(ts, preds[:,i], label="Estimated signal", color="blue")
            axs[i].fill_between(ts, preds[:, i] - 1.96*stds[:, i], preds[:, i] + 1.96*stds[:, i],
                            color="orange", alpha=0.3, label="95% CI")
            axs[i].set_title("Estimated sine wave with confidence bounds")
            axs[i].set_xlabel("t")
            axs[i].set_ylabel("x")
            axs[i].set_ylim(-0.5, 1.5)
            axs[i].legend()
            axs[i].grid()

        # plt.show()
        plt.savefig("plots/RLS_estimates.svg")


class RLSStrategyConservative(MortyRescueStrategy):

    def __init__(self, client: SphinxAPIClient, 
                 sampling_window=1, 
                 forgetting=0.9995,      
                 pessimism=0.45,         # Slightly lowered
                 tu=0.78,                # Slightly lowered
                 te=0.85):               
        super().__init__(client)
        
        T = [10, 20, 200]
        
        self.arms = [RLSArmRegularized(omega=2*np.pi/period, 
                                    forgetting=forgetting,
                                    regularization=1e-6) 
                     for period in T]
        
        self.sampling_window = sampling_window
        self.pessimism = pessimism
        self.tu = tu
        self.te = te

    def morty_activation(self, prob, variance, morties_remaining, trips_made):
        
        # Calculate adjusted probability with pessimism
        adjusted_prob = prob - self.pessimism * np.sqrt(variance)
        
        # SMALL scarcity bonus only when >700 morties remain
        if morties_remaining > 700:
            scarcity_bonus = 0.03  # Very conservative
        elif morties_remaining > 500:
            scarcity_bonus = 0.02
        else:
            scarcity_bonus = 0.0
        
        adjusted_prob += scarcity_bonus
        adjusted_prob = np.clip(adjusted_prob, 0.0, 1.0)
        
        # Use configured thresholds
        if adjusted_prob > self.te:
            return 3
        elif adjusted_prob > self.tu:
            return 2
        else:
            return 1

    def explore_phase(self, trips_per_planet=5, scaling=[1, 2, 8]):
        trips = [s*trips_per_planet for s in scaling]
        all_data = []
        for planet, n in enumerate(trips):
            df = self.collector.explore_planet(planet, n, morty_count=1)
            all_data.append(df)
            survived = df[["steps_taken", "survived"]].to_numpy()
            for t, s in survived:
                self.arms[planet].update(t, s)
        all_data = pd.concat(all_data, ignore_index=True)
        self.exploration_data = all_data
        return all_data

    def execute_strategy(self):
        print("\n=== EXECUTING RLS CONSERVATIVE STRATEGY ===")
        
        status = self.client.get_status()
        morties_remaining = status['morties_in_citadel']
        steps_taken = status['steps_taken']
        trips_made = 0

        choices = [[], [], []]
        preds = [[arm.predict_p(steps_taken) for arm in self.arms]]
        vars = [[arm.predict_p_variance(steps_taken) for arm in self.arms]]
        new_sample = 0

        while morties_remaining > 0:
            if new_sample % self.sampling_window == 0:
                samples = []
                for arm in self.arms:
                    theta_samp = arm.sample_theta()
                    
                    xvec = arm.feature(steps_taken)
                    z_samp = xvec.dot(theta_samp)
                    p_samp = np.clip(0.5*(1.0 + z_samp), 0.01, 0.99)
                    samples.append(p_samp)
                
                planet = int(np.argmax(samples))

            new_sample += 1
            p_estimate = self.arms[planet].predict_p(steps_taken)
            variance_estimate = self.arms[planet].predict_p_variance(steps_taken)
            
            morties_to_send = self.morty_activation(
                p_estimate, 
                variance_estimate,
                morties_remaining,
                trips_made
            )
            morties_to_send = min(morties_to_send, morties_remaining)
            
            choices[planet].append((steps_taken, morties_to_send))

            # Send Morties
            result = self.client.send_morties(planet, int(morties_to_send))
            result["planet"] = planet
            result["planet_name"] = self.client.get_planet_name(planet)
            steps_taken = result["steps_taken"]
            self.collector.trips_data.append(result)

            # Update bandit state
            self.arms[planet].update(steps_taken-1, result["survived"])

            preds.append([arm.predict_p(steps_taken) for arm in self.arms])
            vars.append([arm.predict_p_variance(steps_taken) for arm in self.arms])

            morties_remaining = result['morties_in_citadel']
            trips_made += 1
            
            if trips_made % 50 == 0:
                print(f"  Progress: {trips_made} trips, "
                      f"{result['morties_on_planet_jessica']} saved, "
                      f"{morties_remaining} remaining")

        # Final status
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Total Steps: {final_status['steps_taken']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/1000)*100:.2f}%")

        # Visualization
        preds = np.array(preds)
        stds = np.sqrt(np.array(vars))
        choices = [np.array(c) if len(c) > 0 else np.array([]).reshape(0, 2) for c in choices]
        ts = np.arange(0, preds.shape[0])

        fig, axs = plt.subplots(3, 1, figsize=(12, 12))

        planet_names = ["Planet A (T=10)", "Planet B (T=20)", "Planet C (T=200)"]
        
        for i in range(3):
            c = choices[i]
            if len(c) > 0:
                axs[i].scatter(c[:,0], preds[c[:,0].astype(int), i], 
                              label="Actions",
                              c=c[:,1], cmap='RdYlGn', 
                              alpha=0.6, s=50, vmin=1, vmax=3)
            axs[i].plot(ts, preds[:,i], label="Estimated signal", 
                       color="blue", linewidth=2)
            axs[i].fill_between(ts, 
                               preds[:, i] - 1.96*stds[:, i], 
                               preds[:, i] + 1.96*stds[:, i],
                               color="orange", alpha=0.3, label="95% CI")
            axs[i].set_title(f"{planet_names[i]} - Estimated Survival Pattern")
            axs[i].set_xlabel("Time Step")
            axs[i].set_ylabel("Survival Probability")
            axs[i].set_ylim(-0.1, 1.1)
            axs[i].legend(loc='upper right')
            axs[i].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("plots/RLS_conservative_improved.svg")
        print("\nVisualization saved to: plots/RLS_conservative_improved.svg")




def run_strategy(strategy_class, explore_trips: int = 30):
    """
    Run a complete strategy from exploration to execution.
    
    Args:
        strategy_class: Strategy class to use
        explore_trips: Number of exploration trips per planet
    """
    # Initialize client and strategy
    client = SphinxAPIClient()
    strategy = strategy_class(client)
    
    # Start new episode
    print("Starting new episode...")
    client.start_episode()
    
    if explore_trips > 0:
        # Exploration phase
        strategy.explore_phase(trips_per_planet=explore_trips)
        
        # Analyze results
        analysis = strategy.analyze_planets()
        print("\nPlanet Analysis:")
        for planet_name, data in analysis.items():
            print(f"  {planet_name}: {data['overall_survival_rate']:.2f}% "
                f"({data['trend']})")
    
    # Execute strategy
    strategy.execute_strategy()


if __name__ == "__main__":
    print("Morty Express Challenge - Strategy Module")
    print("="*60)
    
    print("\nAvailable strategies:")
    print("1. SimpleGreedyStrategy - Pick best planet and stick with it")
    print("2. AdaptiveStrategy - Monitor and adapt to changing conditions")
    
    print("\nExample usage:")
    print("  run_strategy(SimpleGreedyStrategy, explore_trips=30)")
    print("  run_strategy(AdaptiveStrategy, explore_trips=30)")
    
    print("\nTo create your own strategy:")
    print("1. Subclass MortyRescueStrategy")
    print("2. Implement the execute_strategy() method")
    print("3. Use self.client to interact with the API")
    print("4. Use self.collector to analyze data")
    
    # Uncomment to run:
    # run_strategy(UCBStratery, explore_trips=30)

    # client = SphinxAPIClient()
    # strategy = KalmanStrategy(client)
    # # client.start_episode()
    # # strategy.execute_strategy()
    # df = strategy.collector.load_data(filename="data/kalman_real_0.csv")
    # plot_moving_average(df, window=50, save_path="plots/real_0.png")

    # run_strategy(DecayingBetaStrategy, explore_trips=0)

    run_strategy(RLSStrategyConservative, explore_trips=0)
