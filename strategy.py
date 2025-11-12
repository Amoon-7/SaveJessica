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
import matplotlib.pyplot as plt
from visualizations import plot_moving_average

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

        while morties_remaining > 0:
            planet, morties_to_send, _ = self.kalman_bandit.select_action()
            morties_to_send = min(morties_to_send, morties_remaining)
            
            # Send Morties
            result = self.client.send_morties(int(planet), int(morties_to_send))
            result["planet"] = int(planet)
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

        self.collector.save_data("data/kalman_real.csv")

        # Plot kalman estimates
        print(estimates)
        estimates = np.array(estimates)
        uncertainties = np.array(uncertainties)

        for i in range(3):
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
        plt.savefig("plots/kalman_estimates.png")

        # Final status
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Total Steps: {final_status['steps_taken']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/1000)*100:.2f}%")

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
    run_strategy(KalmanStrategy, explore_trips=0)
    # run_strategy(UCBStratery, explore_trips=30)

    collector = DataCollector(SphinxAPIClient())
    df = collector.load_data(filename="data/kalman_real.csv")
    plot_moving_average(df)
