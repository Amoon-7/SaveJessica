## A code to explore the evolution of planets survival rates ##
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

trips_per_planet = 1000
num_planets = 3



# explore each planet by sending one Morty each trip and plot the survival rates
def explore_planets():
    client = SphinxAPIClient()
    
    
    
    all_trips = []
    for planet_id in range(num_planets):
        print(f"\n{'='*60}")
        print(f"Starting new episode for Planet {planet_id}")
        print(f"{'='*60}")
        client.start_episode()
        
        data_collector = DataCollector(client)

        planet_trips = data_collector.explore_planet(planet_id, trips_per_planet, morty_count=1)
        all_trips.append(planet_trips)  
    
    df = pd.concat(all_trips, ignore_index=True)
    
    # Vérifier que le DataFrame n'est pas vide
    if len(df) == 0:
        print("Error: No data collected!")
        return
    
    plt.figure(figsize=(12, 6))
    for planet_id in range(num_planets):
        planet_data = df[df['planet'] == planet_id]
        
        if len(planet_data) == 0:
            print(f"Warning: No data for planet {planet_id}")
            continue
            
        planet_name = planet_data['planet_name'].iloc[0]
        
        # Reset index pour le moving average
        planet_data_reset = planet_data.reset_index(drop=True)
        plot_moving_average(planet_data_reset['survived'], window_size=50, label=planet_name)
    
    plt.title('Moving Average of Survival Rates per Planet')
    plt.xlabel('Trip Number')
    plt.ylabel('Survival Rate')
    plt.legend()
    plt.grid()
    plt.show()

    return df



# round robin exploration of planets 
def round_robin_exploration():
    client = SphinxAPIClient()
    
    all_trips = []
    client.start_episode()
    data_collector = DataCollector(client)
    
    for trip_num in range(trips_per_planet * num_planets):
        planet_id = trip_num % num_planets
        print(f"\n{'='*60}")
        print(f"Trip {trip_num + 1}: Exploring Planet {planet_id}")
        print(f"{'='*60}")
        
        planet_trips = data_collector.explore_planet(planet_id, 1, morty_count=1)
        all_trips.append(planet_trips)  
    
    df = pd.concat(all_trips, ignore_index=True)
    
    plt.figure(figsize=(12, 6))
    for planet_id in range(num_planets):
        planet_data = df[df['planet'] == planet_id]
        planet_name = planet_data['planet_name'].iloc[0]
        
        # Reset index pour le moving average
        planet_data_reset = planet_data.reset_index(drop=True)
        plot_moving_average(planet_data_reset['survived'], window_size=50, label=planet_name)
    
    plt.title('Moving Average of Survival Rates per Planet (Round Robin)')
    plt.xlabel('Trip Number')
    plt.ylabel('Survival Rate')
    plt.legend()
    plt.grid()
    plt.show()




# averaging on multipe runs of explore planets
all_runs = []
def multiple_runs_exploration(num_runs=5):
    for run in range(num_runs):
        print(f"\n{'#'*60}")
        print(f"Starting Run {run + 1} of {num_runs}")
        print(f"{'#'*60}\n")
        df = explore_planets()
        all_runs.append(df)
    
    combined_df = pd.concat(all_runs, ignore_index=True)

    plt.figure(figsize=(12, 6))
    for planet_id in range(num_planets):
        planet_data = combined_df[combined_df['planet'] == planet_id]
        planet_name = planet_data['planet_name'].iloc[0]
        
        # Reset index pour le moving average
        planet_data_reset = planet_data.reset_index(drop=True)
        plot_moving_average(planet_data_reset['survived'], window_size=50, label=planet_name)

    plt.title('Moving Average of Survival Rates per Planet (Multiple Runs)')
    plt.xlabel('Trip Number')
    plt.ylabel('Survival Rate')
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    # explore_planets()
    #round_robin_exploration()
    multiple_runs_exploration(num_runs=3)