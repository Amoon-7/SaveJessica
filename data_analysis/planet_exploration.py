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
from datetime import datetime

TRIPS_PER_PLANET = 1000
NUM_PLANETS = 3

# explore each planet by sending one Morty each trip and plot the survival rates
def explore_planets(save_path=None, plot=True):
    client = SphinxAPIClient()
    
    all_trips = []
    for planet_id in range(NUM_PLANETS):
        print(f"\n{'='*60}")
        print(f"Starting new episode for Planet {planet_id}")
        print(f"{'='*60}")
        client.start_episode()
        
        data_collector = DataCollector(client)

        planet_trips = data_collector.explore_planet(planet_id, TRIPS_PER_PLANET, morty_count=1)
        all_trips.append(planet_trips)  
    
    df = pd.concat(all_trips, ignore_index=True)
    df["timestamp"] = datetime.now()

    # Vérifier que le DataFrame n'est pas vide
    if len(df) == 0:
        print("Error: No data collected!")
        return
    
    if save_path is not None:
        df.to_csv(save_path, mode="a+", header=not os.path.exists(save_path))

    if plot:
        plot_moving_average(df, window=50)

        # plt.figure(figsize=(12, 6))
        # for planet_id in range(NUM_PLANETS):
        #     planet_data = df[df['planet'] == planet_id]
            
        #     if len(planet_data) == 0:
        #         print(f"Warning: No data for planet {planet_id}")
        #         continue
                
        #     planet_name = planet_data['planet_name'].iloc[0]
            
        #     # Reset index pour le moving average
        #     planet_data_reset = planet_data.reset_index(drop=True)
        #     plot_moving_average(planet_data_reset, window=50)
        
        #     plt.title('Moving Average of Survival Rates per Planet')
        #     plt.xlabel('Trip Number')
        #     plt.ylabel('Survival Rate')
        #     plt.legend()
        #     plt.grid()
        #     plt.show()

    return df


# round robin exploration of planets 
def round_robin_exploration(save_path=None, plot=True):
    client = SphinxAPIClient()

    all_trips = []
    
    for trip_num in range(TRIPS_PER_PLANET * NUM_PLANETS):
        if trip_num % 1000 == 0:
            print(f"\n{'='*60}")
            print(f"trip_num % 1000 == 0 -> Restarting client")
            print(f"{'='*60}")
            client.start_episode()
            data_collector = DataCollector(client)

        planet_id = trip_num % NUM_PLANETS
        print(f"\n{'='*60}")
        print(f"Trip {trip_num + 1}: Exploring Planet {planet_id}")
        print(f"{'='*60}")
        
        planet_trips = data_collector.explore_planet(planet_id, 1, morty_count=1)
        all_trips.append(planet_trips)  
    
    df = pd.concat(all_trips, ignore_index=True)
    df["timestamp"] = datetime.now()
    
    if save_path is not None:
        df.to_csv(save_path, mode="a+", header=not os.path.exists(save_path))

    if plot:
        plot_moving_average(df, window=50)

        # plt.figure(figsize=(12, 6))
        # for planet_id in range(NUM_PLANETS):
        #     planet_data = df[df['planet'] == planet_id]
        #     planet_name = planet_data['planet_name'].iloc[0]
            
        #     # Reset index pour le moving average
        #     planet_data_reset = planet_data.reset_index(drop=True)
        #     plot_moving_average(planet_data_reset['survived'], window_size=50, label=planet_name)
        
        #     plt.title('Moving Average of Survival Rates per Planet (Round Robin)')
        #     plt.xlabel('Trip Number')
        #     plt.ylabel('Survival Rate')
        #     plt.legend()
        #     plt.grid()
        #     plt.show()


# averaging on multipe runs of explore planets
def multiple_runs_exploration(save_path=None, num_runs=5, plot=True):
    all_runs = []
    initial_run = 0
    if save_path is not None:
        df = pd.read_csv(save_path)
        initial_run = df['run'].max()  # +1-1
    for run in range(initial_run, initial_run+num_runs):
        print(f"\n{'#'*60}")
        print(f"Starting Run {run + 1} of {num_runs}")
        print(f"{'#'*60}\n")
        df = explore_planets(plot=False)
        df['run'] = run+1
        if save_path is not None:
            df.to_csv(save_path, mode="a+", header=not os.path.exists(save_path))
        all_runs.append(df)
    
    combined_df = pd.concat(all_runs, ignore_index=True)

    if plot:
        plot_moving_average(combined_df, window=50)

        # plt.figure(figsize=(12, 6))
        # for planet_id in range(NUM_PLANETS):
        #     planet_data = combined_df[combined_df['planet'] == planet_id]
        #     planet_name = planet_data['planet_name'].iloc[0]
            
        #     # Reset index pour le moving average
        #     planet_data_reset = planet_data.reset_index(drop=True)
        #     plot_moving_average(planet_data_reset['survived'], window_size=50, label=planet_name)

        #     plt.title('Moving Average of Survival Rates per Planet (Multiple Runs)')
        #     plt.xlabel('Trip Number')
        #     plt.ylabel('Survival Rate')
        #     plt.legend()
        #     plt.grid()
        #     plt.show()


if __name__ == "__main__":
    # explore_planets(save_path="../data/explore_planets.csv")
    # round_robin_exploration(save_path="../data/round_robin_exploration.csv")
    multiple_runs_exploration(save_path="../data/multiple_runs_exploration.csv", num_runs=10)