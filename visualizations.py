"""
Visualization functions for the Morty Express Challenge.
"""
import matplotlib
# matplotlib.use('TkAgg')  # ou 'Qt5Agg'

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_survival_rates(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot survival rates over time for each planet.
    
    Args:
        df: DataFrame with trip data
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1'}
    
    for planet in df['planet'].unique():
        planet_data = df[df['planet'] == planet].copy()
        planet_name = planet_data['planet_name'].iloc[0]
        
        # Calculate cumulative survival rate
        planet_data['cumulative_survival'] = (
            planet_data['survived'].astype(int).cumsum() / 
            range(1, len(planet_data) + 1)
        ) * 100
        
        ax.plot(
            planet_data['steps_taken'],
            planet_data['cumulative_survival'],
            label=planet_name,
            color=colors.get(planet, '#95E1D3'),
            linewidth=2,
            marker='o',
            markersize=3,
            alpha=0.7
        )
    
    ax.set_xlabel('Steps Taken', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Survival Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Planet Survival Rates Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_survival_by_planet(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot bar chart comparing overall survival rates by planet.
    
    Args:
        df: DataFrame with trip data
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate survival rate by planet
    survival_by_planet = df.groupby('planet_name')['survived'].agg(['mean', 'count'])
    survival_by_planet['survival_rate'] = survival_by_planet['mean'] * 100
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax.bar(
        survival_by_planet.index,
        survival_by_planet['survival_rate'],
        color=colors,
        alpha=0.7,
        edgecolor='black',
        linewidth=1.5
    )
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.1f}%',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    ax.set_ylabel('Survival Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Planet', fontsize=12, fontweight='bold')
    ax.set_title('Overall Survival Rates by Planet', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_moving_average(df: pd.DataFrame, window: int = 10, save_path: Optional[str] = None):
    """
    Plot moving average of survival rates for each planet.
    
    Args:
        df: DataFrame with trip data
        window: Window size for moving average
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1'}
    
    for planet in df['planet'].unique():
        planet_data = df[df['planet'] == planet].copy()
        planet_name = planet_data['planet_name'].iloc[0]
        
        # Calculate moving average
        planet_data['ma'] = (
            planet_data['survived'].astype(int)
            .rolling(window=window, min_periods=1)
            .mean() * 100
        )
        
        # Plot raw data points
        ax.scatter(
            planet_data['steps_taken'],
            planet_data['survived'].astype(int) * 100,
            color=colors.get(planet, '#95E1D3'),
            alpha=0.2,
            s=20
        )
        
        # Plot moving average
        ax.plot(
            planet_data['steps_taken'],
            planet_data['ma'],
            label=f"{planet_name} (MA-{window})",
            color=colors.get(planet, '#95E1D3'),
            linewidth=2.5,
            alpha=0.8
        )
    
    ax.set_xlabel('Steps Taken', fontsize=12, fontweight='bold')
    ax.set_ylabel('Survival Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Moving Average ({window} trips) of Survival Rates', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_moving_average_multiple_runs(df: pd.DataFrame, window: int = 10, save_path: Optional[str] = None):
    """
    Plot moving average of survival rates for each planet (multiple runs),
    aggregating survival across runs per (planet, trip), and including
    a confidence interval using correct variance averaging.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1'}

    # --- Aggregate across runs: survival mean and std per trip ---
    reduced = (
        df.groupby(["planet", "trip_number"])
        .agg(survived_mean=("survived", "mean"),
             survived_std=("survived", "std"))
        .reset_index()
    )
    
    # Convert std to variance for later window operations
    reduced["survived_var"] = reduced["survived_std"] ** 2
    
    # --- Process each planet ---
    for planet in reduced["planet"].unique():
        
        pdata = reduced[reduced["planet"] == planet].sort_values("trip_number").copy()
        planet_name = df.loc[df["planet"] == planet, "planet_name"].iloc[0] \
                      if "planet_name" in df.columns else f"Planet {planet}"
        
        # ----- Moving average of mean survival -----
        ma_mean = pdata["survived_mean"].rolling(window, min_periods=1).mean()
        
        # ----- Moving average of variance -----
        ma_var = pdata["survived_var"].rolling(window, min_periods=1).mean()
        
        ma_std = ma_var ** 0.5          # convert variance to std
        
        # ----- Confidence Interval -----
        k = 1.96                         # 95% CI
        upper = ma_mean + k * ma_std
        lower = ma_mean - k * ma_std
        
        # Convert to percentages
        ma_mean_pct = ma_mean * 100
        upper_pct = upper * 100
        lower_pct = lower * 100

        # ----- Plot moving average -----
        ax.plot(
            pdata["trip_number"],
            ma_mean_pct,
            label=f"{planet_name} (MA-{window})",
            color=colors.get(planet, '#95E1D3'),
            linewidth=2.5
        )

        # ----- Plot confidence interval -----
        ax.fill_between(
            pdata["trip_number"],
            lower_pct,
            upper_pct,
            color=colors.get(planet, '#95E1D3'),
            alpha=0.2
        )
        
        # Optional: scatter raw per-trip means
        ax.scatter(
            pdata["trip_number"],
            pdata["survived_mean"] * 100,
            color=colors.get(planet, '#95E1D3'),
            alpha=0.4,
            s=25
        )
    
    # --- Axis + formatting ---
    ax.set_xlabel('Trip', fontsize=12, fontweight='bold')
    ax.set_ylabel('Survival Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Moving Average ({window} trips) of Survival Rates (Across Runs)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_risk_evolution(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot how risk evolves over time for each planet (early vs late trips).
    
    Args:
        df: DataFrame with trip data
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1'}
    
    for idx, planet in enumerate(sorted(df['planet'].unique())):
        planet_data = df[df['planet'] == planet].copy()
        planet_name = planet_data['planet_name'].iloc[0]
        
        # Split into thirds for early, mid, late
        n = len(planet_data)
        third = n // 3
        
        periods = ['Early', 'Middle', 'Late']
        survival_rates = []
        
        for i, period in enumerate(periods):
            start = i * third
            end = (i + 1) * third if i < 2 else n
            period_data = planet_data.iloc[start:end]
            survival_rates.append(period_data['survived'].mean() * 100)
        
        # Create bar plot for this planet
        ax = axes[idx]
        bars = ax.bar(periods, survival_rates, color=colors[planet], alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.1f}%',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_title(planet_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Survival Rate (%)', fontsize=10, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Risk Evolution Over Time (Early, Middle, Late Trips)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_episode_summary(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create a comprehensive dashboard with multiple plots.
    
    Args:
        df: DataFrame with trip data
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    colors = {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1'}
    
    # 1. Survival rates over time
    ax1 = fig.add_subplot(gs[0, :])
    for planet in df['planet'].unique():
        planet_data = df[df['planet'] == planet]
        planet_name = planet_data['planet_name'].iloc[0]
        ax1.plot(
            planet_data['steps_taken'],
            planet_data['survived'].astype(int).cumsum() / range(1, len(planet_data) + 1) * 100,
            label=planet_name,
            color=colors[planet],
            linewidth=2
        )
    ax1.set_xlabel('Steps Taken')
    ax1.set_ylabel('Cumulative Survival Rate (%)')
    ax1.set_title('Survival Rates Over Time', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Overall survival rates
    ax2 = fig.add_subplot(gs[1, 0])
    survival_by_planet = df.groupby('planet_name')['survived'].mean() * 100
    bars = ax2.bar(survival_by_planet.index, survival_by_planet.values, 
                   color=[colors[p] for p in sorted(df['planet'].unique())], alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel('Survival Rate (%)')
    ax2.set_title('Overall Survival Rates', fontweight='bold')
    ax2.set_ylim(0, 100)
    
    # 3. Number of trips per planet
    ax3 = fig.add_subplot(gs[1, 1])
    trips_by_planet = df.groupby('planet_name').size()
    ax3.bar(trips_by_planet.index, trips_by_planet.values,
            color=[colors[p] for p in sorted(df['planet'].unique())], alpha=0.7)
    ax3.set_ylabel('Number of Trips')
    ax3.set_title('Trips per Planet', fontweight='bold')
    
    # 4. Final status
    ax4 = fig.add_subplot(gs[2, :])
    if len(df) > 0:
        last_status = df.iloc[-1]
        categories = ['In Citadel', 'On Jessica', 'Lost']
        values = [
            last_status['morties_in_citadel'],
            last_status['morties_on_planet_jessica'],
            last_status['morties_lost']
        ]
        colors_status = ['#FFD93D', '#6BCF7F', '#FF6B6B']
        
        wedges, texts, autotexts = ax4.pie(
            values, labels=categories, autopct='%1.1f%%',
            colors=colors_status, startangle=90
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax4.set_title(f"Final Status (Total: {sum(values)} Morties)", fontweight='bold')
    
    fig.suptitle('Morty Express Challenge - Episode Summary', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def create_all_visualizations(df: pd.DataFrame, output_dir: str = "plots"):
    """
    Create and save all visualizations.
    
    Args:
        df: DataFrame with trip data
        output_dir: Directory to save plots
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating visualizations...")
    
    plot_survival_rates(df, save_path=f"{output_dir}/survival_rates.png")
    plot_survival_by_planet(df, save_path=f"{output_dir}/survival_by_planet.png")
    plot_moving_average(df, window=10, save_path=f"{output_dir}/moving_average.png")
    plot_risk_evolution(df, save_path=f"{output_dir}/risk_evolution.png")
    plot_episode_summary(df, save_path=f"{output_dir}/episode_summary.png")
    
    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    print("Visualization module loaded!")
    print("\nAvailable functions:")
    print("  - plot_survival_rates(df)")
    print("  - plot_survival_by_planet(df)")
    print("  - plot_moving_average(df, window=10)")
    print("  - plot_risk_evolution(df)")
    print("  - plot_episode_summary(df)")
    print("  - create_all_visualizations(df, output_dir='plots')")

    df = pd.read_csv("data/explore_planets.csv")
    plot_moving_average(df, window=10)
    # plot_moving_average_multiple_runs(df, window=50)

