"""
Utility functions and helpers for the Morty Express Challenge.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


def save_results(results: Dict, filename: Optional[str] = None) -> str:
    """
    Save episode results to a JSON file.
    
    Args:
        results: Dictionary with episode results
        filename: Optional filename. If not provided, generates timestamp-based name
        
    Returns:
        Path to saved file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"
    
    # Add timestamp to results
    results['saved_at'] = datetime.now().isoformat()
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")
    return filename


def load_results(filename: str) -> Dict:
    """
    Load episode results from a JSON file.
    
    Args:
        filename: Path to JSON file
        
    Returns:
        Dictionary with episode results
    """
    with open(filename, 'r') as f:
        results = json.load(f)
    
    print(f"Results loaded from {filename}")
    return results


def calculate_success_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate various success metrics from trip data.
    
    Args:
        df: DataFrame with trip data
        
    Returns:
        Dictionary with calculated metrics
    """
    metrics = {}
    
    # Overall metrics
    metrics['total_trips'] = len(df)
    metrics['total_morties_sent'] = df['morties_sent'].sum()
    metrics['total_survivors'] = df['survived'].sum()
    metrics['overall_survival_rate'] = (metrics['total_survivors'] / metrics['total_trips']) * 100
    
    # Per-planet metrics
    metrics['by_planet'] = {}
    for planet in df['planet'].unique():
        planet_data = df[df['planet'] == planet]
        planet_name = planet_data['planet_name'].iloc[0]
        
        metrics['by_planet'][planet_name] = {
            'trips': len(planet_data),
            'survival_rate': planet_data['survived'].mean() * 100,
            'survivors': planet_data['survived'].sum(),
            'deaths': len(planet_data) - planet_data['survived'].sum()
        }
    
    # Final status
    if len(df) > 0:
        last_row = df.iloc[-1]
        metrics['final_status'] = {
            'morties_in_citadel': last_row['morties_in_citadel'],
            'morties_on_planet_jessica': last_row['morties_on_planet_jessica'],
            'morties_lost': last_row['morties_lost'],
            'steps_taken': last_row['steps_taken']
        }
    
    return metrics


def print_metrics(metrics: Dict):
    """
    Pretty print success metrics.
    
    Args:
        metrics: Dictionary with metrics from calculate_success_metrics
    """
    print("\n" + "="*60)
    print("EPISODE METRICS")
    print("="*60)
    
    print(f"\nOverall:")
    print(f"  Total Trips: {metrics['total_trips']}")
    print(f"  Total Morties Sent: {metrics['total_morties_sent']}")
    print(f"  Overall Survival Rate: {metrics['overall_survival_rate']:.2f}%")
    
    print(f"\nBy Planet:")
    for planet_name, planet_metrics in metrics['by_planet'].items():
        print(f"\n  {planet_name}:")
        print(f"    Trips: {planet_metrics['trips']}")
        print(f"    Survival Rate: {planet_metrics['survival_rate']:.2f}%")
        print(f"    Survivors: {planet_metrics['survivors']}")
        print(f"    Deaths: {planet_metrics['deaths']}")
    
    if 'final_status' in metrics:
        print(f"\nFinal Status:")
        status = metrics['final_status']
        print(f"  Morties in Citadel: {status['morties_in_citadel']}")
        print(f"  Morties on Planet Jessica: {status['morties_on_planet_jessica']}")
        print(f"  Morties Lost: {status['morties_lost']}")
        print(f"  Steps Taken: {status['steps_taken']}")
        
        success_rate = (status['morties_on_planet_jessica'] / 1000) * 100
        print(f"\n    Success Rate: {success_rate:.2f}%")
    
    print("\n" + "="*60)


def compare_strategies(results_files: List[str]):
    """
    Compare results from multiple strategy runs.
    
    Args:
        results_files: List of paths to results JSON files
    """
    import pandas as pd
    
    comparisons = []
    
    for filepath in results_files:
        if os.path.exists(filepath):
            results = load_results(filepath)
            
            comparison = {
                'file': os.path.basename(filepath),
                'saved': results['morties_on_planet_jessica'],
                'lost': results['morties_lost'],
                'success_rate': (results['morties_on_planet_jessica'] / 1000) * 100,
                'steps': results['steps_taken']
            }
            
            comparisons.append(comparison)
    
    if comparisons:
        df = pd.DataFrame(comparisons)
        df = df.sort_values('success_rate', ascending=False)
        
        print("\n" + "="*60)
        print("STRATEGY COMPARISON")
        print("="*60)
        print(df.to_string(index=False))
        print("\n" + "="*60)
    else:
        print("No valid results files found")

def print_episode_summary(final_status: Dict):
    """
    Print a formatted summary of the episode.
    
    Args:
        final_status: Final status dictionary from API
    """
    saved = final_status['morties_on_planet_jessica']
    lost = final_status['morties_lost']
    in_citadel = final_status['morties_in_citadel']
    steps = final_status['steps_taken']
    
    success_rate = (saved / 1000) * 100
    
    print("\n" + "="*60)
    print("  EPISODE COMPLETE!")
    print("="*60)
    
    print(f"\n  Final Results:")
    print(f"     Morties Saved: {saved}/1000 ({success_rate:.2f}%)")
    print(f"     Morties Lost: {lost}")
    print(f"     Remaining in Citadel: {in_citadel}")
    print(f"     Steps Taken: {steps}")
    
    print("\n" + "="*60)

def create_experiment_log():
    """
    Create a new experiment log file to track different attempts.
    
    Returns:
        Path to log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"experiment_log_{timestamp}.txt"
    
    with open(log_file, 'w') as f:
        f.write("MORTY EXPRESS CHALLENGE - EXPERIMENT LOG\n")
        f.write("="*60 + "\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
    
    print(f"Experiment log created: {log_file}")
    return log_file


def log_experiment(log_file: str, experiment_name: str, results: Dict):
    """
    Add an experiment entry to the log.
    
    Args:
        log_file: Path to log file
        experiment_name: Name/description of the experiment
        results: Results dictionary
    """
    with open(log_file, 'a') as f:
        f.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Morties Saved: {results.get('morties_on_planet_jessica', 'N/A')}\n")
        f.write(f"Morties Lost: {results.get('morties_lost', 'N/A')}\n")
        f.write(f"Steps Taken: {results.get('steps_taken', 'N/A')}\n")
        
        if 'morties_on_planet_jessica' in results:
            success_rate = (results['morties_on_planet_jessica'] / 1000) * 100
            f.write(f"Success Rate: {success_rate:.2f}%\n")
        
        f.write("-" * 60 + "\n")
    
    print(f"Experiment logged to {log_file}")


# Planet name mapping
PLANET_NAMES = {
    0: '"On a Cob" Planet',
    1: "Cronenberg World",
    2: "The Purge Planet"
}


if __name__ == "__main__":
    print("Utilities module loaded!")
    print("\nAvailable functions:")
    print("  - save_results(results, filename)")
    print("  - load_results(filename)")
    print("  - calculate_success_metrics(df)")
    print("  - print_metrics(metrics)")
    print("  - compare_strategies(results_files)")
    print("  - estimate_leaderboard_position(success_rate)")
    print("  - print_episode_summary(final_status)")
    print("  - create_experiment_log()")
    print("  - log_experiment(log_file, experiment_name, results)")
