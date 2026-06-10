"""
Visualization script for simulation results
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

results_dir = 'results'

def plot_comparison(period_name):
    """Plot comparison for a specific period"""

    filename_base = period_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')

    baseline_file = f'{results_dir}/{filename_base}_baseline.csv'
    mpc_file = f'{results_dir}/{filename_base}_mpc.csv'

    if not os.path.exists(baseline_file) or not os.path.exists(mpc_file):
        print(f"Results not found for {period_name}")
        return

    df_baseline = pd.read_csv(baseline_file)
    df_mpc = pd.read_csv(mpc_file)

    # Plot first 7 days
    n_hours = min(168, len(df_baseline))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Room temperature
    axes[0].plot(df_baseline['T_room'][:n_hours], label='Baseline', linewidth=2)
    axes[0].plot(df_mpc['T_room'][:n_hours], label='MPC', linewidth=2, alpha=0.8)
    axes[0].axhline(20, color='r', linestyle='--', alpha=0.5, label='Comfort min')
    axes[0].axhline(24, color='r', linestyle='--', alpha=0.5, label='Comfort max')
    axes[0].set_ylabel('Room Temp [°C]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'{period_name} - First 7 Days Comparison')

    # Supply temperature
    axes[1].plot(df_baseline['T_supply'][:n_hours], label='Baseline', linewidth=2)
    axes[1].plot(df_mpc['T_supply'][:n_hours], label='MPC', linewidth=2, alpha=0.8)
    axes[1].set_ylabel('Supply Temp [°C]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Heat output
    axes[2].plot(df_baseline['Q_heat'][:n_hours]/1000, label='Baseline', linewidth=2)
    axes[2].plot(df_mpc['Q_heat'][:n_hours]/1000, label='MPC', linewidth=2, alpha=0.8)
    axes[2].set_ylabel('Heat Output [kW]')
    axes[2].set_xlabel('Hour')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{results_dir}/{filename_base}_comparison.png', dpi=150)
    print(f"✅ Saved: {results_dir}/{filename_base}_comparison.png")
    plt.close()


if __name__ == '__main__':
    print("Creating comparison plots...")
    periods = [
        'Winter 2024',
        'Year 2024',
        'Year 2025',
        'Year 2026 (Jan-May)'
    ]

    for period in periods:
        plot_comparison(period)

    print("\n✅ All plots created!")
