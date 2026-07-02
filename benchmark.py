# -*- coding: utf-8 -*-
"""
Benchmark: PID + HeatCurve vs RL Agent
=======================================
Runs classical control baselines (HeatCurve-only and PID+HeatCurve) through
the same RoomHeatEnv used by the RL agent, so all comparisons share identical
building physics, gains, weather, and price data.

IMPORTANT: controllers are run through OUR simulator, not the MPC team's
separate IntegratedSystem/BufferTank stack. Their simulator has different
building parameters (H_tr=185 W/K vs our 281 W/K) and a buffer tank we
don't model, so running their controllers through their simulator would
produce incomparable results. Using RoomHeatEnv as the common ground is
the only fair basis for comparison.

Usage
-----
  # Baselines only:
  python benchmark.py --data data/test_heating_2024_25.csv --days 180

  # Baselines + RL model side-by-side:
  python benchmark.py --data data/test_heating_2024_25.csv --days 180
                      --model runs/sac_v10/best_model.zip

  # Save results:
  python benchmark.py --data data/test_heating_2024_25.csv --days 180
                      --model runs/sac_v10/best_model.zip
                      --save results/benchmark_heating_24_25
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from src.gains import build_gains_series, DEFAULT_PROFILES_DIR
from src.room_env import RoomHeatEnv
from src.heatcurve import HeatCurveController
from src.pid import PIDController
from models.vonovia_model import vonovia_model

T_HP_MIN = 20.0
T_HP_MAX = 65.0


# ── Action conversion ─────────────────────────────────────────────────────────

def supply_temp_to_action(T_supply: float) -> np.ndarray:
    """Convert supply temperature [°C] to normalized action in [-1, 1]."""
    action = 2.0 * (T_supply - T_HP_MIN) / (T_HP_MAX - T_HP_MIN) - 1.0
    return np.array([np.clip(action, -1.0, 1.0)], dtype=np.float32)


# ── Controller runners ────────────────────────────────────────────────────────

def run_heatcurve(data: pd.DataFrame, days: int, seed: int = 0) -> pd.DataFrame:
    """
    Pure heat-curve baseline: supply temperature set entirely by outdoor
    temperature, no feedback from actual room temperature.

    Parameters corrected for this building:
    - T_supply_nom=55°C (not 45°C): at T_amb_design=-12°C this building
      needs ~45°C supply just to cover heat losses; 55°C gives the curve
      enough slope to deliver adequate supply at mild temperatures too.
    - clip to [20, 65]°C (not 20-50°C): the MPC team's heatcurve.py clips
      at 50°C assuming underfloor heating, but this building's HP operates
      up to 65°C and needs 45-55°C supply on cold days. The 50°C cap caused
      massive underheating (66-80% of heating-period hours below 20°C).
    """
    hc = HeatCurveController(
        T_supply_nom=55.0,    # corrected from 45°C -- see docstring
        T_return_nom=45.0,
        T_amb_design=-12.0,   # Hannover norm outdoor temp
        T_room_set=21.0,      # midpoint of [20, 22] comfort band
    )

    env = RoomHeatEnv(disturbances=data, days=days, random_init=False,
                      forecast_steps=24)
    obs, _ = env.reset(seed=seed)
    done = False
    records = []

    while not done:
        T_amb = float(env.p.iloc[env.t]['T_amb'])
        if T_amb >= vonovia_model['T_amb_lim']:
            action = supply_temp_to_action(T_HP_MIN)
        else:
            T_sup = float(np.clip(hc.calc_supply_temp(T_amb), T_HP_MIN, T_HP_MAX))
            action = supply_temp_to_action(T_sup)

        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        records.append(info)

    ep = pd.DataFrame(records)
    ep.index = data.index[:len(ep)]
    return ep


def run_pid_heatcurve(data: pd.DataFrame, days: int, seed: int = 0,
                       Kp: float = 3.0, Ki: float = 0.05,
                       Kd: float = 0.02) -> pd.DataFrame:
    """
    PID + HeatCurve baseline: heat curve provides feedforward supply
    temperature from T_amb; PID adds a correction term based on the
    error between actual T_room and the 21°C setpoint.

    Same parameter corrections as run_heatcurve (T_supply_nom=55°C,
    clip to env range). PID correction is clamped to ±10°C around the
    heat-curve base so the curve still drives the main load.
    """
    hc = HeatCurveController(
        T_supply_nom=55.0, T_return_nom=45.0,
        T_amb_design=-12.0, T_room_set=21.0,
    )
    pid = PIDController(
        Kp=Kp, Ki=Ki, Kd=Kd,
        setpoint=21.0,
        output_limits=(T_HP_MIN, T_HP_MAX),
    )
    pid.reset()

    env = RoomHeatEnv(disturbances=data, days=days, random_init=False,
                      forecast_steps=24)
    obs, _ = env.reset(seed=seed)
    done = False
    records = []

    while not done:
        T_amb  = float(env.p.iloc[env.t]['T_amb'])
        T_room = float(env._phys_state['T_room'])

        if T_amb >= vonovia_model['T_amb_lim']:
            pid.reset()
            action = supply_temp_to_action(T_HP_MIN)
        else:
            T_base     = float(hc.calc_supply_temp(T_amb))
            pid_output = pid.update(measured_value=T_room, dt=3600.0)
            correction = np.clip(pid_output - 21.0, -10.0, 10.0)
            T_sup      = float(np.clip(T_base + correction, T_HP_MIN, T_HP_MAX))
            action     = supply_temp_to_action(T_sup)

        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        records.append(info)

    ep = pd.DataFrame(records)
    ep.index = data.index[:len(ep)]
    return ep


def run_rl_model(model_path: str, data: pd.DataFrame,
                 days: int, seed: int = 0) -> pd.DataFrame:
    """Run the RL agent through the same env for a direct side-by-side."""
    from stable_baselines3 import SAC
    env = RoomHeatEnv(disturbances=data, days=days, random_init=False,
                      forecast_steps=24)
    model = SAC.load(model_path.replace('.zip', ''), env=env)
    obs, _ = env.reset(seed=seed)
    done = False
    records = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        records.append(info)
    ep = pd.DataFrame(records)
    ep.index = data.index[:len(ep)]
    return ep


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(ep: pd.DataFrame, label: str) -> dict:
    T = ep['T_room']
    hp_kWh    = ep['Qdot_th_kW'].sum()
    gains_kWh = ep['Qdot_gains_kW'].sum() if 'Qdot_gains_kW' in ep.columns else 0.0
    n_hours   = len(ep)
    ann       = lambda x: x / n_hours * 8760
    n_cycles  = int((pd.Series(ep['hp_on'].astype(int)).diff() == 1).sum())

    return {
        'label':            label,
        'comfort_pct':      round(100 * T.between(20, 22).mean(), 1),
        'under_pct':        round(100 * (T < 20).mean(), 1),
        'over_pct':         round(100 * (T > 22).mean(), 1),
        'T_room_mean':      round(T.mean(), 2),
        'T_room_max':       round(T.max(), 2),
        'T_room_min':       round(T.min(), 2),
        'n_cycles':         n_cycles,
        'cycles_per_year':  round(n_cycles / n_hours * 8760),
        'elec_kWh':         round(ep['E_el_kWh'].sum(), 1),
        'elec_kWh_a':       round(ann(ep['E_el_kWh'].sum())),
        'cost_eur':         round((ep['price'] * ep['E_el_kWh']).sum(), 2),
        'HP_kWh':           round(hp_kWh, 1),
        'HP_kWh_a':         round(ann(hp_kWh)),
        'gains_kWh':        round(gains_kWh, 1),
        'combined_kWh':     round(hp_kWh + gains_kWh, 1),
        'combined_kWh_a':   round(ann(hp_kWh + gains_kWh)),
    }


def print_metrics(m: dict):
    print(f"\n  {'─'*54}")
    print(f"  {m['label']}")
    print(f"  {'─'*54}")
    print(f"  Comfort %     : {m['comfort_pct']}%  "
          f"(under {m['under_pct']}%,  over {m['over_pct']}%)")
    print(f"  T_room        : mean {m['T_room_mean']}°C  "
          f"min {m['T_room_min']}°C  max {m['T_room_max']}°C")
    print(f"  HP cycles     : {m['n_cycles']}  (~{m['cycles_per_year']}/year)")
    print(f"  Electricity   : {m['elec_kWh']} kWh  (~{m['elec_kWh_a']} kWh/a)")
    print(f"  Cost          : €{m['cost_eur']}")
    print(f"  HP delivered  : {m['HP_kWh']} kWh  (~{m['HP_kWh_a']} kWh/a)")
    print(f"  HP + gains    : {m['combined_kWh']} kWh  "
          f"(~{m['combined_kWh_a']} kWh/a  [ref: 30,828])")


def print_monthly(ep: pd.DataFrame, label: str):
    print(f"\n  Monthly breakdown — {label}:")
    monthly = ep.resample('ME').apply(lambda x: pd.Series({
        'comfort%': round(100 * x['T_room'].between(20, 22).mean(), 1),
        'under%':   round(100 * (x['T_room'] < 20).mean(), 1),
        'over%':    round(100 * (x['T_room'] > 22).mean(), 1),
        'T_max':    round(x['T_room'].max(), 1),
        'cycles':   int((pd.Series(x['hp_on'].astype(int)).diff() == 1).sum()),
        'HP_kWh':   round(x['Qdot_th_kW'].sum(), 0),
    }))
    print(monthly.to_string())


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    'HeatCurve':     '#f0883e',
    'PID+HeatCurve': '#79c0ff',
    'RL Agent':      '#3fb950',
}

def plot_comparison(episodes: dict, metrics: list, save_path: str = None):
    """
    3-panel comparison plot:
      Panel 1: Room temperature for all controllers
      Panel 2: Comfort % bar chart per controller
      Panel 3: Electricity cost per controller
    """
    fig = plt.figure(figsize=(16, 11), facecolor='#0d1117')
    fig.suptitle('Controller Benchmark — Heating Period', color='#e6edf3',
                 fontsize=14, y=0.98)

    axes = fig.subplots(3, 1)
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        for sp in ax.spines.values():
            sp.set_color('#30363d')
        ax.grid(axis='y', color='#30363d', linewidth=0.5)

    # ── Panel 1: T_room traces ────────────────────────────────────────────
    ax1 = axes[0]
    for label, ep in episodes.items():
        t = ep.index
        ax1.plot(t, ep['T_room'], color=COLORS.get(label, '#e6edf3'),
                 linewidth=0.8, alpha=0.85, label=label)
    ax1.axhspan(20, 22, color='#238636', alpha=0.15, label='Comfort band')
    ax1.axhline(20, color='#3fb950', linewidth=0.6, linestyle='--', alpha=0.5)
    ax1.axhline(22, color='#3fb950', linewidth=0.6, linestyle='--', alpha=0.5)
    ax1.set_ylabel('Room Temp [°C]', color='#e6edf3')
    ax1.tick_params(axis='y', colors='#e6edf3')
    ax1.set_title('Room Temperature', color='#e6edf3', fontsize=11, pad=6)
    ax1.legend(loc='upper right', framealpha=0.3, labelcolor='#e6edf3',
               facecolor='#161b22', edgecolor='#30363d')

    # ── Panel 2: Comfort bar chart ────────────────────────────────────────
    ax2 = axes[1]
    labels  = [m['label']       for m in metrics]
    comfort = [m['comfort_pct'] for m in metrics]
    under   = [m['under_pct']   for m in metrics]
    over    = [m['over_pct']    for m in metrics]
    x = np.arange(len(labels))
    w = 0.25
    colors_list = [COLORS.get(l, '#e6edf3') for l in labels]
    bars = ax2.bar(x - w, comfort, w, label='Comfort %',
                   color=[c for c in colors_list], alpha=0.85)
    ax2.bar(x,     under,   w, label='Under 20°C %', color='#6e76ae', alpha=0.85)
    ax2.bar(x + w, over,    w, label='Over 22°C %',  color='#da3633', alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, color='#e6edf3')
    ax2.set_ylabel('% Time', color='#e6edf3')
    ax2.set_title('Comfort Performance', color='#e6edf3', fontsize=11, pad=6)
    ax2.legend(loc='upper right', framealpha=0.3, labelcolor='#e6edf3',
               facecolor='#161b22', edgecolor='#30363d')
    for bar, val in zip(bars, comfort):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom',
                 color='#e6edf3', fontsize=8)

    # ── Panel 3: Cost + cycles ────────────────────────────────────────────
    ax3 = axes[2]
    costs  = [m['cost_eur']  for m in metrics]
    cycles = [m['n_cycles']  for m in metrics]
    bars3 = ax3.bar(x - w/2, costs, w*1.5,
                    color=[COLORS.get(l, '#e6edf3') for l in labels], alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, color='#e6edf3')
    ax3.set_ylabel('Electricity Cost [€]', color='#e6edf3')
    ax3.set_title('Electricity Cost & HP Cycles', color='#e6edf3', fontsize=11, pad=6)
    for bar, val, cyc in zip(bars3, costs, cycles):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'€{val:.0f}\n{cyc} cyc', ha='center', va='bottom',
                 color='#e6edf3', fontsize=8)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.subplots_adjust(hspace=0.4)

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(f'{save_path}.png', dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"\nPlot saved → {save_path}.png")
    else:
        plt.show()
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description='Run PID+HeatCurve benchmark through RoomHeatEnv.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data',         type=str, default=None,
                   help='Test CSV (T_amb, price_eur_kwh). Uses synthetic data if omitted.')
    p.add_argument('--days',         type=int, default=180,
                   help='Episode length in days.')
    p.add_argument('--model',        type=str, default=None,
                   help='Optional RL model .zip for side-by-side comparison.')
    p.add_argument('--profiles_dir', type=str, default=DEFAULT_PROFILES_DIR,
                   help='Folder with solar_*.csv + internal_gains.csv.')
    p.add_argument('--no_gains',     action='store_true',
                   help='Disable solar/internal gains (Qdot_gains=0).')
    p.add_argument('--save',         type=str, default=None,
                   help='Base path for saving plot (no extension).')
    p.add_argument('--pid_kp',       type=float, default=3.0)
    p.add_argument('--pid_ki',       type=float, default=0.05)
    p.add_argument('--pid_kd',       type=float, default=0.02)
    p.add_argument('--seed',         type=int,   default=0)
    return p.parse_args()


def load_data(path, profiles_dir, no_gains):
    if path is not None:
        df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        df = df[~df.index.duplicated(keep='first')]
        print(f"Loaded {path}: {len(df):,} hours "
              f"({df.index[0].date()} → {df.index[-1].date()})")
    else:
        n = 181 * 24 + 48
        idx = pd.date_range('2024-10-01', periods=n, freq='h')
        t = np.arange(n)
        df = pd.DataFrame({
            'T_amb':         3 - 8*np.cos(2*np.pi*t/(24*365)),
            'price_eur_kwh': 0.25 + 0.05*np.sin(2*np.pi*t/24),
        }, index=idx)
        print("Using synthetic heating-season data")

    if not no_gains:
        try:
            gains_df = build_gains_series(df.index, profiles_dir=profiles_dir,
                                           area_floor=vonovia_model['area_floor'])
            df['Qdot_gains'] = gains_df['Qdot_gains']
            df['Q_sol_W']    = gains_df['Q_sol_W']
            print(f"Gains merged: mean {gains_df['Qdot_gains'].mean():.0f} W, "
                  f"max {gains_df['Qdot_gains'].max():.0f} W")
        except FileNotFoundError as e:
            print(f"WARNING: gains not loaded ({e}); Qdot_gains=0.")
    return df


def main():
    args = get_args()
    print('=' * 58)
    print('  CONTROLLER BENCHMARK — RoomHeatEnv')
    print('=' * 58)

    data = load_data(args.data, args.profiles_dir, args.no_gains)

    episodes = {}
    all_metrics = []

    print('\nRunning HeatCurve...')
    ep_hc = run_heatcurve(data, args.days, args.seed)
    episodes['HeatCurve'] = ep_hc
    m_hc = compute_metrics(ep_hc, 'HeatCurve')
    all_metrics.append(m_hc)
    print_metrics(m_hc)
    print_monthly(ep_hc, 'HeatCurve')

    print('\nRunning PID+HeatCurve...')
    ep_pid = run_pid_heatcurve(data, args.days, args.seed,
                                args.pid_kp, args.pid_ki, args.pid_kd)
    episodes['PID+HeatCurve'] = ep_pid
    m_pid = compute_metrics(ep_pid, 'PID+HeatCurve')
    all_metrics.append(m_pid)
    print_metrics(m_pid)
    print_monthly(ep_pid, 'PID+HeatCurve')

    if args.model:
        print(f'\nRunning RL Agent ({os.path.basename(args.model)})...')
        ep_rl = run_rl_model(args.model, data, args.days, args.seed)
        episodes['RL Agent'] = ep_rl
        m_rl = compute_metrics(ep_rl, 'RL Agent')
        all_metrics.append(m_rl)
        print_metrics(m_rl)
        print_monthly(ep_rl, 'RL Agent')

    # Summary table
    print(f"\n{'='*58}")
    print('  SUMMARY')
    print(f"{'='*58}")
    summary = pd.DataFrame(all_metrics).set_index('label')[[
        'comfort_pct', 'under_pct', 'over_pct',
        'T_room_max', 'n_cycles', 'cost_eur', 'combined_kWh_a'
    ]]
    print(summary.to_string())

    plot_comparison(episodes, all_metrics, args.save)


if __name__ == '__main__':
    main()