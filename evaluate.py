"""
Evaluation Script — SAC Agent for Heat Pump Control
====================================================
Runs a trained model on the environment and plots:
  - Ambient temperature (T_amb)
  - Supply temperature (T_hp_sup)
  - Room temperature (T_room) with comfort band

Usage
-----
    python src/evaluate.py --model runs/sac_hp/best_model.zip
    python src/evaluate.py --model runs/sac_hp/best_model.zip \\
                           --data data/test.csv \\
                           --days 14
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.room_env import RoomHeatEnv


# ── Argument parser ───────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model',    type=str, required=True,
                   help='Path to saved model (.zip)')
    p.add_argument('--data',     type=str, default=None,
                   help='CSV with T_amb and price_eur_kwh. Synthetic if not provided.')
    p.add_argument('--days',     type=int, default=14,
                   help='Number of days to evaluate.')
    p.add_argument('--save',     type=str, default=None,
                   help='Path to save the plot (e.g. eval.png). Shows interactively if not set.')
    p.add_argument('--T_comfort_low',  type=float, default=20.0)
    p.add_argument('--T_comfort_high', type=float, default=22.0)
    p.add_argument('--vec_normalize', type=str, default=None,
                   help='Path to saved VecNormalize statistics (.pkl). '
                        'If not set, will look for vec_normalize.pkl in the model directory.')
    p.add_argument('--bldg_idx', type=int, default=None,
                   help='Specific building index (0-10) to evaluate. If not set, randomly chosen.')
    return p.parse_args()


# ── Data ──────────────────────────────────────────────────────────────────────

def load_or_generate(path, days):
    if path is not None:
        df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        df = df[~df.index.duplicated(keep='first')]  
        print(f"Loaded {path}: {len(df):,} hours")
        return df

    n  = days * 24 + 48   # a bit extra for forecast lookahead
    idx = pd.date_range('2024-01-01', periods=n, freq='h', tz='Europe/Berlin')
    t   = np.arange(n)
    print("Using synthetic data")
    return pd.DataFrame({
        'T_amb':         2 - 8*np.cos(2*np.pi*t/(24*365)) - 4*np.cos(2*np.pi*t/24),
        'price_eur_kwh': 0.22 + 0.12*np.sin(2*np.pi*t/24) + 0.03*np.random.randn(n),
    }, index=idx)


# ── Run episode ───────────────────────────────────────────────────────────────

def run_episode(model, env):
    """Run one full episode and collect all relevant signals."""
    records = []
    obs = env.reset()
    done   = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_array, info_list = env.step(action)
        done = done_array[0]
        info = info_list[0]
        raw_env = env.envs[0].unwrapped
        records.append({
            'time':      raw_env.get_cur_time(),
            'T_room':    info['T_room'],
            'T_amb':     info['T_amb'],
            'u':         info['u'],              # T_hp_sup actually applied
            'hp_on':     info['hp_on'],          # compressor on/off (modulation floor)
            'price':     info['price'],
            'E_el_kWh':  info['E_el_kWh'],
            'dev_neg':   info['dev_neg_max'],
            'reward':    reward[0],
        })

    return pd.DataFrame(records).set_index('time')


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_results(df, T_low, T_high, save_path=None):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor('#0f1117')
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        ax.spines[:].set_color('#30363d')
        ax.yaxis.label.set_color('#e6edf3')
        ax.xaxis.label.set_color('#8b949e')

    t = df.index

    # ── Panel 1: Room temperature + comfort band ──────────────────────────────
    ax1 = axes[0]
    ax1.fill_between(t, T_low, T_high, alpha=0.15, color='#3fb950',
                     label=f'Comfort band [{T_low}–{T_high}°C]')
    ax1.axhline(T_low,  color='#3fb950', linewidth=0.8, linestyle='--', alpha=0.6)
    ax1.axhline(T_high, color='#3fb950', linewidth=0.8, linestyle='--', alpha=0.6)
    ax1.plot(t, df['T_room'], color='#58a6ff', linewidth=1.5, label='T_room')
    ax1.set_ylabel('Temperature [°C]')
    ax1.set_title('Room Temperature vs Comfort Band', color='#e6edf3',
                  fontsize=11, pad=8)
    ax1.legend(loc='upper right', framealpha=0.3,
               labelcolor='#e6edf3', facecolor='#161b22', edgecolor='#30363d')
    ax1.grid(axis='y', color='#30363d', linewidth=0.5)

    # Shade comfort violations red
    violations_under = df['T_room'] < T_low
    if violations_under.any():
        ax1.fill_between(t, df['T_room'], T_low,
                         where=violations_under, alpha=0.3, color='#f85149',
                         label='Under-heating')
    violations_over = df['T_room'] > T_high
    if violations_over.any():
        ax1.fill_between(t, T_high, df['T_room'],
                         where=violations_over, alpha=0.3, color='#ff7b72',
                         label='Over-heating')

    # ── Panel 2: Supply temperature ───────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(t, df['u'], color='#ff7b72', linewidth=1.5, label='T_hp_sup (applied)')
    ax2.set_ylabel('Supply Temp [°C]')
    ax2.set_title('HP Supply Temperature', color='#e6edf3', fontsize=11, pad=8)
    ax2.legend(loc='upper right', framealpha=0.3,
               labelcolor='#e6edf3', facecolor='#161b22', edgecolor='#30363d')
    ax2.grid(axis='y', color='#30363d', linewidth=0.5)

    # ── Panel 3: Ambient temperature ──────────────────────────────────────────
    ax3 = axes[2]
    ax3.plot(t, df['T_amb'], color='#d2a8ff', linewidth=1.5, label='T_amb')
    ax3.axhline(0, color='#8b949e', linewidth=0.6, linestyle=':')
    ax3.set_ylabel('Ambient Temp [°C]')
    ax3.set_xlabel('Time')
    ax3.set_title('Outdoor Ambient Temperature', color='#e6edf3', fontsize=11, pad=8)
    ax3.legend(loc='upper right', framealpha=0.3,
               labelcolor='#e6edf3', facecolor='#161b22', edgecolor='#30363d')
    ax3.grid(axis='y', color='#30363d', linewidth=0.5)
    ax3.xaxis.set_major_formatter(DateFormatter('%d %b'))

    # ── Summary stats ─────────────────────────────────────────────────────────
    total_cost   = (df['price'] * df['E_el_kWh']).sum()
    total_energy = df['E_el_kWh'].sum()
    in_band      = (df['T_room'] >= T_low) & (df['T_room'] <= T_high)
    pct_comfort  = 100 * in_band.mean()
    n_cycles     = int((df['hp_on'].astype(int).diff().abs() > 0).sum() / 2)

    stats = (f"Energy: {total_energy:.1f} kWh  |  "
             f"Cost: €{total_cost:.2f}  |  "
             f"Comfort: {pct_comfort:.1f}%  |  "
             f"HP cycles: {n_cycles}")
    fig.text(0.5, 0.01, stats, ha='center', color='#8b949e', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.subplots_adjust(hspace=0.35)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Plot saved → {save_path}")
    else:
        plt.show()

    return {
        'total_energy_kWh': total_energy,
        'total_cost_eur':   total_cost,
        'pct_comfort':      pct_comfort,
        'n_cycles':         n_cycles,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # Data
    data = load_or_generate(args.data, args.days)

    # Environment
    def make_env():
        return RoomHeatEnv(
            disturbances=data,
            days=args.days,
            random_init=False,
            forecast_steps=24,
            fixed_bldg_idx=args.bldg_idx,
        )
    env = DummyVecEnv([make_env])

    # Handle VecNormalize
    import os
    vec_path = args.vec_normalize
    if vec_path is None:
        # Auto-detect in model directory
        model_dir = os.path.dirname(args.model)
        possible_path = os.path.join(model_dir, 'vec_normalize.pkl')
        if os.path.exists(possible_path):
            vec_path = possible_path

    if vec_path and os.path.exists(vec_path):
        print(f"Loading VecNormalize statistics from: {vec_path}")
        env = VecNormalize.load(vec_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("WARNING: Running evaluation WITHOUT VecNormalize statistics. "
              "If the model was trained with VecNormalize, evaluation will be incorrect!")

    # Model
    print(f"Loading model: {args.model}")
    model_path = args.model.replace('.zip', '')
    model = SAC.load(model_path, env=env)

    # Run
    print(f"Running {args.days}-day episode...")
    df = run_episode(model, env)

    # Fetch building params
    bldg_params = env.get_attr('current_building_params')[0]
    bldg_name = bldg_params.get('name', f"Building {args.bldg_idx}")
    num_pumps = int(bldg_params.get('num_pumps', 1))

    # Summary
    print(f"\n=== Evaluating {bldg_name} ===")
    print(f"── Results ──────────────────────────────")

    print(f"  Steps         : {len(df)}")
    print(f"  T_room range  : {df['T_room'].min():.1f} – {df['T_room'].max():.1f} °C")
    print(f"  T_amb range   : {df['T_amb'].min():.1f} – {df['T_amb'].max():.1f} °C")
    print(f"  T_sup range   : {df['u'].min():.1f} – {df['u'].max():.1f} °C")
    print(f"  Num pumps     : {num_pumps}")

    # Plot
    stats = plot_results(df, args.T_comfort_low, args.T_comfort_high, args.save)
    print(f"  Total energy  : {stats['total_energy_kWh']:.1f} kWh")
    print(f"  Total cost    : €{stats['total_cost_eur']:.2f}")
    print(f"  Comfort %     : {stats['pct_comfort']:.1f}%")
    print(f"  HP cycles     : {stats['n_cycles']}")


if __name__ == '__main__':
    main()