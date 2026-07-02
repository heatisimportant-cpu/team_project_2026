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
from src.room_env import RoomHeatEnv
from src.gains import build_gains_series, DEFAULT_PROFILES_DIR
from models.vonovia_model import vonovia_model


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
    p.add_argument('--profiles_dir', type=str, default=DEFAULT_PROFILES_DIR,
                   help='Folder with solar_*.csv file(s) + internal_gains.csv.')
    p.add_argument('--no_gains',   action='store_true',
                   help='Disable solar/internal gains (Qdot_gains=0).')
    return p.parse_args()


# ── Data ──────────────────────────────────────────────────────────────────────

def load_or_generate(path, days, profiles_dir=None, no_gains=False):
    if path is not None:
        df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        df = df[~df.index.duplicated(keep='first')]  
        print(f"Loaded {path}: {len(df):,} hours")
    else:
        n  = days * 24 + 48   # a bit extra for forecast lookahead
        idx = pd.date_range('2024-01-01', periods=n, freq='h', tz='Europe/Berlin')
        t   = np.arange(n)
        print("Using synthetic data")
        df = pd.DataFrame({
            'T_amb':         2 - 8*np.cos(2*np.pi*t/(24*365)) - 4*np.cos(2*np.pi*t/24),
            'price_eur_kwh': 0.22 + 0.12*np.sin(2*np.pi*t/24) + 0.03*np.random.randn(n),
        }, index=idx)

    if no_gains:
        print("--no_gains set: Qdot_gains=0 (solar/internal gains disabled)")
    else:
        try:
            gains_df = build_gains_series(
                df.index, profiles_dir=profiles_dir,
                area_floor=vonovia_model['area_floor'])
            df['Qdot_gains'] = gains_df['Qdot_gains']
            df['Q_sol_W']    = gains_df['Q_sol_W']
            print(f"Gains merged from {profiles_dir}: "
                  f"mean {gains_df['Qdot_gains'].mean():.0f} W, "
                  f"max {gains_df['Qdot_gains'].max():.0f} W")
        except FileNotFoundError as e:
            print(f"WARNING: could not load gains ({e}); falling back to "
                  f"Qdot_gains=0.")
    return df


# ── Run episode ───────────────────────────────────────────────────────────────

def run_episode(model, env):
    """Run one full episode and collect all relevant signals."""
    records = []
    obs, _ = env.reset()
    done   = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        records.append({
            'time':      env.get_cur_time(),
            'T_room':    info['T_room'],
            'T_amb':     info['T_amb'],
            'u':         info['u'],              # T_hp_sup actually applied
            'hp_on':     info['hp_on'],          # compressor on/off (modulation floor)
            'price':     info['price'],
            'E_el_kWh':  info['E_el_kWh'],
            'P_el_kW':   info['P_el_kW'],        # electrical power drawn this step
            'Qdot_th_kW': info['Qdot_th_kW'],    # thermal power delivered to the building
            'Qdot_gains_kW': info['Qdot_gains_kW'],  # solar+internal gains this step
            'dev_neg':   info['dev_neg_max'],
            'reward':    reward,
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

    # Shade comfort violations (both under- and over-heating)
    viol_low  = df['T_room'] < T_low
    viol_high = df['T_room'] > T_high
    if viol_low.any():
        ax1.fill_between(t, df['T_room'], T_low,
                         where=viol_low, alpha=0.3, color='#f85149',
                         label='Under-heating')
    if viol_high.any():
        ax1.fill_between(t, df['T_room'], T_high,
                         where=viol_high, alpha=0.3, color='#d29922',
                         label='Over-heating')

    # ── Panel 2: Supply temperature + electrical power ────────────────────────
    ax2 = axes[1]
    l1, = ax2.plot(t, df['u'], color='#ff7b72', linewidth=1.5, label='T_hp_sup (applied)')
    ax2.set_ylabel('Supply Temp [°C]', color='#ff7b72')
    ax2.tick_params(axis='y', colors='#ff7b72')
    ax2.set_title('HP Supply Temperature & Electrical Power', color='#e6edf3',
                  fontsize=11, pad=8)
    ax2.grid(axis='y', color='#30363d', linewidth=0.5)

    ax2b = ax2.twinx()
    ax2b.set_facecolor('#161b22')
    ax2b.tick_params(colors='#8b949e')
    ax2b.spines[:].set_color('#30363d')
    l2, = ax2b.plot(t, df['P_el_kW'], color='#79c0ff', linewidth=1.3,
                     alpha=0.9, label='P_el (electrical power)')
    ax2b.fill_between(t, 0, df['P_el_kW'], color='#79c0ff', alpha=0.10)
    ax2b.set_ylabel('Electrical Power [kW]', color='#79c0ff')
    ax2b.tick_params(axis='y', colors='#79c0ff')
    ax2b.set_ylim(bottom=0)

    ax2.legend(handles=[l1, l2], loc='upper right', framealpha=0.3,
               labelcolor='#e6edf3', facecolor='#161b22', edgecolor='#30363d')

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
    total_thermal_kWh = df['Qdot_th_kW'].sum()   # hourly steps -> kWh == kW summed
    total_gains_kWh   = df['Qdot_gains_kW'].sum() if 'Qdot_gains_kW' in df.columns else 0.0
    total_combined_kWh = total_thermal_kWh + total_gains_kWh
    n_hours      = len(df)
    annualized_thermal_kWh  = total_thermal_kWh / n_hours * 8760.0
    annualized_gains_kWh    = total_gains_kWh / n_hours * 8760.0
    annualized_combined_kWh = total_combined_kWh / n_hours * 8760.0
    viol_low     = df['T_room'] < T_low
    viol_high    = df['T_room'] > T_high
    pct_comfort  = 100 * (1 - (viol_low | viol_high).mean())
    pct_under    = 100 * viol_low.mean()
    pct_over     = 100 * viol_high.mean()

    # Count off→on transitions (rising edges) — each one is exactly one
    # compressor start-up. This avoids the parity problem of counting
    # all transitions and dividing by 2, which undercounts whenever the
    # episode starts already "on" or ends while still "on".
    hp_on_int = df['hp_on'].astype(int)
    n_cycles  = int((hp_on_int.diff() == 1).sum())
    if hp_on_int.iloc[0] == 1:
        n_cycles += 1   # compressor was already on at the first recorded step

    stats = (f"Energy: {total_energy:.1f} kWh  |  "
             f"Cost: €{total_cost:.2f}  |  "
             f"HP+gains: {total_combined_kWh:.0f} kWh (~{annualized_combined_kWh:,.0f} kWh/a)  |  "
             f"Comfort: {pct_comfort:.1f}% (under: {pct_under:.1f}%, over: {pct_over:.1f}%)  |  "
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
        'total_energy_kWh':  total_energy,
        'total_cost_eur':    total_cost,
        'total_thermal_kWh': total_thermal_kWh,
        'total_gains_kWh':   total_gains_kWh,
        'total_combined_kWh': total_combined_kWh,
        'annualized_thermal_kWh':  annualized_thermal_kWh,
        'annualized_gains_kWh':    annualized_gains_kWh,
        'annualized_combined_kWh': annualized_combined_kWh,
        'pct_comfort':       pct_comfort,
        'pct_under':         pct_under,
        'pct_over':          pct_over,
        'n_cycles':          n_cycles,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # Data
    data = load_or_generate(args.data, args.days, args.profiles_dir, args.no_gains)

    # Environment
    env = RoomHeatEnv(
        disturbances=data,
        days=args.days,
        random_init=False,
        forecast_steps=24,
    )

    # Model
    print(f"Loading model: {args.model}")
    model_path = args.model.replace('.zip', '')
    model = SAC.load(model_path, env=env)

    # Run
    print(f"Running {args.days}-day episode...")
    df = run_episode(model, env)

    # Summary
    print(f"\n── Results ──────────────────────────────")
    print(f"  Steps         : {len(df)}")
    print(f"  T_room range  : {df['T_room'].min():.1f} – {df['T_room'].max():.1f} °C")
    print(f"  T_amb range   : {df['T_amb'].min():.1f} – {df['T_amb'].max():.1f} °C")
    print(f"  T_sup range   : {df['u'].min():.1f} – {df['u'].max():.1f} °C")

    # Plot
    stats = plot_results(df, args.T_comfort_low, args.T_comfort_high, args.save)
    print(f"  Total energy  : {stats['total_energy_kWh']:.1f} kWh")
    print(f"  Total cost    : €{stats['total_cost_eur']:.2f}")
    print(f"  HP delivered  : {stats['total_thermal_kWh']:.1f} kWh "
          f"(~{stats['annualized_thermal_kWh']:,.0f} kWh/a) -- depends on policy quality")
    print(f"  Gains (solar+internal): {stats['total_gains_kWh']:.1f} kWh "
          f"(~{stats['annualized_gains_kWh']:,.0f} kWh/a) -- policy-independent")
    print(f"  HP + gains    : {stats['total_combined_kWh']:.1f} kWh "
          f"(~{stats['annualized_combined_kWh']:,.0f} kWh/a) --  "
          f" documented annual heating demand: 30,828 kWh/a")
    print(f"  Comfort %     : {stats['pct_comfort']:.1f}%  "
          f"(under: {stats['pct_under']:.1f}%, over: {stats['pct_over']:.1f}%)")
    print(f"  HP cycles     : {stats['n_cycles']}")


if __name__ == '__main__':
    main()