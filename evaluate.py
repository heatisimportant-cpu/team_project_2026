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
    p.add_argument('--days',     type=int, default=212,
                   help='Episode length in days. Default 212 = one full Oct–Apr heating period.')
    p.add_argument('--save',     type=str, default=None,
                   help='Base path to save outputs (no extension). '
                        'Saves <save>.png and <save>_monthly.csv.')
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
    # ── Light theme colours ───────────────────────────────────────────────────
    BG        = '#FAFAF7'    # warm off-white matching beige presentation
    AX_BG     = '#FFFFFF'
    GRID      = '#D8D4CC'
    TEXT      = '#1A1A1A'
    SUBTEXT   = '#444444'
    COMFORT   = '#1B8C2E'    # vivid green comfort band
    T_ROOM    = '#0057D9'    # strong blue room temperature
    UNDER     = '#D62728'    # strong red underheating
    OVER      = '#FF6B00'    # vivid orange overheating
    T_AMB     = '#7B2FBE'    # vivid purple ambient

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(AX_BG)
        ax.tick_params(colors=TEXT, labelsize=10)
        ax.spines[:].set_color(GRID)
        ax.yaxis.label.set_color(TEXT)
        ax.xaxis.label.set_color(SUBTEXT)

    t = df.index

    # ── Panel 1: Room temperature + comfort band ──────────────────────────────
    ax1 = axes[0]
    ax1.fill_between(t, T_low, T_high, alpha=0.25, color=COMFORT,
                     label=f'Comfort band [{T_low}–{T_high}°C]')
    ax1.axhline(T_low,  color=COMFORT, linewidth=1.8, linestyle='--', alpha=1.0)
    ax1.axhline(T_high, color=COMFORT, linewidth=1.8, linestyle='--', alpha=1.0)
    ax1.plot(t, df['T_room'], color=T_ROOM, linewidth=1.2, label='Room temperature')


    ax1.set_ylabel('Room Temperature [°C]', color=TEXT, fontsize=11)
    ax1.set_title('Room Temperature vs Comfort Band', color=TEXT, fontsize=12, pad=8)
    ax1.legend(loc='upper right', framealpha=0.8, labelcolor=TEXT,
               facecolor=AX_BG, edgecolor=GRID, fontsize=9)
    ax1.grid(axis='y', color=GRID, linewidth=0.6)

    # ── Panel 2: Ambient temperature ─────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(t, df['T_amb'], color=T_AMB, linewidth=1.2, label='Outdoor temperature')
    ax2.axhline(0, color=SUBTEXT, linewidth=0.7, linestyle=':', alpha=0.6)
    ax2.axhline(vonovia_model['T_amb_lim'], color=COMFORT, linewidth=1.8,
                linestyle='--', alpha=1.0,
                label=f"Heating limit ({vonovia_model['T_amb_lim']}°C)")
    ax2.set_ylabel('Outdoor Temperature [°C]', color=TEXT, fontsize=11)
    ax2.set_xlabel('Date', color=SUBTEXT, fontsize=10)
    ax2.set_title('Outdoor Ambient Temperature', color=TEXT, fontsize=12, pad=8)
    ax2.legend(loc='upper right', framealpha=0.8, labelcolor=TEXT,
               facecolor=AX_BG, edgecolor=GRID, fontsize=9)
    ax2.grid(axis='y', color=GRID, linewidth=0.6)
    ax2.xaxis.set_major_formatter(DateFormatter('%d %b'))

    # ── Summary stats text ────────────────────────────────────────────────────
    total_cost         = (df['price'] * df['E_el_kWh']).sum()
    total_energy       = df['E_el_kWh'].sum()
    total_thermal_kWh  = df['Qdot_th_kW'].sum()
    total_gains_kWh    = df['Qdot_gains_kW'].sum() if 'Qdot_gains_kW' in df.columns else 0.0
    total_combined_kWh = total_thermal_kWh + total_gains_kWh
    n_hours            = len(df)
    n_days             = n_hours / 24.0

    viol_low    = df['T_room'] < T_low
    viol_high   = df['T_room'] > T_high
    pct_comfort = 100 * (1 - (viol_low | viol_high).mean())
    pct_under   = 100 * viol_low.mean()
    pct_over    = 100 * viol_high.mean()

    T = df['T_room']
    T_mean      = float(T.mean())
    T_median    = float(T.median())
    T_std       = float(T.std())
    dev_mean    = float((T - 21.0).abs().mean())   # mean absolute deviation from 21°C setpoint
    hours_cold  = int(viol_low.sum())              # hours below 20°C
    hours_hot   = int(viol_high.sum())             # hours above 22°C

    hp_on_int = df['hp_on'].astype(int)
    n_cycles  = int((hp_on_int.diff() == 1).sum())
    if hp_on_int.iloc[0] == 1:
        n_cycles += 1

    stats = (f"Energy: {total_energy:.1f} kWh  |  "
             f"Cost: €{total_cost:.2f}  |  "
             f"HP+gains: {total_combined_kWh:.0f} kWh over {n_days:.0f} days  |  "
             f"Comfort: {pct_comfort:.1f}% (under: {pct_under:.1f}%, over: {pct_over:.1f}%)  |  "
             f"HP cycles: {n_cycles}")
    plt.tight_layout(rect=[0, 0.01, 1, 1])
    plt.subplots_adjust(hspace=0.35)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Plot saved → {save_path}")
    else:
        plt.show()

    return {
        'total_energy_kWh':    total_energy,
        'total_cost_eur':      total_cost,
        'total_thermal_kWh':   total_thermal_kWh,
        'total_gains_kWh':     total_gains_kWh,
        'total_combined_kWh':  total_combined_kWh,
        'n_days':              n_days,
        'T_mean':              round(T_mean, 2),
        'T_median':            round(T_median, 2),
        'T_std':               round(T_std, 3),
        'dev_from_setpoint':   round(dev_mean, 3),
        'hours_too_cold':      hours_cold,
        'hours_too_hot':       hours_hot,
        'pct_comfort':         pct_comfort,
        'pct_under':           pct_under,
        'pct_over':            pct_over,
        'n_cycles':            n_cycles,
    }


def monthly_breakdown(df, T_low, T_high):
    """Return a per-month comfort/energy summary DataFrame."""
    return df.resample('ME').apply(lambda x: pd.Series({
        'comfort%':      round(100 * x['T_room'].between(T_low, T_high).mean(), 1),
        'under%':        round(100 * (x['T_room'] < T_low).mean(), 1),
        'over%':         round(100 * (x['T_room'] > T_high).mean(), 1),
        'T_mean':        round(x['T_room'].mean(), 2),
        'T_median':      round(x['T_room'].median(), 2),
        'T_std':         round(x['T_room'].std(), 2),
        'dev_setpoint':  round((x['T_room'] - 21.0).abs().mean(), 2),
        'h_too_cold':    int((x['T_room'] < T_low).sum()),
        'h_too_hot':     int((x['T_room'] > T_high).sum()),
        'T_max':         round(x['T_room'].max(), 1),
        'T_min':         round(x['T_room'].min(), 1),
        'cycles':        int((pd.Series(x['hp_on'].astype(int)).diff() == 1).sum()),
        'HP_kWh':        round(x['Qdot_th_kW'].sum(), 1),
        'elec_kWh':      round(x['E_el_kWh'].sum(), 1),
        'cost_eur':      round((x['price'] * x['E_el_kWh']).sum(), 2),
    }))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # Data
    data = load_or_generate(args.data, args.days, args.profiles_dir, args.no_gains)

    # Environment — random_init=False for deterministic post-training evaluation.
    # (random_init=True is for training's EvalCallback which benefits from varied
    #  start conditions; here we want a reproducible fixed-start episode.)
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

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n── Results ──────────────────────────────────────────────────────")
    print(f"  Period        : {df.index[0].date()} → {df.index[-1].date()}  "
          f"({len(df)} steps / {len(df)/24:.0f} days)")
    print(f"  T_room range  : {df['T_room'].min():.1f} – {df['T_room'].max():.1f} °C")
    print(f"  T_amb range   : {df['T_amb'].min():.1f} – {df['T_amb'].max():.1f} °C")
    print(f"  T_sup range   : {df['u'].min():.1f} – {df['u'].max():.1f} °C")

    stats = plot_results(df, args.T_comfort_low, args.T_comfort_high,
                         save_path=f'{args.save}.png' if args.save else None)

    print(f"\n── Room temperature ─────────────────────────────────────────────")
    print(f"  Mean          : {stats['T_mean']:.2f} °C  (setpoint: 21.0 °C)")
    print(f"  Median        : {stats['T_median']:.2f} °C")
    print(f"  Std deviation : {stats['T_std']:.2f} °C")
    print(f"  Mean |dev| from 21°C setpoint : {stats['dev_from_setpoint']:.2f} K")
    print(f"  Hours too cold (<{args.T_comfort_low:.0f}°C) : "
          f"{stats['hours_too_cold']:,} h  ({stats['pct_under']:.1f}%)")
    print(f"  Hours too hot  (>{args.T_comfort_high:.0f}°C) : "
          f"{stats['hours_too_hot']:,} h  ({stats['pct_over']:.1f}%)")
    print(f"  Comfort %     : {stats['pct_comfort']:.1f}%")

    print(f"\n── Energy & cost ────────────────────────────────────────────────")
    print(f"  Electricity   : {stats['total_energy_kWh']:.1f} kWh")
    print(f"  Cost          : €{stats['total_cost_eur']:.2f}")
    print(f"  HP delivered  : {stats['total_thermal_kWh']:.1f} kWh")
    print(f"  Gains         : {stats['total_gains_kWh']:.1f} kWh  (solar + internal)")
    print(f"  HP + gains    : {stats['total_combined_kWh']:.1f} kWh")
    print(f"  HP cycles     : {stats['n_cycles']}")

    # ── Monthly breakdown ─────────────────────────────────────────────────────
    monthly = monthly_breakdown(df, args.T_comfort_low, args.T_comfort_high)
    print(f"\n── Monthly breakdown ────────────────────────────────────────────")
    print(monthly.to_string())

    if args.save:
        csv_path = f'{args.save}_monthly.csv'
        monthly.to_csv(csv_path)
        print(f"\nMonthly breakdown saved → {csv_path}")


if __name__ == '__main__':
    main()