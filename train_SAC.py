# -*- coding: utf-8 -*-
"""
Training Script — SAC Agent for Heat Pump Control
==================================================
Uses Stable-Baselines3 SAC to train on RoomHeatEnv.

SAC is preferred over PPO here because:
  - Off-policy: more sample efficient
  - Built-in entropy maximisation: automatic exploration
  - Well-suited to continuous action spaces

Usage
-----
    python train_SAC.py                              # synthetic data
    python train_SAC.py --data data/train.csv        # real training data
    python train_SAC.py --data data/train.csv \\
                        --eval_data data/test.csv \\
                        --timesteps 500000

Output
------
    runs/<run_name>/
        best_model.zip       ← best policy by mean eval reward
        final_model.zip      ← policy at end of training
        checkpoints/         ← snapshots every 50k steps
        logs/                ← TensorBoard logs
"""

import os
import argparse
import numpy as np
import pandas as pd


from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from src.room_env import RoomHeatEnv
from src.gains import build_gains_series, DEFAULT_PROFILES_DIR
from models.vonovia_model import vonovia_model


# ── Argument parser ───────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data
    p.add_argument('--data',       type=str, default=None,
                   help='Training CSV (columns: T_amb, price_eur_kwh). '
                        'Synthetic data used if not provided.')
    p.add_argument('--eval_data',  type=str, default=None,
                   help='Separate evaluation CSV (e.g. test.csv). '
                        'Falls back to training data if not provided.')
    p.add_argument('--profiles_dir', type=str, default=DEFAULT_PROFILES_DIR,
                   help='Folder with solar_*.csv file(s) + internal_gains.csv '
                        '(auto-discovered; see gains.py). Combined into a '
                        'Qdot_gains disturbance each run -- not cached to '
                        'a file, so changes to source profiles always apply.')
    p.add_argument('--no_gains',   action='store_true',
                   help='Disable solar/internal gains (Qdot_gains=0), e.g. '
                        'for quick synthetic-data smoke tests without the '
                        'profiles folder available.')

    # Environment
    p.add_argument('--days',           type=int,   default=212,
                   help='Episode length in days. Default 212 = one full Oct–Apr '
                        'heating period (matches train_heating.csv period length).')
    p.add_argument('--comfort_weight', type=float, default=50.0,
                   help='Tier 1: weight on comfort penalty.')
    p.add_argument('--price_weight',   type=float, default=20.0,
                   help='Tier 2: weight on electricity cost penalty.')
    p.add_argument('--cycle_weight',   type=float, default=50.0,
                   help='Tier 3: weight on compressor start-cycling penalty.')
    p.add_argument('--forecast_steps', type=int,   default=24)

    # Training
    p.add_argument('--timesteps',       type=int, default=100_000)
    p.add_argument('--run_name',        type=str, default='sac_hp')
    p.add_argument('--seed',            type=int, default=42)
    p.add_argument('--n_eval_episodes', type=int, default=5)
    p.add_argument('--eval_freq',       type=int, default=10_000,
                   help='Evaluate every n env steps. Default 10,000 aligns '
                        'with ~2 episodes at 212-day episode length and avoids '
                        'evaluating during the learning_starts random phase.')

    # SAC hyperparameters
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--gamma',           type=float, default=0.99)
    p.add_argument('--buffer_size',     type=int,   default=300_000,
                   help='Replay buffer size.')
    p.add_argument('--learning_starts', type=int,   default=10_000,
                   help='Steps of random exploration before training starts.')
    p.add_argument('--batch_size',      type=int,   default=256)
    p.add_argument('--tau',             type=float, default=0.005,
                   help='Soft update coefficient for target network.')
    p.add_argument('--train_freq',      type=int,   default=1,
                   help='Update the model every n steps.')
    p.add_argument('--gradient_steps',  type=int,   default=1,
                   help='Gradient steps per env step.')
    p.add_argument('--ent_coef',        type=str,   default='auto',
                   help="SAC entropy coefficient ('auto' for automatic tuning, "
                        "or a fixed float value as a string).")

    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(path, profiles_dir=None, no_gains=False):
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    df = df[~df.index.duplicated(keep='first')]  
    assert 'T_amb' in df.columns and 'price_eur_kwh' in df.columns, \
        f"{path} must have columns: T_amb, price_eur_kwh"
    print(f"  Loaded {path}: {len(df):,} hours "
          f"({df.index[0].date()} → {df.index[-1].date()})")

    if no_gains:
        print("  --no_gains set: Qdot_gains=0 (solar/internal gains disabled)")
    else:
        try:
            gains_df = build_gains_series(
                df.index, profiles_dir=profiles_dir,
                area_floor=vonovia_model['area_floor'])
            df['Qdot_gains'] = gains_df['Qdot_gains']
            df['Q_sol_W']    = gains_df['Q_sol_W']
            print(f"  Gains merged from {profiles_dir}: "
                  f"mean {gains_df['Qdot_gains'].mean():.0f} W, "
                  f"max {gains_df['Qdot_gains'].max():.0f} W")
        except FileNotFoundError as e:
            print(f"  WARNING: could not load gains ({e}); falling back to "
                  f"Qdot_gains=0. Pass --no_gains to silence this, or fix "
                  f"--profiles_dir.")
    return df


def synthetic_data():
    print("  No CSV provided — generating synthetic data (1 year, hourly)")
    idx = pd.date_range('2021-01-01', periods=8760, freq='h', tz='Europe/Berlin')
    t   = np.arange(8760)
    return pd.DataFrame({
        'T_amb':         5 - 12*np.cos(2*np.pi*t/8760) - 5*np.cos(2*np.pi*t/24),
        'price_eur_kwh': 0.20 + 0.10*np.sin(2*np.pi*t/24) + 0.05*np.random.randn(8760),
    }, index=idx)


def make_env(disturbances, args, random_init):
    def _init():
        env = RoomHeatEnv(
            disturbances=disturbances,
            days=args.days,
            random_init=random_init,
            forecast_steps=args.forecast_steps,
            comfort_weight=args.comfort_weight,
            price_weight=args.price_weight,
            cycle_weight=args.cycle_weight,
        )
        return Monitor(env)
    return _init


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    np.random.seed(args.seed)

    run_dir = os.path.join('runs', args.run_name)
    log_dir = os.path.join(run_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Data:")
    train_data = load_data(args.data, args.profiles_dir, args.no_gains) if args.data else synthetic_data()
    eval_data  = load_data(args.eval_data, args.profiles_dir, args.no_gains) if args.eval_data else train_data
    if args.eval_data is None:
        print("  No --eval_data provided — evaluating on training data. "
              "Pass --eval_data data/test.csv for a proper generalisation test.")

    # ── Environments ──────────────────────────────────────────────────────────
    train_env = DummyVecEnv([make_env(train_data, args, random_init=True)])
    # random_init=True here too: with random_init=False every one of the
    # n_eval_episodes evaluations was the exact same fixed scenario (start
    # at t=0), so the reported eval reward was really just one episode
    # repeated 5x -- not informative about how the policy generalizes.
    # Seeding the eval env right after creation keeps the sequence of
    # random start points reproducible across training runs that share
    # --seed, so eval curves stay comparable run-to-run while each
    # individual EvalCallback invocation samples varied conditions.
    eval_env = DummyVecEnv([make_env(eval_data, args, random_init=True)])
    eval_env.seed(args.seed)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=run_dir,
            log_path=log_dir,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=50_000,
            save_path=os.path.join(run_dir, 'checkpoints'),
            name_prefix='sac_hp',
            verbose=0,
        ),
    ]

    # ── SAC model ─────────────────────────────────────────────────────────────
    model = SAC(
        policy          = 'MlpPolicy',
        env             = train_env,
        learning_rate   = args.lr,
        gamma           = args.gamma,
        buffer_size     = args.buffer_size,
        learning_starts = args.learning_starts,
        batch_size      = args.batch_size,
        tau             = args.tau,
        train_freq      = args.train_freq,
        gradient_steps  = args.gradient_steps,
        ent_coef        = args.ent_coef,
        verbose         = 1,
        tensorboard_log = log_dir,
        seed            = args.seed,
    )

    print(f"\nTraining SAC — {args.timesteps:,} timesteps")
    print(f"  Episode        : {args.days} days")
    print(f"  Reward weights : comfort={args.comfort_weight}  price={args.price_weight}  cycle={args.cycle_weight}")
    print(f"  Buffer size    : {args.buffer_size:,}  |  Learning starts: {args.learning_starts:,}")
    print(f"  Eval every     : {args.eval_freq:,} steps  ({args.n_eval_episodes} episodes)")
    print(f"  Run dir        : {run_dir}")
    print(f"  TensorBoard    : tensorboard --logdir {log_dir}\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    final_path = os.path.join(run_dir, 'final_model')
    model.save(final_path)
    print(f"\nDone.  Best model  → {run_dir}/best_model.zip")
    print(f"       Final model → {final_path}.zip")


if __name__ == '__main__':
    main()