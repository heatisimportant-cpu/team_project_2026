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
    python src/train.py                              # synthetic data
    python src/train.py --data data/train.csv        # real training data
    python src/train.py --data data/train.csv \\
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

    # Environment
    p.add_argument('--days',           type=int,   default=30)
    p.add_argument('--comfort_weight', type=float, default=10.0)
    p.add_argument('--cycle_weight',   type=float, default=1.0)
    p.add_argument('--forecast_steps', type=int,   default=24)

    # Training
    p.add_argument('--timesteps',       type=int, default=100_000)
    p.add_argument('--run_name',        type=str, default='sac_hp')
    p.add_argument('--seed',            type=int, default=42)
    p.add_argument('--n_eval_episodes', type=int, default=5)

    # SAC hyperparameters
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--gamma',           type=float, default=0.99)
    p.add_argument('--buffer_size',     type=int,   default=100_000,
                   help='Replay buffer size.')
    p.add_argument('--learning_starts', type=int,   default=5_000,
                   help='Steps of random exploration before training starts.')
    p.add_argument('--batch_size',      type=int,   default=256)
    p.add_argument('--tau',             type=float, default=0.005,
                   help='Soft update coefficient for target network.')
    p.add_argument('--train_freq',      type=int,   default=1,
                   help='Update the model every n steps.')
    p.add_argument('--gradient_steps',  type=int,   default=1,
                   help='Gradient steps per env step.')

    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(path):
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    df = df[~df.index.duplicated(keep='first')]  
    assert 'T_amb' in df.columns and 'price_eur_kwh' in df.columns, \
        f"{path} must have columns: T_amb, price_eur_kwh"
    print(f"  Loaded {path}: {len(df):,} hours "
          f"({df.index[0].date()} → {df.index[-1].date()})")
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
    train_data = load_data(args.data) if args.data else synthetic_data()
    eval_data  = load_data(args.eval_data) if args.eval_data else train_data
    if args.eval_data is None:
        print("  No --eval_data provided — evaluating on training data. "
              "Pass --eval_data data/test.csv for a proper generalisation test.")

    # ── Environments ──────────────────────────────────────────────────────────
    train_env = DummyVecEnv([make_env(train_data, args, random_init=True)])
    eval_env  = DummyVecEnv([make_env(eval_data,  args, random_init=False)])

    # ── Callbacks ─────────────────────────────────────────────────────────────
    eval_freq = 5_000

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=run_dir,
            log_path=log_dir,
            eval_freq=eval_freq,
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
        verbose         = 1,
        tensorboard_log = log_dir,
        seed            = args.seed,
    )

    print(f"\nTraining SAC — {args.timesteps:,} timesteps")
    print(f"  Episode        : {args.days} days")
    print(f"  Comfort weight : {args.comfort_weight}  |  Cycle weight: {args.cycle_weight}")
    print(f"  Buffer size    : {args.buffer_size:,}  |  Learning starts: {args.learning_starts:,}")
    print(f"  Eval every     : {eval_freq:,} steps  ({args.n_eval_episodes} episodes)")
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