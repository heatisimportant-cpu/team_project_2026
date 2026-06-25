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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

from src.room_env import RoomHeatEnv


class SaveVecNormalizeCallback(BaseCallback):
    """Saves VecNormalize stats alongside every periodic checkpoint.

    Without this, CheckpointCallback saves model weights every save_freq
    steps, but the matching normalization stats are only ever saved once,
    at the very end of training -- meaning any checkpoint or best_model
    inspected mid-run has no exact matching vecnormalize.pkl to evaluate
    it correctly with.
    """
    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            vec_normalize_env = self.model.get_vec_normalize_env()
            if vec_normalize_env is not None:
                path = f"{self.save_path}/vecnormalize_{self.num_timesteps}_steps.pkl"
                vec_normalize_env.save(path)
                if self.verbose:
                    print(f"  Saved VecNormalize stats → {path}")
        return True


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
    p.add_argument('--T_room_set_lower', type=float, default=20.0,
                   help='Lower comfort bound [degC].')
    p.add_argument('--T_room_set_upper', type=float, default=22.0,
                   help='Upper comfort bound [degC].')
    p.add_argument('--comfort_weight', type=float, default=50.0,
                   help='Tier 1 (dominant) -- comfort violation penalty weight.')
    p.add_argument('--cost_weight',    type=float, default=1.0,
                   help='Tier 2 -- electricity cost penalty weight.')
    p.add_argument('--cycle_weight',   type=float, default=1.0,
                   help='Tier 3 (smallest) -- HP on/off cycling penalty weight.')
    p.add_argument('--comfort_penalty_cap', type=float, default=100.0,
                   help='Max squared comfort deviation [(degC)^2] before weighting. '
                        'Bounds worst-case per-step penalty from pathological '
                        'excursions during early random exploration.')
    p.add_argument('--forecast_steps', type=int,   default=24)

    # Training
    p.add_argument('--timesteps',       type=int, default=100_000,
                   help='When resuming (--resume_from), this is ADDITIONAL steps '
                        'to run beyond the checkpoint, not an absolute total.')
    p.add_argument('--run_name',        type=str, default='sac_hp')
    p.add_argument('--seed',            type=int, default=42)
    p.add_argument('--n_eval_episodes', type=int, default=5)
    p.add_argument('--resume_from',     type=str, default=None,
                   help='Path to a saved model (.zip) to resume from. Preserves '
                        'policy/critic weights and optimizer state. Use a NEW '
                        '--run_name so this does not overwrite the original run.')
    p.add_argument('--resume_buffer',   type=str, default=None,
                   help='Path to a saved replay buffer (.pkl)')

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
    p.add_argument('--ent_coef',        type=str,   default='auto',
                   help="Entropy coefficient (controls exploration vs exploitation)")
    p.add_argument('--target_entropy',  type=str,   default='auto',
                   help="Target entropy for automatic ent_coef tuning")
    p.add_argument('--net_arch',        type=str,   default='256,256',
                   help="Comma-separated hidden layer sizes, shared by actor and "
                        "critic (e.g. '256,256' = SB3 default, '400,300' or "
                        "'256,256,256' for more capacity).")
    p.add_argument('--normalize_obs',   action='store_true', default=True,
                   help="Normalize observations via a running mean/std (VecNormalize)")
    p.add_argument('--no_normalize_obs', dest='normalize_obs', action='store_false',
                   help="Disable observation normalization.")
    p.add_argument('--resume_vecnormalize', type=str, default=None,
                   help="Path to saved VecNormalize stats (.pkl)")

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
            cost_weight=args.cost_weight,
            cycle_weight=args.cycle_weight,
            comfort_penalty_cap=args.comfort_penalty_cap,
            T_room_set_lower=args.T_room_set_lower,
            T_room_set_upper=args.T_room_set_upper,
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

    if args.normalize_obs:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)
        eval_env  = VecNormalize(eval_env,  norm_obs=True, norm_reward=False, training=False)
        if args.resume_vecnormalize:
            train_env = VecNormalize.load(args.resume_vecnormalize, train_env.venv)
            train_env.training = True
            eval_env = VecNormalize.load(args.resume_vecnormalize, eval_env.venv)
            eval_env.training = False
            print(f"  VecNormalize stats restored from {args.resume_vecnormalize}")

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
    if args.normalize_obs:
        callbacks.append(SaveVecNormalizeCallback(
            save_freq=50_000,
            save_path=os.path.join(run_dir, 'checkpoints'),
            verbose=1,
        ))

    # ── SAC model ─────────────────────────────────────────────────────────────
    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")
        model_path = args.resume_from.replace('.zip', '')
        model = SAC.load(model_path, env=train_env, tensorboard_log=log_dir)
        print(f"  Loaded at num_timesteps={model.num_timesteps:,}")
        if args.resume_buffer:
            model.load_replay_buffer(args.resume_buffer)
            print(f"  Replay buffer restored from {args.resume_buffer} "
                  f"({model.replay_buffer.size():,} transitions) -- training resumes immediately.")
        else:
            model.learning_starts = model.num_timesteps + args.learning_starts
            print(f"  No --resume_buffer given -- replay buffer starts EMPTY. "
                  f"Policy/critic weights are preserved, but past experience is "
                  f"not. Re-warming buffer for {args.learning_starts:,} steps "
                  f"(until num_timesteps={model.learning_starts:,}) before training resumes.")
    else:
        net_arch = [int(x) for x in args.net_arch.split(',')]
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
            target_entropy  = args.target_entropy,
            policy_kwargs   = dict(net_arch=net_arch),
            verbose         = 1,
            tensorboard_log = log_dir,
            seed            = args.seed,
        )

    print(f"\nTraining SAC — {args.timesteps:,} timesteps")
    print(f"  Episode        : {args.days} days")
    print(f"  Comfort band   : [{args.T_room_set_lower}, {args.T_room_set_upper}] degC")
    print(f"  Comfort weight : {args.comfort_weight}  |  Cost weight: {args.cost_weight}  "
          f"|  Cycle weight: {args.cycle_weight}")
    print(f"  Buffer size    : {args.buffer_size:,}  |  Learning starts: {args.learning_starts:,}")
    print(f"  Ent coef       : {args.ent_coef}  |  Target entropy: {args.target_entropy}")
    print(f"  Net arch       : {args.net_arch}  |  Normalize obs: {args.normalize_obs}")
    print(f"  Eval every     : {eval_freq:,} steps  ({args.n_eval_episodes} episodes)")
    print(f"  Run dir        : {run_dir}")
    print(f"  TensorBoard    : tensorboard --logdir {log_dir}\n")
    if args.resume_from:
        print(f"  Resuming: will run {args.timesteps:,} additional steps "
              f"(target num_timesteps = {model.num_timesteps + args.timesteps:,})\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=(args.resume_from is None),
    )

    final_path = os.path.join(run_dir, 'final_model')
    model.save(final_path)
    buffer_path = os.path.join(run_dir, 'replay_buffer.pkl')
    model.save_replay_buffer(buffer_path)
    print(f"\nDone.  Best model     → {run_dir}/best_model.zip")
    print(f"       Final model    → {final_path}.zip")
    print(f"       Replay buffer  → {buffer_path}  (pass via --resume_buffer to continue later)")
    if args.normalize_obs:
        vecnorm_path = os.path.join(run_dir, 'vecnormalize.pkl')
        train_env.save(vecnorm_path)
        print(f"       VecNormalize   → {vecnorm_path}  (REQUIRED for correct evaluation -- "
              f"see evaluate.py --vecnormalize)")


if __name__ == '__main__':
    main()