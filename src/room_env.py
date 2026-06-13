# -*- coding: utf-8 -*-
"""
Room Heating Gymnasium Environment
====================================
Wraps the Simulator into a Gymnasium-compatible RL environment.

Observation (69 dims)
---------------------
  [0:3]   Building states   : T_room, T_wall, T_hp_ret
  [3]     T_amb             : current outdoor temperature  [°C]
  [4]     price_eur_kwh     : current electricity price    [€/kWh]
  [5:29]  Price forecast    : price(k+1) … price(k+24)    [€/kWh]
  [29:53] Weather forecast  : T_amb(k+1) … T_amb(k+24)   [°C]
  [53:55] Hour encoding     : sin(hour), cos(hour)        [-1, 1]
  [55:57] Day-of-week enc.  : sin(dow),  cos(dow)         [-1, 1]
  [57]    Previous action   : last normalised action       [-1, 1]
  [58:69] Building ID       : one-hot encoded bldg index   [0, 1]

Action (1 dim)
--------------
  Normalized supply temperature in [-1, 1]
  mapped to [T_HP_MIN, T_HP_MAX] = [20, 65] °C

"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional

from models.vonovia_model import Building, BUILDING_MODELS
from models.heatpump_model import iDM_AERO_ALM_4_12
from src.simulator import Simulator

# ── Observation space bounds ──────────────────────────────────────────────────
OBS_LIMITS = {
    'T_room':        (5.0,  60.0),
    'T_wall':        (5.0,  60.0),
    'T_hp_ret':      (5.0,  65.0),
    'T_amb':         (-25.0, 45.0),
    'price_eur_kwh': (-0.5,   2.0),
}

T_HP_MIN = 20.0   # °C  — HP minimum supply temperature
T_HP_MAX = 65.0   # °C  — HP maximum supply temperature


class RoomHeatEnv(gym.Env):
    """Gymnasium environment for heat pump space heating control.

    Parameters
    ----------
    disturbances : pd.DataFrame
        DatetimeIndex, columns: T_amb [°C], price_eur_kwh [€/kWh].
        Optional column: Qdot_gains [W] — defaults to 0 if absent.
    mdot_HP : float
        HP mass flow rate [kg/s].
    delta_t : int
        Timestep [s]. Default 3600 (1 hour).
    days : int, optional
        Episode length in days. None = full dataset length.
    random_init : bool
        Randomise start index and initial state on each reset().
    forecast_steps : int
        Hours ahead for price + weather forecast in observation.
    comfort_weight : float
        Penalty weight for comfort violations in reward.
    noise_level : float
        Std-dev of Gaussian noise added to observations.
    T_room_set_lower : float
        Lower comfort bound [°C].
    T_room_set_upper : float
        Upper comfort bound [°C].
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        disturbances: pd.DataFrame,
        mdot_HP: float = 0.27,
        delta_t: int = 3600,
        days: Optional[int] = None,
        random_init: bool = False,
        forecast_steps: int = 24,
        comfort_weight: float = 0.0,
        cycle_weight: float = 0.0,
        noise_level: float = 0.0,
        T_room_set_lower: float = 20.0,
        T_room_set_upper: float = 22.0,
        fixed_bldg_idx: int = None,
    ):
        super().__init__()

        self.delta_t        = delta_t
        self.random_init    = random_init
        self.forecast_steps = forecast_steps
        self.comfort_weight = comfort_weight
        self.cycle_weight   = cycle_weight
        self.noise_level    = noise_level
        self.fixed_bldg_idx = fixed_bldg_idx

        self.mdot_HP = mdot_HP
        self.T_room_set_lower = T_room_set_lower
        self.T_room_set_upper = T_room_set_upper

        # Sanitize building models: only include those that can be physically heated by a 12kW HP
        # Criteria: At -10C ambient (30K delta), heat loss must be <= 12kW. So H_ve + H_tr <= 400 W/K
        self.building_models = [b for b in BUILDING_MODELS if (b['H_ve'] + b['H_tr']) <= 400.0]
        self.current_building_params = None

        self.current_heatpump_class = iDM_AERO_ALM_4_12

        # ── Disturbance profile ───────────────────────────────────────────────
        self.p = disturbances[['T_amb', 'price_eur_kwh']].copy()
        self.p['Qdot_gains'] = (
            disturbances['Qdot_gains']
            if 'Qdot_gains' in disturbances.columns
            else 0.0
        )
        self.p = self.p.resample(f'{delta_t}s').ffill().astype(np.float32)

        # ── Models & simulator ────────────────────────────────────────────────
        self._select_models()

        # ── Episode length ────────────────────────────────────────────────────
        if days is not None:
            self.max_steps = days * 24 * int(3600 / delta_t)
            if self.max_steps + forecast_steps >= len(self.p):
                raise ValueError(
                    f"Requested {days} days ({self.max_steps} steps) + "
                    f"{forecast_steps} forecast steps exceeds data length {len(self.p)}"
                )
        else:
            self.max_steps = len(self.p) - forecast_steps - 1

        # ── Spaces ────────────────────────────────────────────────────────────
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = self._build_obs_space()

        # ── Episode state ─────────────────────────────────────────────────────
        self.t           = 0
        self._cur_steps  = 0
        self.state       = None
        self.prev_action = None
        self._hp_was_on  = False   # tracks HP on/off state for cycle detection
        self.reset()
    
    def _select_models(self):
        """Randomly select one building model and one heat pump model."""
        # Select building
        if self.fixed_bldg_idx is not None:
            bldg_idx = self.fixed_bldg_idx
        else:
            bldg_idx = int(self.np_random.integers(0, len(self.building_models)))
        self.current_bldg_idx = bldg_idx
        self.current_building_params = self.building_models[bldg_idx]

        self.bldg_model = Building(
            params=self.current_building_params,
            mdot_hp=self.current_building_params.get('mdot_hp', self.mdot_HP),
            T_room_set_lower=self.T_room_set_lower,
            T_room_set_upper=self.T_room_set_upper,
        )

        # Always use iDM AERO ALM 4-12 (single HP model)
        self.hp_model = self.current_heatpump_class()

        # Recreate simulator with the selected building + heat pump
        self.simulator = Simulator(
            hp_model=self.hp_model,
            bldg_model=self.bldg_model,
            timestep=self.delta_t,
        )

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)

        self._select_models()

        if self.random_init:
            max_start = len(self.p) - self.max_steps - self.forecast_steps - 1
            self.t = int(self.np_random.integers(0, max(1, max_start)))
        else:
            self.t = 0

        self._cur_steps  = 0
        self.prev_action = None
        self._prev_norm_action = 0.0   # normalised previous action for obs
        self._hp_was_on  = False
        self.state       = self._build_obs(self._initial_state_dict())
        return self.state.copy(), {}

    def step(self, action: np.ndarray):
        state_dict = {
            key: float(self.state[i])
            for i, key in enumerate(self.bldg_model.state_keys)
        }
        pk = self._get_pk(self.t)

        # Denormalise action → physical supply temperature
        T_hp_sup = self._denorm_action(float(action[0]))

        # Heating cutoff: no heating when T_amb ≥ T_amb_lim
        if pk['T_amb'] < self.bldg_model.params['T_amb_lim']:
            T_hp_sup = max(
                T_hp_sup + self.bldg_model.params['T_offset'],
                state_dict['T_hp_ret'],
            )
        else:
            T_hp_sup = state_dict['T_hp_ret']

        # Clip to HP operating limits
        T_hp_sup = float(np.clip(T_hp_sup, self.hp_model.T_flow_min, self.hp_model.T_flow_max))
        T_hp_sup = max(T_hp_sup, state_dict['T_hp_ret'])
        self.prev_action = T_hp_sup
        self._prev_norm_action = float(action[0])   # store normalised action

        # Simulate one step
        result     = self.simulator.get_next_state(state_dict, T_hp_sup, pk)
        next_state = result['state']
        costs      = result['cost']

        self.t          += 1
        self._cur_steps += 1
        self.state       = self._build_obs(next_state)

        E_el_kWh  = costs['E_el'] / 1000.0
        hp_is_on  = costs['hp_on']   # from simulator's modulation-floor logic
        cycle     = (hp_is_on != self._hp_was_on)   # state changed → cycle event
        self._hp_was_on = hp_is_on
        costs['T_room_last'] = next_state['T_room']  # pass T_room to reward fn
        reward    = self._reward(E_el_kWh, pk['price_eur_kwh'], costs, cycle)

        terminated = False
        truncated  = (
            self.t >= len(self.p) - self.forecast_steps - 1
            or self._cur_steps >= self.max_steps
        )

        info = {
            'cost':        float(costs['dev_neg_max']),  # SafeRL convention
            'E_el_kWh':    float(E_el_kWh),
            'price':       float(pk['price_eur_kwh']),
            'energy_cost': float(pk['price_eur_kwh'] * E_el_kWh),
            'dev_neg_max': float(costs['dev_neg_max']),
            'dev_neg_sum': float(costs['dev_neg_sum']),
            'T_room':      float(next_state['T_room']),
            'T_amb':       float(pk['T_amb']),
            'u':           float(T_hp_sup),
            'hp_on':       bool(hp_is_on),
            't':           int(self.t),
        }

        obs = self.state.copy()
        if self.noise_level > 0:
            obs += self.np_random.normal(0, self.noise_level, obs.shape).astype(np.float32)
            obs  = obs.clip(self.observation_space.low, self.observation_space.high)

        return obs, float(reward), terminated, truncated, info

    # internal 

    def _build_obs_space(self) -> spaces.Box:
        low, high = [], []
        for key in self.bldg_model.state_keys:
            lo, hi = OBS_LIMITS[key]
            low.append(lo); high.append(hi)
        for key in ('T_amb', 'price_eur_kwh'):
            lo, hi = OBS_LIMITS[key]
            low.append(lo); high.append(hi)
        lo, hi = OBS_LIMITS['price_eur_kwh']
        low  += [lo] * self.forecast_steps
        high += [hi] * self.forecast_steps
        lo, hi = OBS_LIMITS['T_amb']
        low  += [lo] * self.forecast_steps
        high += [hi] * self.forecast_steps
        # Time features: sin/cos hour, sin/cos day-of-week (range [-1, 1])
        low  += [-1.0] * 4
        high += [ 1.0] * 4
        # Previous action (normalised, range [-1, 1])
        low  += [-1.0]
        high += [ 1.0]
        
        # Building ID (one-hot encoded, length = len(building_models))
        num_bldgs = len(self.building_models)
        low  += [0.0] * num_bldgs
        high += [1.0] * num_bldgs
        
        return spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

    def _build_obs(self, state_dict: Dict) -> np.ndarray:
        obs = [state_dict[k] for k in self.bldg_model.state_keys]
        pk  = self._get_pk(self.t)
        obs.append(pk['T_amb'])
        obs.append(pk['price_eur_kwh'])
        for i in range(1, self.forecast_steps + 1):
            obs.append(float(self.p.iloc[min(self.t + i, len(self.p) - 1)]['price_eur_kwh']))
        for i in range(1, self.forecast_steps + 1):
            obs.append(float(self.p.iloc[min(self.t + i, len(self.p) - 1)]['T_amb']))
        # Time features
        ts   = self.p.index[min(self.t, len(self.p) - 1)]
        hour = ts.hour + ts.minute / 60.0
        dow  = ts.dayofweek                        # 0=Mon … 6=Sun
        obs.append(float(np.sin(2 * np.pi * hour / 24.0)))
        obs.append(float(np.cos(2 * np.pi * hour / 24.0)))
        obs.append(float(np.sin(2 * np.pi * dow  /  7.0)))
        obs.append(float(np.cos(2 * np.pi * dow  /  7.0)))
        # Previous action (normalised)
        obs.append(float(self._prev_norm_action))
        
        # Building ID (one-hot)
        bldg_one_hot = [0.0] * len(self.building_models)
        if hasattr(self, 'current_bldg_idx'):
            bldg_one_hot[self.current_bldg_idx] = 1.0
        obs.extend(bldg_one_hot)
        
        return np.array(obs, dtype=np.float32)

    def _reward(self, E_el_kWh: float, price: float, costs: Dict, cycle: bool) -> float:
        """
        r = comfort_penalty - electricity_cost - cycle_penalty

        comfort_penalty:
            0.0                              if T_room in [T_lower, T_upper]
            -5.0 * (T_lower - T_room)²       if T_room < T_lower  (underheating, harsh)
            -3.0 * (T_room  - T_upper)²      if T_room > T_upper  (overheating, strong)

        electricity_cost = price [€/kWh] * E_el_kWh

        cycle_penalty = cycle_weight  if HP switched on↔off this step
        """
        T_room = costs.get('T_room_last', 21.0)

        # Comfort: asymmetric penalty (under 5×, over 3×)
        if T_room < self.T_room_set_lower:
            comfort = -5.0 * (self.T_room_set_lower - T_room) ** 2
        elif T_room > self.T_room_set_upper:
            comfort = -3.0 * (T_room - self.T_room_set_upper) ** 2
        else:
            comfort = 0.0

        # Electricity cost penalty
        cost_penalty = price * E_el_kWh

        # Cycle penalty
        cycle_penalty = self.cycle_weight * (1.0 if cycle else 0.0)

        return float(comfort - cost_penalty - cycle_penalty)

    def _get_pk(self, t: int) -> Dict:
        return self.p.iloc[min(t, len(self.p) - 1)].to_dict()

    def _denorm_action(self, a: float) -> float:
        return float(a * (T_HP_MAX - T_HP_MIN) / 2.0 + (T_HP_MIN + T_HP_MAX) / 2.0)

    def norm_action(self, T_sup: float) -> float:
        """Physical supply temperature [°C] → normalised action [-1, 1]."""
        return float((T_sup - (T_HP_MIN + T_HP_MAX) / 2.0) * 2.0 / (T_HP_MAX - T_HP_MIN))

    def _initial_state_dict(self) -> Dict:
        if self.random_init:
            return {k: float(self.np_random.uniform(*OBS_LIMITS[k])) for k in self.bldg_model.state_keys}
        return {k: 20.0 for k in self.bldg_model.state_keys}

    def get_obs(self) -> np.ndarray:
        return self.state.copy()

    def get_cur_T_amb(self) -> float:
        return float(self.p.iloc[self.t]['T_amb'])

    def get_cur_price(self) -> float:
        return float(self.p.iloc[self.t]['price_eur_kwh'])

    def get_cur_pk(self) -> Dict:
        return self._get_pk(self.t)

    def get_cur_time(self):
        return self.p.index[self.t]