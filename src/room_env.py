# -*- coding: utf-8 -*-
"""
Room Heating Gymnasium Environment
====================================
Wraps the Simulator into a Gymnasium-compatible RL environment.

Observation (58 dims)
---------------------
  [0:3]   Building states   : T_room, T_wall, T_hp_ret
  [3]     T_amb             : current outdoor temperature  [°C]
  [4]     price_eur_kwh     : current electricity price    [€/kWh]
  [5:29]  Price forecast    : price(k+1) … price(k+24)    [€/kWh]
  [29:53] Weather forecast  : T_amb(k+1) … T_amb(k+24)   [°C]
  [53]    H_ve              : ventilation losses           [W/K]  (normalised)
  [54]    H_tr              : envelope transmission        [W/K]  (normalised)
  [55]    c_bldg            : thermal mass                 [Wh/m²K] (normalised)
  [56]    area_floor        : floor area                   [m²]  (normalised)
  [57]    mdot_hp           : HP mass flow rate            [kg/s] (normalised)
  Note: T_amb_lim excluded — constant 20 °C across all building models.

Action (1 dim)
--------------
  Normalized supply temperature in [-1, 1]
  mapped to [T_HP_MIN, T_HP_MAX] = [20, 65] °C

Reward
------
  r = comfort_weight * comfort - price * E_el_kWh - cycle_penalty
  comfort = -(T_room - 21)²   (parabola, peak at 21 °C)

"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional

from models.vonovia_model import Building, vonovia_model
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

# Building parameter bounds 
BLDG_PARAM_LIMITS = {
    'H_ve':       (40.0,  200.0),   
    'H_tr':       (100.0, 1100.0),  
    'c_bldg':     (20.0,   90.0),   
    'area_floor': (80.0,  350.0),   
    'mdot_hp':    (0.18,   0.35),   
}
BLDG_PARAM_KEYS = list(BLDG_PARAM_LIMITS.keys())


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
    randomise_building : bool
        Randomise building parameters on each reset(). If curriculum_steps > 0
        the sampling range widens gradually from vonovia_model ± 0 to the full
        BLDG_PARAM_LIMITS over the first curriculum_steps environment steps.
    curriculum_steps : int
        Number of env steps over which the building sampling range widens from
        the base building to the full BLDG_PARAM_LIMITS. 0 = full range immediately.
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
        randomise_building: bool = False,
        curriculum_steps: int = 0,
        forecast_steps: int = 24,
        comfort_weight: float = 0.0,
        cycle_weight: float = 0.0,
        noise_level: float = 0.0,
        T_room_set_lower: float = 20.0,
        T_room_set_upper: float = 26.0,
    ):
        super().__init__()

        self.delta_t            = delta_t
        self.random_init        = random_init
        self.randomise_building = randomise_building
        self.curriculum_steps   = curriculum_steps
        self.forecast_steps     = forecast_steps
        self.comfort_weight     = comfort_weight
        self.cycle_weight       = cycle_weight
        self.noise_level        = noise_level
        self.T_room_set_lower   = T_room_set_lower
        self.T_room_set_upper   = T_room_set_upper
        self._default_mdot_HP   = mdot_HP
        self._total_steps       = 0   # global env step counter for curriculum

        # ── Disturbance profile ───────────────────────────────────────────────
        self.p = disturbances[['T_amb', 'price_eur_kwh']].copy()
        self.p['Qdot_gains'] = (
            disturbances['Qdot_gains']
            if 'Qdot_gains' in disturbances.columns
            else 0.0
        )
        self.p = self.p.resample(f'{delta_t}s').ffill().astype(np.float32)

        # ── Models & simulator ────────────────────────────────────────────────
        # Initialise with vonovia_model as the default; reset() will resample
        # if randomise_building=True.
        self.hp_model = iDM_AERO_ALM_4_12()
        self._init_building(vonovia_model, mdot_HP)

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

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)

        # Randomise building parameters each episode if requested 
        if self.randomise_building:
            sampled_params = dict(vonovia_model)   # start from a full valid base
            if self.curriculum_steps > 0:
                progress = float(np.clip(self._total_steps / self.curriculum_steps, 0.0, 1.0))
            else:
                progress = 1.0
            for key, (lo, hi) in BLDG_PARAM_LIMITS.items():
                base_val = float(vonovia_model.get(key, (lo + hi) / 2.0))
                cur_lo   = base_val - progress * (base_val - lo)
                cur_hi   = base_val + progress * (hi - base_val)
                sampled_params[key] = float(self.np_random.uniform(cur_lo, cur_hi))
            mdot = sampled_params.pop('mdot_hp')
            self._init_building(sampled_params, mdot)

        if self.random_init:
            max_start = len(self.p) - self.max_steps - self.forecast_steps - 1
            self.t = int(self.np_random.integers(0, max(1, max_start)))
        else:
            self.t = 0

        self._cur_steps  = 0
        self.prev_action = None
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

        # Simulate one step
        result     = self.simulator.get_next_state(state_dict, T_hp_sup, pk)
        next_state = result['state']
        costs      = result['cost']

        self.t          += 1
        self._cur_steps += 1
        self._total_steps += 1
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

    # ── Building construction helper ──────────────────────────────────────────

    def _init_building(self, params: dict, mdot_hp: float):
        """Instantiate Building and Simulator from a parameter dict.

        Called once at startup and on every reset() when randomise_building=True.
        """
        self.bldg_model = Building(
            params=params,
            mdot_hp=mdot_hp,
            T_room_set_lower=self.T_room_set_lower,
            T_room_set_upper=self.T_room_set_upper,
        )
        self.simulator = Simulator(
            hp_model=self.hp_model,
            bldg_model=self.bldg_model,
            timestep=self.delta_t,
        )

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
        # Building parameters (normalised to [0, 1])
        low  += [0.0] * len(BLDG_PARAM_KEYS)
        high += [1.0] * len(BLDG_PARAM_KEYS)
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
        # Building parameters — normalised to [0, 1]
        p = self.bldg_model.params
        for key in BLDG_PARAM_KEYS:
            lo, hi = BLDG_PARAM_LIMITS[key]
            val = float(p.get(key, lo))
            obs.append(float(np.clip((val - lo) / (hi - lo), 0.0, 1.0)))
        return np.array(obs, dtype=np.float32)

    def _reward(self, E_el_kWh: float, price: float, costs: Dict, cycle: bool) -> float:
        """
        r = comfort_weight * comfort - price * E_el_kWh - cycle_penalty

        comfort = -(T_room - 21)²   (parabola, peak 0 at 21 °C)

        electricity_cost = price [€/kWh] * E_el_kWh

        cycle_penalty = cycle_weight  if HP switched on↔off this step
        """
        T_room = costs.get('T_room_last', 21.0)
        scale  = 1 / 1000

        # Comfort term: parabola with peak at 21°C
        comfort = -((T_room - 21.0) ** 2)

        # Electricity cost penalty
        cost_penalty = price * E_el_kWh

        # Cycle penalty
        cycle_penalty = self.cycle_weight if cycle else 0.0

        return scale * float(self.comfort_weight * comfort - cost_penalty - cycle_penalty)

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