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
  [53:57] Cyclical time     : sin(hour), cos(hour), sin(day-of-year), cos(day-of-year)
  [57]    Heating cutoff    : 1.0 if T_amb >= T_amb_lim this step, else 0.0

Action (1 dim)
--------------
  Normalized supply temperature in [-1, 1]
  mapped to [T_HP_MIN, T_HP_MAX] = [20, 65] °C

Disturbances input
-------------------
  T_amb, price_eur_kwh   : required
  Qdot_solar             : optional, defaults to 0 if absent [W]
  Qdot_internal          : optional, defaults to 0 if absent [W]

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


class RoomHeatEnv(gym.Env):
    """Gymnasium environment for heat pump space heating control.

    Parameters
    ----------
    disturbances : pd.DataFrame
        DatetimeIndex, columns: T_amb [°C], price_eur_kwh [€/kWh].
        Optional columns: Qdot_solar [W], Qdot_internal [W] -- each
        defaults to 0 if absent. Kept separate rather than pre-summed,
        so downstream benchmarking (e.g. heating-curve + PID baseline)
        can use each disturbance independently.
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
        Penalty weight for comfort violations in reward (tier 1 -- dominant).
    cost_weight : float
        Penalty weight for electricity cost in reward (tier 2).
    cycle_weight : float
        Penalty weight for HP on/off cycling in reward (tier 3 -- smallest).
    comfort_penalty_cap : float
        Maximum value of the squared comfort deviation, in (°C)^2, before
        weighting. Bounds the worst-case single-step penalty so pathological
        excursions (mainly seen during early random exploration, before the
        policy has learned anything) don't create extreme outlier Q-value
        targets that are disproportionately hard for the critic to fit.
        Default 100.0 = a 10°C deviation; deviations beyond that are
        already a total loss of control and don't need to be penalized
        even harder to convey that.
    comfort_inband_weight : float
        Small weight on a gentle ((T_room-center)/half_width)^2 gradient
        INSIDE the comfort band. A pure zero-inside-band reward gives no
        learning signal once the policy is "good enough", which can cause
        training to plateau with no incentive to refine further. Default
        0.005 -- validated numerically so the worst in-band penalty (right
        at either edge) stays below even a 0.1degC violation just outside
        the band (the first value tried, 0.1, was NOT small enough and
        actually made sitting at the edge worse than a tiny violation
        outside it -- caught by testing rather than assumed).
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
        cost_weight: float = 1.0,
        cycle_weight: float = 0.0,
        comfort_penalty_cap: float = 100.0,
        comfort_inband_weight: float = 0.005,
        noise_level: float = 0.0,
        T_room_set_lower: float = 20.0,
        T_room_set_upper: float = 22.0,
    ):
        super().__init__()

        self.delta_t               = delta_t
        self.random_init           = random_init
        self.forecast_steps        = forecast_steps
        self.comfort_weight        = comfort_weight
        self.cost_weight           = cost_weight
        self.cycle_weight          = cycle_weight
        self.comfort_penalty_cap   = comfort_penalty_cap
        self.comfort_inband_weight = comfort_inband_weight
        self.noise_level           = noise_level
        self.T_room_set_lower      = T_room_set_lower
        self.T_room_set_upper      = T_room_set_upper

        # ── Disturbance profile ───────────────────────────────────────────────
        self.p = disturbances[['T_amb', 'price_eur_kwh']].copy()
        for col in ('Qdot_solar', 'Qdot_internal'):
            self.p[col] = (
                disturbances[col] if col in disturbances.columns else 0.0
            )
        self.p = self.p.resample(f'{delta_t}s').ffill().astype(np.float32)

        # ── Models & simulator ────────────────────────────────────────────────
        self.bldg_model = Building(
            params=vonovia_model,
            mdot_hp=mdot_HP,
            T_room_set_lower=T_room_set_lower,
            T_room_set_upper=T_room_set_upper,
        )
        self.hp_model  = iDM_AERO_ALM_4_12()
        self.simulator = Simulator(
            hp_model=self.hp_model,
            bldg_model=self.bldg_model,
            timestep=delta_t,
        )

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
        pk['Qdot_gains'] = pk['Qdot_solar'] + pk['Qdot_internal']

        # Denormalise action → physical supply temperature
        T_hp_sup = self._denorm_action(float(action[0]))

        # Heating cutoff: no heating when T_amb ≥ T_amb_lim
        heating_cutoff_active = pk['T_amb'] >= self.bldg_model.params['T_amb_lim']
        if not heating_cutoff_active:
            T_hp_sup = max(
                T_hp_sup + self.bldg_model.params['T_offset'],
                state_dict['T_hp_ret'],
            )
            T_hp_sup = float(np.clip(T_hp_sup, self.hp_model.T_flow_min, self.hp_model.T_flow_max))
            T_hp_sup = max(T_hp_sup, state_dict['T_hp_ret'])
        else:
            T_hp_sup = state_dict['T_hp_ret']
        self.prev_action = T_hp_sup

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
        reward    = self._reward(E_el_kWh, pk['price_eur_kwh'], costs, cycle,
                                  heating_cutoff_active)

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
            'Qdot_solar':    float(pk['Qdot_solar']),
            'Qdot_internal': float(pk['Qdot_internal']),
            'Qdot_gains':    float(pk['Qdot_gains']),
            'u':           float(T_hp_sup),
            'hp_on':       bool(hp_is_on),
            'heating_cutoff_active': bool(heating_cutoff_active),
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
        # Cyclical time features: sin/cos are always bounded in [-1, 1]
        low  += [-1.0] * 4
        high += [1.0] * 4
        # Heating cutoff flag: 1.0 if T_amb >= T_amb_lim this step, else 0.0
        low  += [0.0]
        high += [1.0]
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
        obs.extend(self._time_features(self.t))
        obs.append(1.0 if pk['T_amb'] >= self.bldg_model.params['T_amb_lim'] else 0.0)
        return np.array(obs, dtype=np.float32)

    def _time_features(self, t: int):
        """Cyclical encoding of hour-of-day and day-of-year.

        Requires no sensor or forecast -- the calendar is known with
        certainty in both simulation and deployment
        """
        ts = self.p.index[min(t, len(self.p) - 1)]
        hour_frac = ts.hour + ts.minute / 60.0
        doy       = ts.dayofyear
        days_in_year = 366.0 if ts.is_leap_year else 365.0
        return (
            np.sin(2 * np.pi * hour_frac / 24.0),
            np.cos(2 * np.pi * hour_frac / 24.0),
            np.sin(2 * np.pi * doy / days_in_year),
            np.cos(2 * np.pi * doy / days_in_year),
        )

    def _reward(self, E_el_kWh: float, price: float, costs: Dict, cycle: bool,
                heating_cutoff_active: bool) -> float:
        """
        Three-tier priority reward: comfort >> cost >> cycling.
        """
        T_room = costs.get('T_room_last', 21.0)   # fallback; set below in step()
        T_low, T_high = self.T_room_set_lower, self.T_room_set_upper
        T_center = (T_low + T_high) / 2.0
        half_width = (T_high - T_low) / 2.0
        scale = 1 / 500

        if heating_cutoff_active:
            comfort_penalty = 0.0
        elif T_room < T_low:
            comfort_penalty = min((T_low - T_room) ** 2, self.comfort_penalty_cap)
        elif T_room > T_high:
            comfort_penalty = min((T_room - T_high) ** 2, self.comfort_penalty_cap)
        else:
            comfort_penalty = self.comfort_inband_weight * ((T_room - T_center) / half_width) ** 2

        cost_penalty  = price * E_el_kWh
        cycle_penalty = 1.0 if cycle else 0.0

        total = (self.comfort_weight * comfort_penalty
                 + self.cost_weight   * cost_penalty
                 + self.cycle_weight  * cycle_penalty)
        return -scale * float(total)

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