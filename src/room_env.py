# -*- coding: utf-8 -*-
"""
Room Heating Gymnasium Environment
====================================
Wraps the Simulator into a Gymnasium-compatible RL environment.

Observation (66 dims)
---------------------
  [0:3]   Building states   : T_room, T_wall, T_hp_ret
  [3]     T_amb             : current outdoor temperature  [°C]
  [4]     price_eur_kwh     : current electricity price    [€/kWh]
  [5:29]  Price forecast    : price(k+1) … price(k+24)    [€/kWh]
  [29:53] Weather forecast  : T_amb(k+1) … T_amb(k+24)   [°C]
  [53]    sin(hour-of-day)  : current step's time-of-day, cyclical
  [54]    cos(hour-of-day)
  [55]    sin(day-of-year)  : current step's time-of-year, cyclical
  [56]    cos(day-of-year)
  [57]    is_weekend        : 1.0 Sat/Sun, 0.0 Mon-Fri
  [58]    Q_sol_W(t)        : current solar gain through windows [W]
  [59:65] Q_sol_W(t+1…t+6) : 6-hour solar gain forecast [W]
  [65]    hp_was_on         : 1.0 if HP was on last step, 0.0 otherwise

  hp_was_on gives the agent explicit knowledge of the compressor's
  previous state without needing to infer it from T_hp_ret dynamics.
  This helps it learn smoother control (e.g. avoid immediately restarting
  the HP after it just turned off) without relying solely on the cycle
  penalty in the reward. The feature is the same as self._hp_was_on
  which is already tracked for cycle detection -- it costs nothing to
  expose it.

  Solar gains (Q_sol_W) are exposed because cloud cover varies day to
  day within a season -- the time features give the agent the average
  seasonal/diurnal pattern but not whether today is clear or overcast.
  A 6-hour solar forecast gives the agent the window to pre-emptively
  reduce heating before a solar peak arrives, which is the primary
  cause of heating-period overheating (room hitting 30°C+ on sunny
  March/October days when the HP was still running in the morning).
  Normalization bounds for Q_sol_W: [0, 8000] W → [-1, 1].
  (Bound lowered from 15,000 W to 8,000 W to match real DWD-measured
   solar data: max Q_sol_W in Braunschweig station 2021-2026 = 5,947 W;
   8,000 W gives physical headroom while using ~75% of the [-1,1] range
   vs only ~40% at the old 15,000 W ceiling.)

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
        Weight on the comfort penalty term in the reward (tier 1).
    price_weight : float
        Weight on the electricity cost term in the reward (tier 2).
    cycle_weight : float
        Weight on the compressor start-cycling penalty in the reward (tier 3).
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
        comfort_weight: float = 50.0,
        price_weight: float = 20.0,
        cycle_weight: float = 25.0,
        noise_level: float = 0.0,
        T_room_set_lower: float = 20.0,
        T_room_set_upper: float = 22.0,
    ):
        super().__init__()

        self.delta_t        = delta_t
        self.random_init    = random_init
        self.forecast_steps = forecast_steps
        self.comfort_weight = comfort_weight
        self.price_weight   = price_weight
        self.cycle_weight   = cycle_weight
        self.noise_level    = noise_level

        # ── Disturbance profile ───────────────────────────────────────────────
        self.p = disturbances[['T_amb', 'price_eur_kwh']].copy()
        self.p['Qdot_gains'] = (
            disturbances['Qdot_gains']
            if 'Qdot_gains' in disturbances.columns
            else 0.0
        )
        self.p['Q_sol_W'] = (
            disturbances['Q_sol_W']
            if 'Q_sol_W' in disturbances.columns
            else 0.0
        )
        # No resample here: env accesses rows by integer position (iloc),
        # not by timestamp, so gaps between concatenated heating periods
        # are handled correctly -- rows simply follow each other in order.
        # Resampling non-contiguous periods (e.g. Oct-Apr, Oct-Apr) with
        # ffill would fill the summer gap with stale April values, creating
        # months of garbage data in self.p.
        self.p = self.p.astype(np.float32)

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
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Observations are normalized to [-1, 1] using the known physical
        # bounds in OBS_LIMITS (fixed min-max scaling, not running stats).
        # The raw, physically-scaled temperatures/prices the policy would
        # otherwise see span wildly different magnitudes (e.g. T_room in
        # [5, 60] vs price in [-0.5, 2.0]), which makes gradients/losses
        # noisy since a few state dimensions dominate the input scale.
        # A fixed-bound scaler (vs e.g. VecNormalize's running mean/std) is
        # accurate from step one (no warm-up period skewed by the random
        # learning_starts exploration phase) and needs no separate stats
        # file kept in sync between training and evaluation.
        self._raw_low, self._raw_high = self._build_obs_bounds()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=self._raw_low.shape, dtype=np.float32)

        # ── Episode state ─────────────────────────────────────────────────────
        self.t           = 0
        self._cur_steps  = 0
        self.state       = None   # normalized [-1,1] observation returned to the agent
        self._phys_state = None   # raw physical building state (T_room, T_wall, T_hp_ret)
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
        self._phys_state = self._initial_state_dict()
        raw_obs          = self._build_obs(self._phys_state)
        self.state       = self._normalize(raw_obs)
        return self.state.copy(), {}

    def step(self, action: np.ndarray):
        state_dict = dict(self._phys_state)
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
        self._phys_state = next_state
        raw_obs           = self._build_obs(next_state)
        self.state        = self._normalize(raw_obs)

        E_el_kWh  = costs['E_el'] / 1000.0
        hp_is_on  = costs['hp_on']   # from simulator's modulation-floor logic
        # A "cycle" that wears the compressor is a START (off->on), not
        # every on<->off transition. Penalizing both the start AND the stop
        # of the same on-period double-counts a single real cycle, and
        # doesn't match evaluate.py's n_cycles metric (which also only
        # counts off->on rising edges). Count rising edges only here too,
        # so the reward signal and the reported metric agree.
        cycle_start     = bool(hp_is_on and not self._hp_was_on)
        self._hp_was_on = hp_is_on
        costs['T_room_last'] = next_state['T_room']  # pass T_room to reward fn
        reward    = self._reward(E_el_kWh, pk['price_eur_kwh'], costs, cycle_start, pk['T_amb'], hp_is_on)

        terminated = False
        truncated  = (
            self.t >= len(self.p) - self.forecast_steps - 1
            or self._cur_steps >= self.max_steps
        )

        info = {
            'cost':        float(costs['dev_neg_max']),  # SafeRL convention
            'E_el_kWh':    float(E_el_kWh),
            'P_el_kW':     float(costs['P_el'] / 1000.0),  # instantaneous electrical power
            'Qdot_th_kW':  float(costs['Qdot_th'] / 1000.0),  # instantaneous thermal power into building
            'Qdot_gains_kW': float(pk['Qdot_gains']) / 1000.0,  # solar+internal gains this step
            'price':       float(pk['price_eur_kwh']),
            'energy_cost': float(pk['price_eur_kwh'] * E_el_kWh),
            'dev_neg_max': float(costs['dev_neg_max']),
            'dev_neg_sum': float(costs['dev_neg_sum']),
            'T_room':      float(next_state['T_room']),
            'T_amb':       float(pk['T_amb']),
            'u':           float(T_hp_sup),
            'hp_on':       bool(hp_is_on),
            'hp_was_on':   bool(self._hp_was_on),   # state AFTER update — hp_on from this step
            'cycle_start': bool(cycle_start),
            't':           int(self.t),
        }

        obs = self.state.copy()
        if self.noise_level > 0:
            obs += self.np_random.normal(0, self.noise_level, obs.shape).astype(np.float32)
            obs  = obs.clip(self.observation_space.low, self.observation_space.high)

        return obs, float(reward), terminated, truncated, info

    # internal 

    def _build_obs_bounds(self):
        """Raw physical low/high bounds for each observation dimension,
        in the same order _build_obs() constructs them. Used only to
        normalize observations to [-1, 1]; the agent never sees these
        raw values directly."""
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
        # Cyclical time features: sin/cos are exactly bounded in [-1,1]
        # already (so normalization is an identity pass-through); is_weekend
        # is a 0/1 flag, scaled to [-1,1] like everything else for consistency.
        low  += [-1.0, -1.0, -1.0, -1.0, 0.0]
        high += [ 1.0,  1.0,  1.0,  1.0, 1.0]
        # Solar gain: current + 6-hour forecast, bounded [0, 8000] W.
        # Lowered from 15,000 W: real DWD-measured data (Braunschweig station
        # 2021-2026) peaks at 5,947 W; 8,000 W gives physical headroom while
        # using ~75% of the [-1,1] range (vs only ~40% at 15,000 W).
        Q_SOL_MAX = 8_000.0
        low  += [0.0]       * 7   # current + 6 forecast steps
        high += [Q_SOL_MAX] * 7
        # HP on/off state from the previous step — binary 0/1 flag.
        low  += [0.0]
        high += [1.0]
        return (np.array(low, dtype=np.float32), np.array(high, dtype=np.float32))

    def _normalize(self, raw_obs: np.ndarray) -> np.ndarray:
        """Fixed min-max scale raw physical observations to [-1, 1]
        using the known bounds in OBS_LIMITS."""
        span = self._raw_high - self._raw_low
        norm = (raw_obs - self._raw_low) / span * 2.0 - 1.0
        return np.clip(norm, -1.0, 1.0).astype(np.float32)

    def _build_obs(self, state_dict: Dict) -> np.ndarray:
        obs = [state_dict[k] for k in self.bldg_model.state_keys]
        pk  = self._get_pk(self.t)
        obs.append(pk['T_amb'])
        obs.append(pk['price_eur_kwh'])
        for i in range(1, self.forecast_steps + 1):
            obs.append(float(self.p.iloc[min(self.t + i, len(self.p) - 1)]['price_eur_kwh']))
        for i in range(1, self.forecast_steps + 1):
            obs.append(float(self.p.iloc[min(self.t + i, len(self.p) - 1)]['T_amb']))

        # Cyclical time features (current step only -- no forecast, see
        # module docstring). Computed from the actual disturbance index
        # timestamp, so they reflect the real calendar (including weekday
        # vs weekend), not just elapsed episode steps.
        ts = self.p.index[min(self.t, len(self.p) - 1)]
        hour_frac = ts.hour + ts.minute / 60.0
        doy_frac  = (ts.dayofyear - 1) + hour_frac / 24.0
        obs.append(float(np.sin(2 * np.pi * hour_frac / 24.0)))
        obs.append(float(np.cos(2 * np.pi * hour_frac / 24.0)))
        obs.append(float(np.sin(2 * np.pi * doy_frac / 365.25)))
        obs.append(float(np.cos(2 * np.pi * doy_frac / 365.25)))
        obs.append(1.0 if ts.dayofweek >= 5 else 0.0)   # Sat=5, Sun=6

        # Solar gain: current value + 6-hour ahead forecast.
        # Clipped to [0, 8000] before normalization handles any rare spikes.
        for i in range(7):   # i=0: current, i=1..6: forecast
            q = float(self.p.iloc[min(self.t + i, len(self.p) - 1)]['Q_sol_W'])
            obs.append(max(0.0, q))

        # HP on/off state from the PREVIOUS step (self._hp_was_on is already
        # tracked for cycle detection -- free to expose in obs at no extra cost).
        # Gives the agent explicit knowledge of whether it just started/stopped
        # the compressor, helping it learn smoother control without needing to
        # infer HP state from T_hp_ret dynamics alone.
        obs.append(1.0 if self._hp_was_on else 0.0)

        return np.array(obs, dtype=np.float32)

    def _reward(self, E_el_kWh: float, price: float, costs: Dict, cycle_start: bool,
                T_amb: float, hp_is_on: bool) -> float:
        """
        3-tier weighted reward:

            r = scale * ( comfort_weight * comfort_term
                          - price_weight   * price_term
                          - cycle_weight   * cycle_term )

        Tier 1 -- comfort_term:
            -(T_room - 21)²   pure parabola, peak (zero penalty) at the
                              21°C setpoint. No flat zone; deviation is
                              penalized continuously.

            EXCEPTION -- uncontrollable overheating gate:
            When the HP is actually off (hp_is_on=False) and T_room is
            above T_room_set_upper, the comfort penalty is zeroed out.
            The agent has no ability to cool the room (no cooling
            capability exists), so overheating while the HP is off is
            physically outside the agent's control -- penalizing it
            would create an unlearnable gradient that actively fights
            the rest of the reward signal.

            This replaces the previous gate that checked T_amb >=
            T_amb_lim, which missed the most common real case: solar
            gains push T_room above 22°C on cool days (T_amb < 15°C)
            while the HP is already off. Measured: all 100% of
            overheated-HP-off steps in a 30-day evaluation window had
            T_amb < 15°C and were therefore incorrectly penalized by
            the old gate.

            Underheating (T_room < T_room_set_lower) is always penalized
            regardless of HP state -- that reflects the agent's prior
            heating choices, not an uncontrollable physics constraint.

        Tier 2 -- price_term:
            price [€/kWh] * E_el_kWh   this step's electricity cost.
            scale was raised (1/100 vs old 1/500) so price has meaningful
            influence: measured at old scale, price contrib was ~23x
            smaller than comfort contrib on average, effectively making
            the agent price-agnostic despite price_weight=20.

        Tier 3 -- cycle_term:
            1.0 if this step is a compressor START (off->on transition),
            else 0.0.
        """
        T_room = costs.get('T_room_last', 21.0)
        scale = 1/100   # raised from 1/500 -- at 1/500 price was ~23x too small

        # Tier 1: comfort
        overheated = T_room > self.bldg_model.T_room_set_upper
        if (not hp_is_on) and overheated:
            comfort_term = 0.0   # HP is off, can't cool -- unlearnable penalty
        else:
            comfort_term = -((T_room - 21.0) ** 2)

        # Tier 2: price
        price_term = price * E_el_kWh

        # Tier 3: cycling
        cycle_term = 1.0 if cycle_start else 0.0

        return scale * float(
            self.comfort_weight * comfort_term
            - self.price_weight * price_term
            - self.cycle_weight * cycle_term
        )

    def _get_pk(self, t: int) -> Dict:
        return self.p.iloc[min(t, len(self.p) - 1)].to_dict()

    def _denorm_action(self, a: float) -> float:
        return float(a * (T_HP_MAX - T_HP_MIN) / 2.0 + (T_HP_MIN + T_HP_MAX) / 2.0)

    def norm_action(self, T_sup: float) -> float:
        """Physical supply temperature [°C] → normalised action [-1, 1]."""
        return float((T_sup - (T_HP_MIN + T_HP_MAX) / 2.0) * 2.0 / (T_HP_MAX - T_HP_MIN))

    def _initial_state_dict(self) -> Dict:
        if self.random_init:
            # Independently sampling T_room/T_wall/T_hp_ret across their full
            # OBS_LIMITS ranges can produce physically incoherent starting
            # points (e.g. T_room=55, T_wall=6, T_hp_ret=64 simultaneously),
            # which creates large, unrealistic transients right after reset
            # that have nothing to do with any real operating scenario.
            # Instead: sample one plausible base room temperature, then
            # derive the wall and HP-return states with realistic offsets
            # relative to it (wall lags and runs slightly cooler in winter;
            # HP return water is at or above room temperature, having just
            # come off a heating cycle or sat idle near room temp).
            T_room = float(self.np_random.uniform(15.0, 24.0))
            T_wall = T_room - float(self.np_random.uniform(0.0, 3.0))
            T_hp_ret = T_room + float(self.np_random.uniform(0.0, 10.0))

            lo, hi = OBS_LIMITS['T_room']
            T_room = float(np.clip(T_room, lo, hi))
            lo, hi = OBS_LIMITS['T_wall']
            T_wall = float(np.clip(T_wall, lo, hi))
            lo, hi = OBS_LIMITS['T_hp_ret']
            T_hp_ret = float(np.clip(T_hp_ret, lo, hi))

            return {'T_room': T_room, 'T_wall': T_wall, 'T_hp_ret': T_hp_ret}
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