import numpy as np

import sys
sys.path.append("..\\..")

from src.controller.mpc import MPCController
from src.controller.pid import PIDController
from src.controller.heatcurve import HeatCurveController


class SupervisoryController:
    """
    Supervisory controller with two levels:

    1. Primary controller: MPC
    2. Baseline controller: Heat curve + PID

    If MPC fails or returns invalid output, the controller switches
    automatically to the baseline controller.
    """

    def __init__(
        self,
        building_model,
        dt=3600.0,
        horizon=24,
        room_setpoint=20.0,
        T_room_min=20.0,
        T_room_max=24.0,
        supply_limits=(20.0, 50.0),
        pid_gains=(2.0, 0.1, 0.05),
    ):
        self.building = building_model
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.room_setpoint = float(room_setpoint)
        self.T_room_min = float(T_room_min)
        self.T_room_max = float(T_room_max)
        self.supply_limits = tuple(supply_limits)

        self.last_mode = None
        self.last_error = None

        # Primary controller
        self.mpc = MPCController(
            building_model=building_model,
            horizon=self.horizon,
            dt=self.dt
        )

        # Baseline controller = Heat curve + PID
        self.heatcurve = HeatCurveController(
            T_supply_nom=45.0,
            T_return_nom=35.0,
            T_amb_design=-12.0,
            T_room_set=self.room_setpoint
        )

        self.pid = PIDController(
            Kp=pid_gains[0],
            Ki=pid_gains[1],
            Kd=pid_gains[2],
            setpoint=self.room_setpoint,
            output_limits=(-10.0, 10.0)
        )

    def _validate_mpc_output(self, T_supply_opt, cost):
        if T_supply_opt is None:
            raise ValueError("MPC returned None.")
        if len(T_supply_opt) == 0:
            raise ValueError("MPC returned empty output.")
        if not np.all(np.isfinite(T_supply_opt)):
            raise ValueError("MPC returned non-finite values.")
        if not np.isfinite(cost):
            raise ValueError("MPC cost is invalid.")

        T_supply_now = float(T_supply_opt[0])

        if not (self.supply_limits[0] <= T_supply_now <= self.supply_limits[1]):
            raise ValueError("MPC output outside supply limits.")

        return T_supply_now

    def baseline_control(self, T_room, T_amb):
        """
        Baseline controller = Heat curve + PID correction
        """
        T_base = float(self.heatcurve.calc_supply_temp(T_amb))
        pid_correction = float(self.pid.update(measured_value=T_room, dt=self.dt))
        T_supply = T_base + pid_correction
        T_supply = float(np.clip(T_supply, self.supply_limits[0], self.supply_limits[1]))
        return T_supply

    def compute_control(self, state, T_amb_forecast, Q_gains_forecast, price_forecast):
        state = np.asarray(state, dtype=float)
        T_amb_forecast = np.asarray(T_amb_forecast, dtype=float)
        Q_gains_forecast = np.asarray(Q_gains_forecast, dtype=float)
        price_forecast = np.asarray(price_forecast, dtype=float)

        try:
            T_supply_opt, cost = self.mpc.optimize(
                state_init=state.copy(),
                T_amb_forecast=T_amb_forecast,
                Q_gains_forecast=Q_gains_forecast,
                price_forecast=price_forecast,
                T_room_min=self.T_room_min,
                T_room_max=self.T_room_max
            )

            T_supply_now = self._validate_mpc_output(T_supply_opt, cost)

            self.last_mode = "MPC"
            self.last_error = None

            return {
                "mode": "MPC",
                "T_supply": T_supply_now,
                "trajectory": np.asarray(T_supply_opt, dtype=float),
                "cost": float(cost),
                "error": None
            }

        except Exception as e:
            self.last_mode = "BASELINE"
            self.last_error = str(e)

            T_room = float(state[0])
            T_amb = float(T_amb_forecast[0])

            T_supply_baseline = self.baseline_control(T_room=T_room, T_amb=T_amb)

            return {
                "mode": "BASELINE",
                "T_supply": T_supply_baseline,
                "trajectory": None,
                "cost": None,
                "error": str(e)
            }