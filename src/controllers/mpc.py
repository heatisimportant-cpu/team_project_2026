"""
Model Predictive Controller (MPC) for price-optimized heating control
"""

import numpy as np
from scipy.optimize import minimize


class MPCController:
    """
    MPC controller for cost-optimal heating with comfort constraints
    """

    def __init__(self, building_model, horizon=24, dt=3600):
        """
        Parameters
        ----------
        building_model : Building
            Building thermal model
        horizon : int
            Prediction horizon [hours]
        dt : float
            Time step [s]
        """
        self.building = building_model
        self.horizon = horizon
        self.dt = dt

    def optimize(self, state_init, T_amb_forecast, Q_gains_forecast, price_forecast,
                 T_room_min=20.0, T_room_max=24.0):
        """
        Optimize heating schedule over prediction horizon

        Parameters
        ----------
        state_init : array
            Initial state [T_room, T_wall, T_hp_ret]
        T_amb_forecast : array
            Ambient temperature forecast [°C]
        Q_gains_forecast : array
            Internal gains forecast [W]
        price_forecast : array
            Electricity price forecast [EUR/kWh]
        T_room_min : float
            Minimum room temperature [°C]
        T_room_max : float
            Maximum room temperature [°C]

        Returns
        -------
        T_supply_optimal : array
            Optimal supply temperatures [°C]
        cost : float
            Total cost over horizon [EUR]
        """

        n_steps = min(self.horizon, len(T_amb_forecast))

        # Initial guess: heating curve
        T_supply_init = np.zeros(n_steps)
        for i in range(n_steps):
            T_supply_init[i] = 20 + 0.78 * (20 - T_amb_forecast[i])
            T_supply_init[i] = np.clip(T_supply_init[i], 25, 50)

        # Bounds for supply temperature
        bounds = [(25.0, 50.0) for _ in range(n_steps)]

        # Objective function: minimize cost
        def objective(T_supply):
            state = state_init.copy()
            total_cost = 0.0

            for i in range(n_steps):
                # Simulate building
                rhs = self.building.calc_4r3c(
                    0, state, T_supply[i],
                    [T_amb_forecast[i], Q_gains_forecast[i]]
                )
                state = state + rhs * self.dt

                # Estimate heat pump power (simplified COP model)
                COP = 3.5 - 0.03 * (T_supply[i] - T_amb_forecast[i])
                COP = np.clip(COP, 2.0, 5.0)

                cp_water = 4180.0
                Q_heat = self.building.mdot_hp * cp_water * (T_supply[i] - state[2])
                Q_heat = max(0, Q_heat)
                P_el = Q_heat / COP / 1000  # kW

                # Cost
                cost = P_el * price_forecast[i] * (self.dt / 3600)
                total_cost += cost

            return total_cost

        # Constraint: maintain comfort
        def constraint_comfort(T_supply):
            state = state_init.copy()
            violations = 0.0

            for i in range(n_steps):
                rhs = self.building.calc_4r3c(
                    0, state, T_supply[i],
                    [T_amb_forecast[i], Q_gains_forecast[i]]
                )
                state = state + rhs * self.dt

                # Penalize violations
                if state[0] < T_room_min:
                    violations += (T_room_min - state[0])**2
                if state[0] > T_room_max:
                    violations += (state[0] - T_room_max)**2

            return -violations  # Must be >= 0 for constraint

        constraints = [{'type': 'ineq', 'fun': constraint_comfort}]

        # Optimize
        result = minimize(
            objective,
            T_supply_init,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 50, 'disp': False}
        )

        return result.x, result.fun
