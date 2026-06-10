"""
Integrated heating system model

Combines:
- Building (4R3C thermal model)
- Heat pump (iDM AERO ALM 4-12)
- Buffer tank (Kermi x-buffer 500L)
"""

import numpy as np
from .building import Building
from .heatpump import HeatPump
from .tank import BufferTank


class IntegratedSystem:
    """
    Integrated heating system combining building, heat pump, and buffer tank

    System configuration:
    HP (primary) → Tank → Building (secondary)

    Primary circuit: Heat pump → Tank (mdot_hp = 0.27 kg/s)
    Secondary circuit: Tank → Building (mdot_bldg = 0.25 kg/s)
    """

    def __init__(self, building_params, mdot_hp=0.27, mdot_bldg=0.25):
        """
        Parameters
        ----------
        building_params : dict
            Building parameters
        mdot_hp : float
            Heat pump mass flow rate (primary circuit) [kg/s]
        mdot_bldg : float
            Building mass flow rate (secondary circuit) [kg/s]
        """
        # Initialize components
        self.building = Building(building_params, mdot_hp=mdot_bldg)
        self.heatpump = HeatPump(P_nom=12000, mdot_nom=mdot_hp)
        self.tank = BufferTank(volume=500, n_zones=3)

        # Mass flow rates
        self.mdot_hp = mdot_hp
        self.mdot_bldg = mdot_bldg

        # Water properties
        self.cp_water = 4180.0  # J/(kg·K)

    def step(self, state, T_supply_hp, T_amb, Q_gains, dt=3600):
        """
        Simulate one time step of the integrated system

        Parameters
        ----------
        state : array
            System state [T_room, T_wall, T_hp_ret_bldg, T_tank]
        T_supply_hp : float
            Heat pump supply temperature setpoint [°C]
        T_amb : float
            Ambient temperature [°C]
        Q_gains : float
            Internal gains [W]
        dt : float
            Time step [s]

        Returns
        -------
        state_new : array
            New system state
        results : dict
            Simulation results
        """
        # Unpack state
        T_room = state[0]
        T_wall = state[1]
        T_hp_ret_bldg = state[2]
        T_tank = state[3]

        # --- Building dynamics ---
        bldg_state = np.array([T_room, T_wall, T_hp_ret_bldg])
        T_supply_bldg = T_tank  # Tank supplies building

        # Simulate building
        bldg_rhs = self.building.calc_4r3c(0, bldg_state, T_supply_bldg, 
                                          [T_amb, Q_gains])

        # --- Tank dynamics ---
        # Heat from HP to tank
        T_return_hp = T_tank  # Simplified: tank return to HP
        P_el_hp, Q_hp, COP = self.heatpump.calc_power(T_amb, T_supply_hp, T_return_hp, 
                                                      self.mdot_hp)

        # Heat from tank to building
        Q_to_bldg = self.mdot_bldg * self.cp_water * (T_tank - T_hp_ret_bldg)
        Q_to_bldg = max(0, Q_to_bldg)

        # Tank energy balance
        T_tank_new, Q_loss = self.tank.step(T_tank, Q_hp, Q_to_bldg, dt, T_amb)

        # --- Update states ---
        state_new = np.array([
            T_room + bldg_rhs[0] * dt,
            T_wall + bldg_rhs[1] * dt,
            T_hp_ret_bldg + bldg_rhs[2] * dt,
            T_tank_new
        ])

        # --- Results ---
        results = {
            'T_room': state_new[0],
            'T_wall': state_new[1],
            'T_hp_ret_bldg': state_new[2],
            'T_tank': state_new[3],
            'T_supply_hp': T_supply_hp,
            'T_supply_bldg': T_supply_bldg,
            'P_el': P_el_hp,
            'Q_heat': Q_hp,
            'Q_to_bldg': Q_to_bldg,
            'Q_loss': Q_loss,
            'COP': COP,
        }

        return state_new, results

    def simulate(self, x_init, control_sequence, weather, gains, dt=3600):
        """
        Simulate entire sequence

        Parameters
        ----------
        x_init : array
            Initial state [T_room, T_wall, T_hp_ret, T_tank]
        control_sequence : array
            Supply temperature sequence [°C]
        weather : DataFrame
            Weather data with T_amb
        gains : DataFrame
            Internal gains data
        dt : float
            Time step [s]

        Returns
        -------
        results : DataFrame
            Complete simulation results
        """
        n_steps = len(control_sequence)
        state = x_init.copy()

        results_list = []

        for i in range(n_steps):
            T_supply = control_sequence[i]
            T_amb = weather.iloc[i]['T_amb']
            Q_gains_val = gains.iloc[i]['Q_int_total']

            state, res = self.step(state, T_supply, T_amb, Q_gains_val, dt)

            res['timestamp'] = weather.iloc[i]['timestamp']
            res['T_amb'] = T_amb
            res['Q_gains'] = Q_gains_val

            results_list.append(res)

        import pandas as pd
        return pd.DataFrame(results_list)
