"""
Heat pump model - iDM AERO ALM 4-12

Air-to-water heat pump with variable capacity
Based on technical specifications from Data-sheet.pdf
"""

import numpy as np


class HeatPump:
    """
    Heat pump model for iDM AERO ALM 4-12

    Specifications:
    - Type: Air-to-water
    - Capacity: 4-12 kW (modulating)
    - Refrigerant: R-290 (Propane)
    - Nominal mass flow: 0.27 kg/s
    """

    def __init__(self, P_nom=12000, P_min=4000, mdot_nom=0.27):
        """
        Parameters
        ----------
        P_nom : float
            Nominal heating capacity [W]
        P_min : float
            Minimum heating capacity [W]
        mdot_nom : float
            Nominal mass flow rate [kg/s]
        """
        self.P_nom = P_nom
        self.P_min = P_min
        self.mdot_nom = mdot_nom

    def calc_COP(self, T_amb, T_supply):
        """
        Calculate COP based on temperatures

        Uses simplified Carnot efficiency model:
        COP = eta_carnot * COP_carnot

        Parameters
        ----------
        T_amb : float
            Ambient air temperature [°C]
        T_supply : float
            Supply water temperature [°C]

        Returns
        -------
        COP : float
            Coefficient of Performance [-]
        """
        # Convert to Kelvin
        T_hot_K = T_supply + 273.15
        T_cold_K = T_amb + 273.15

        # Carnot COP
        if T_hot_K <= T_cold_K:
            return 2.0  # Minimum COP

        COP_carnot = T_hot_K / (T_hot_K - T_cold_K)

        # Real COP (typically 40-50% of Carnot)
        eta_carnot = 0.45
        COP = eta_carnot * COP_carnot

        # Clip to realistic range
        COP = np.clip(COP, 2.0, 6.0)

        return COP

    def calc_power(self, T_amb, T_supply, T_return, mdot=None):
        """
        Calculate heat pump power consumption and heat output

        Parameters
        ----------
        T_amb : float
            Ambient air temperature [°C]
        T_supply : float
            Supply water temperature [°C]
        T_return : float
            Return water temperature [°C]
        mdot : float, optional
            Mass flow rate [kg/s], default uses nominal

        Returns
        -------
        P_el : float
            Electrical power consumption [W]
        Q_heat : float
            Heat output [W]
        COP : float
            Coefficient of Performance [-]
        """
        if mdot is None:
            mdot = self.mdot_nom

        # Calculate COP
        COP = self.calc_COP(T_amb, T_supply)

        # Heat output from water flow
        cp_water = 4180.0  # J/(kg·K)
        Q_heat = mdot * cp_water * (T_supply - T_return)
        Q_heat = max(0, Q_heat)

        # Clip to capacity limits
        Q_heat = np.clip(Q_heat, self.P_min, self.P_nom)

        # Electrical power
        if Q_heat > 0:
            P_el = Q_heat / COP
        else:
            P_el = 0.0

        return P_el, Q_heat, COP

    def calc_modulation(self, Q_demand):
        """
        Calculate modulation level based on heat demand

        Parameters
        ----------
        Q_demand : float
            Heat demand [W]

        Returns
        -------
        modulation : float
            Modulation level [0-1]
        Q_actual : float
            Actual heat output [W]
        """
        if Q_demand <= 0:
            return 0.0, 0.0

        Q_actual = np.clip(Q_demand, self.P_min, self.P_nom)
        modulation = Q_actual / self.P_nom

        return modulation, Q_actual
