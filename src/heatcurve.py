"""
Heat curve controller for baseline heating control
"""

import numpy as np


class HeatCurveController:
    """
    Heat curve controller calculates supply temperature based on outdoor temperature.

    T_supply = T_room_set + slope * (T_room_set - T_amb)
    """

    def __init__(self, T_supply_nom=45.0, T_return_nom=35.0, 
                 T_amb_design=-12.0, T_room_set=20.0):
        """
        Parameters
        ----------
        T_supply_nom : float
            Nominal supply temperature at design conditions [°C]
        T_return_nom : float
            Nominal return temperature [°C]
        T_amb_design : float
            Design outdoor temperature [°C]
        T_room_set : float
            Room temperature setpoint [°C]
        """
        self.T_supply_nom = T_supply_nom
        self.T_return_nom = T_return_nom
        self.T_amb_design = T_amb_design
        self.T_room_set = T_room_set

        # Calculate heating curve slope
        self.slope = (T_supply_nom - T_room_set) / (T_room_set - T_amb_design)

    def calc_supply_temp(self, T_amb):
        """
        Calculate supply temperature based on outdoor temperature

        Parameters
        ----------
        T_amb : float or array
            Outdoor temperature [°C]

        Returns
        -------
        T_supply : float or array
            Supply temperature [°C]
        """
        T_supply = self.T_room_set + self.slope * (self.T_room_set - T_amb)

        # Clip to reasonable range for underfloor heating
        T_supply = np.clip(T_supply, 20.0, 50.0)

        return T_supply