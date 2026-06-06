"""
Thermal storage tank model - Kermi x-buffer compact cool 500

Buffer tank with stratification
Based on tank-model.jpg specifications
"""

import numpy as np


class BufferTank:
    """
    Buffer tank model for Kermi x-buffer 500L

    Specifications:
    - Volume: 500 liters
    - Type: Buffer tank with 3 zones
    - Stratification: Simplified 1-node model
    """

    def __init__(self, volume=500, n_zones=3, height=1.8):
        """
        Parameters
        ----------
        volume : float
            Tank volume [liters]
        n_zones : int
            Number of stratification zones (for future extension)
        height : float
            Tank height [m]
        """
        self.volume = volume / 1000  # Convert to m³
        self.n_zones = n_zones
        self.height = height

        # Water properties
        self.rho_water = 1000  # kg/m³
        self.cp_water = 4180  # J/(kg·K)

        # Tank thermal mass
        self.mass = self.rho_water * self.volume
        self.capacity = self.mass * self.cp_water  # J/K

        # Heat loss coefficient (simplified)
        self.UA = 5.0  # W/K (well-insulated tank)

    def calc_dT_dt(self, T_tank, Q_in, Q_out, T_amb=20.0):
        """
        Calculate temperature change rate

        Energy balance: m*cp*dT/dt = Q_in - Q_out - Q_loss

        Parameters
        ----------
        T_tank : float
            Current tank temperature [°C]
        Q_in : float
            Heat input from heat pump [W]
        Q_out : float
            Heat output to building [W]
        T_amb : float
            Ambient temperature around tank [°C]

        Returns
        -------
        dT_dt : float
            Temperature change rate [°C/s]
        """
        # Heat loss to ambient
        Q_loss = self.UA * (T_tank - T_amb)

        # Energy balance
        dT_dt = (Q_in - Q_out - Q_loss) / self.capacity

        return dT_dt

    def step(self, T_tank, Q_in, Q_out, dt=3600, T_amb=20.0):
        """
        Simulate one time step

        Parameters
        ----------
        T_tank : float
            Current tank temperature [°C]
        Q_in : float
            Heat input [W]
        Q_out : float
            Heat output [W]
        dt : float
            Time step [s]
        T_amb : float
            Ambient temperature [°C]

        Returns
        -------
        T_new : float
            New tank temperature [°C]
        Q_loss : float
            Heat loss [W]
        """
        dT_dt = self.calc_dT_dt(T_tank, Q_in, Q_out, T_amb)
        T_new = T_tank + dT_dt * dt

        # Physical limits
        T_new = np.clip(T_new, 5.0, 90.0)

        Q_loss = self.UA * (T_tank - T_amb)

        return T_new, Q_loss

    def calc_stratification(self, T_tank, Q_in, Q_out):
        """
        Calculate stratification (placeholder for future 3-zone model)

        Currently returns single temperature, can be extended to:
        - T_top (supply to building)
        - T_mid (middle zone)
        - T_bottom (return from building)

        Parameters
        ----------
        T_tank : float
            Average tank temperature [°C]
        Q_in : float
            Heat input [W]
        Q_out : float
            Heat output [W]

        Returns
        -------
        T_top : float
            Top zone temperature [°C]
        T_mid : float
            Middle zone temperature [°C]
        T_bottom : float
            Bottom zone temperature [°C]
        """
        # Simplified: assume 3°C stratification
        T_top = T_tank + 1.5
        T_mid = T_tank
        T_bottom = T_tank - 1.5

        return T_top, T_mid, T_bottom
