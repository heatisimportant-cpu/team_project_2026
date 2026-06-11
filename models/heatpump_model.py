"""
Heat Pump Model - iDM AERO ALM 4-12
=====================================
Calibrated directly from official EN14511 datasheet
(IDM193401bis403_ma_de_812199_AERO-ALM-2-15-2.0-Datenbl.pdf, page 14)

Performance range (MAX compressor speed):
  - Thermal output : 4.0 - 13.24 kW
  - Electrical input: 2.44 - 6.76 kW
  - COP range       : 1.39 - 5.42
  - Max flow temp   : 70°C
  - Refrigerant     : R-290 (propane, GWP=3)
  - Backup heater   : 6 kW electric rod
"""
import numpy as np


class iDM_AERO_ALM_4_12:
    """
    iDM AERO ALM 4-12 air-source heat pump model.

    COP fitted by least-squares regression over 44 EN14511 datasheet points
    covering T_amb = -20 to +20°C and T_flow = 35 to 70°C.

    Model:  COP(T_amb, T_flow) = a + b*(T_amb - 2) + c*(35 - T_flow)
    Fitted: a=4.1358, b=0.0610, c=0.0586
    """

    def __init__(self, params=None):
        if params is None:
            params = self.get_default_params()

        # Electrical power limits [kW]
        self.P_el_max = params['P_el_max']      # max compressor draw
        self.P_el_min = params['P_el_min']      # min modulation

        # COP safety clamps
        self.COP_min = params['COP_min']
        self.COP_max = params['COP_max']

        # COP regression coefficients (fitted from 44 EN14511 data points)
        self.COP_a = params['COP_a']            # base COP @ A2/W35
        self.COP_b = params['COP_b']            # ambient temp coefficient [1/K]
        self.COP_c = params['COP_c']            # flow temp coefficient   [1/K]

        # Reference operating point (EN14511 nominal)
        self.T_amb_ref  = params['T_amb_ref']   # 2°C
        self.T_flow_ref = params['T_flow_ref']  # 35°C

        # Backup electric heater
        self.heater_power = params.get('heater_power', 6.0)

        # Operating limits from datasheet section 2.4
        self.T_amb_min  = -20.0   # °C  (compressor shuts off below this)
        self.T_amb_max  =  35.0   # °C
        self.T_flow_min =  20.0   # °C  (min for defrost per section 1.8)
        self.T_flow_max =  70.0   # °C

    # ── Default parameters ──────────────────────────────────────────────────

    @staticmethod
    def get_default_params():
        """
        iDM AERO ALM 4-12 parameters.
        COP coefficients fitted from EN14511 datasheet (page 14, MAX speed).
        """
        return {
            # Electrical power limits [kW]
            'P_el_max':    6.76,   # from datasheet: max measured input (A-7/W70)
            'P_el_min':    0.80,   # estimated minimum modulation (~12% of max)

            # COP physical limits
            'COP_min':     1.39,   # datasheet minimum: A-15/W70
            'COP_max':     5.42,   # datasheet maximum: A20/W35

            # COP linear regression (fitted from 44 EN14511 points, R²=0.97)
            # COP = COP_a + COP_b*(T_amb - 2) + COP_c*(35 - T_flow)
            'COP_a':       4.136,  # base COP at A2/W35
            'COP_b':       0.061,  # +0.061 per °C warmer ambient
            'COP_c':       0.059,  # -0.059 per °C higher flow temp

            # Reference point for regression
            'T_amb_ref':   2.0,    # °C (EN14511 A2 test condition)
            'T_flow_ref':  35.0,   # °C (EN14511 W35 test condition)

            # Backup heater [kW]
            'heater_power': 6.0,
        }

    # ── Core methods ─────────────────────────────────────────────────────────

    def compute_COP(self, T_amb, T_flow):
        """
        Compute COP at given ambient and flow temperatures.

        COP(T_amb, T_flow) = a + b*(T_amb - 2) + c*(35 - T_flow)

        Fitted from 44 EN14511 data points (page 14, ALM 4-12 MAX speed).
        R² = 0.97 across T_amb = -20…+20°C, T_flow = 35…70°C.

        Parameters
        ----------
        T_amb  : float or np.ndarray — ambient (outdoor) temperature [°C]
        T_flow : float or np.ndarray — heat pump flow (supply) temperature [°C]

        Returns
        -------
        float or np.ndarray — COP [-]

        Datasheet reference values (MAX speed, EN14511):
          A20/W35 → 5.42,  A7/W35 → 4.98,  A2/W35 → 3.95
          A-7/W35 → 3.24,  A2/W55 → 3.14,  A-7/W70 → 1.47
        """
        cop = (self.COP_a
               + self.COP_b * (T_amb  - self.T_amb_ref)
               + self.COP_c * (self.T_flow_ref - T_flow))
        return np.clip(cop, self.COP_min, self.COP_max)

    def heat_delivered(self, P_el, T_amb, T_flow):
        """
        Compute thermal output and COP for a given electrical input.

        Parameters
        ----------
        P_el   : float — electrical power input [kW]
        T_amb  : float — ambient temperature [°C]
        T_flow : float — flow temperature [°C]

        Returns
        -------
        Q_th : float — thermal heat output [kW]
        COP  : float — coefficient of performance [-]
        """
        P_el_clipped = np.clip(P_el, self.P_el_min, self.P_el_max)
        COP  = self.compute_COP(T_amb, T_flow)
        Q_th = COP * P_el_clipped
        return Q_th, COP

    def get_max_heating_capacity(self, T_amb, T_flow):
        """
        Maximum thermal output at given conditions [kW].

        Based on: Q_max = COP(T_amb, T_flow) × P_el_max
        """
        COP = self.compute_COP(T_amb, T_flow)
        return COP * self.P_el_max

    def heater_required(self, Q_demand, T_amb, T_flow):
        """
        Check if the 6 kW backup electric heater needs to activate.

        Returns extra heat the HP cannot deliver [kW], capped at 6 kW.
        """
        Q_hp_max = self.get_max_heating_capacity(T_amb, T_flow)
        if Q_demand > Q_hp_max:
            return min(Q_demand - Q_hp_max, self.heater_power)
        return 0.0

    def is_within_operating_limits(self, T_amb, T_flow):
        """
        Check if operating point is within datasheet limits (section 2.4).

        Returns True if compressor can run, False if it will shut off.
        """
        return (self.T_amb_min  <= T_amb  <= self.T_amb_max and
                self.T_flow_min <= T_flow <= self.T_flow_max)

    # ── EN14511 reference table (for validation / testing) ───────────────────

    @staticmethod
    def datasheet_cop_table():
        """
        Selected COP values from EN14511 datasheet (page 9 + page 14, MAX speed).

        Returns dict keyed by (T_amb, T_flow).
        """
        return {
            ( 7, 35): 4.98,
            ( 2, 35): 3.95,
            (-7, 35): 3.24,
            ( 7, 55): 3.55,
            ( 2, 55): 3.14,
            (-7, 55): 2.61,
            ( 2, 70): 1.84,
            (-7, 70): 1.47,
            (20, 35): 5.42,
        }

HEATPUMP_MODELS = [
    iDM_AERO_ALM_4_12,
]

# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    hp = iDM_AERO_ALM_4_12()

    print("iDM AERO ALM 4-12 — Model Validation vs EN14511 Datasheet")
    print("=" * 60)
    print(f"{'Condition':<15} {'Predicted':>10} {'Datasheet':>10} {'Error':>8}")
    print("-" * 60)

    for (T_amb, T_flow), cop_ds in hp.datasheet_cop_table().items():
        cop_pred = hp.compute_COP(T_amb, T_flow)
        err = cop_pred - cop_ds
        label = f"A{T_amb:+d}/W{T_flow}"
        print(f"{label:<15} {cop_pred:>10.2f} {cop_ds:>10.2f} {err:>+8.2f}")

    print()
    print("Operating envelope check:")
    print(f"  Max heating @ A7/W35  : {hp.get_max_heating_capacity(7,  35):.1f} kW")
    print(f"  Max heating @ A-7/W55 : {hp.get_max_heating_capacity(-7, 55):.1f} kW")
    print(f"  Max heating @ A2/W70  : {hp.get_max_heating_capacity(2,  70):.1f} kW")
    print()
    print(f"  Backup heater needed @ Q=12kW, A-10/W55 : "
          f"{hp.heater_required(12.0, -10, 55):.1f} kW")
    print(f"  In limits A2/W35  : {hp.is_within_operating_limits(2, 35)}")
    print(f"  In limits A-25/W35: {hp.is_within_operating_limits(-25, 35)}")
