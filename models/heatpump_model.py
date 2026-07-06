import numpy as np
from scipy.interpolate import RegularGridInterpolator


class iDM_AERO_ALM_4_12:
    _TAMB = np.array([-20, -15, -10, -7, 2, 7, 10, 12, 15, 20], dtype=float)
    _TFLOW = np.array([35, 45, 50, 55, 60, 70], dtype=float)

    # Heizleistung [kW] per Tflow row, columns matching _TAMB order
    _QMAX = {
        35: [7.95, 8.97, 9.85, 10.30, 11.80, 12.41, 12.95, 13.10, 13.12, 13.24],
        45: [7.94, 8.84, 9.72, 10.14, 11.27, 12.06, 12.51, 12.60, 12.68, 12.75],
        50: [7.86, 8.76, 9.65, 10.00, 11.01, 11.87, 12.29, 12.35, 12.46, 12.51],
        55: [7.99, 8.67, 9.58, 9.85, 10.74, 11.68, 12.07, 12.10, 12.24, 12.26],
        60: [7.71, 8.59, 9.43, 9.71, 10.47, 11.49, 11.85, 11.85, 12.02, 12.02],
        70: [None, None, 9.21, 9.42, 9.94, 10.59, 11.02, 11.35, 11.38, 11.46],
    }
    # Elektrische Leistungsaufnahme [kW] per Tflow row
    _PMAX = {
        35: [3.94, 3.82, 3.79, 3.73, 3.64, 3.14, 2.60, 2.48, 2.45, 2.44],
        45: [4.42, 4.35, 4.21, 4.14, 4.00, 3.52, 3.17, 2.94, 2.89, 2.85],
        50: [4.61, 4.63, 4.45, 4.44, 4.22, 3.78, 3.46, 3.23, 3.15, 3.10],
        55: [4.82, 4.95, 4.72, 4.80, 4.48, 4.10, 3.83, 3.61, 3.48, 3.42],
        60: [5.06, 5.33, 5.30, 5.25, 5.13, 4.49, 4.31, 4.11, 3.89, 3.84],
        70: [None, None, 6.63, 6.49, 6.76, 5.76, 4.99, 4.89, 4.82, 4.74],
    }

    def __init__(self, params=None):
        if params is None:
            params = self.get_default_params()

        self.P_el_max = params['P_el_max']
        self.P_el_min = params['P_el_min']
        self.COP_min = params['COP_min']
        self.COP_max = params['COP_max']
        self.heater_power = params.get('heater_power', 6.0)

        # Operating limits (datasheet section 2.4/2.5)
        self.T_amb_min = -20.0
        self.T_amb_max = 35.0
        self.T_flow_min = 20.0
        self.T_flow_max = 70.0

        # ── Build interpolators from the real grid ────────────────────
        def grid(d):
            return np.array([[v if v is not None else np.nan for v in d[tf]]
                              for tf in self._TFLOW])

        Qg = grid(self._QMAX)
        Pg = grid(self._PMAX)
        # Fill NaN edge cells (Tflow=70, cold Tamb -- outside the real
        # envelope for that flow temp) by holding the nearest valid value,
        # just so the interpolator has no NaNs to propagate.
        for g in (Qg, Pg):
            for i in range(g.shape[0]):
                row = g[i]
                valid = ~np.isnan(row)
                if valid.any() and not valid.all():
                    first_valid = np.argmax(valid)
                    row[:first_valid] = row[first_valid]

        self._Q_interp = RegularGridInterpolator(
            (self._TFLOW, self._TAMB), Qg, bounds_error=False, fill_value=None)
        self._P_interp = RegularGridInterpolator(
            (self._TFLOW, self._TAMB), Pg, bounds_error=False, fill_value=None)

    @staticmethod
    def get_default_params():
        return {
            'P_el_max': 6.76,   # kW, datasheet max electrical input (A-7/W70 region)
            'P_el_min': 0.61,   # kW, lowest MIN-speed input in the real table (A20/W35)
            'COP_min': 1.39,    # datasheet minimum (A-15/W70 area, extrapolated)
            'COP_max': 5.42,    # datasheet maximum (A20/W35)
            'heater_power': 6.0,
        }

    # ── Core methods ─────────────────────────────────────────────────────

    def _lookup(self, T_amb, T_flow):
        """Bilinear lookup of (Q_max [kW], P_max [kW]) at the MAX-speed curve.
        Inputs are clipped to the real datasheet grid range before lookup."""
        Ta = np.clip(T_amb, self._TAMB.min(), self._TAMB.max())
        Tf = np.clip(T_flow, self._TFLOW.min(), self._TFLOW.max())
        Q = float(self._Q_interp(np.array([[Tf, Ta]]))[0])
        P = float(self._P_interp(np.array([[Tf, Ta]]))[0])
        return Q, P

    def compute_COP(self, T_amb, T_flow):
        """
        COP(T_amb, T_flow) via bilinear interpolation of the real
        datasheet MAX-speed Heizleistung/Leistungsaufnahme grid.

        Reproduces the datasheet exactly at measured (Tamb, Tflow) points,
        e.g. COP(2, 35) = 3.24, COP(7, 35) = 3.95, COP(-7, 35) = 2.76,
        COP(2, 55) = 2.40 -- matching p.14 of the AERO ALM datasheet.
        """
        Q, P = self._lookup(T_amb, T_flow)
        cop = Q / P if P > 0 else self.COP_min
        return float(np.clip(cop, self.COP_min, self.COP_max))

    def heat_delivered(self, P_el, T_amb, T_flow):
        """Thermal output for a given electrical input, scaled from the
        max-capacity COP at this (T_amb, T_flow) -- the same simplification
        the original model made for part-load behavior."""
        COP = self.compute_COP(T_amb, T_flow)
        P_el_clipped = np.clip(P_el, self.P_el_min, self.P_el_max)
        return COP * P_el_clipped, COP

    def get_max_heating_capacity(self, T_amb, T_flow):
        """Real max thermal output [kW] at this operating point, read
        directly off the datasheet grid (not COP * P_el_max, which would
        overstate capacity at cold/high-flow-temp points where the
        compressor itself is capacity-limited, not just power-limited)."""
        Q, _ = self._lookup(T_amb, T_flow)
        return Q

    def heater_required(self, Q_demand, T_amb, T_flow):
        Q_hp_max = self.get_max_heating_capacity(T_amb, T_flow)
        if Q_demand > Q_hp_max:
            return min(Q_demand - Q_hp_max, self.heater_power)
        return 0.0

    def is_within_operating_limits(self, T_amb, T_flow):
        return (self.T_amb_min <= T_amb <= self.T_amb_max and
                self.T_flow_min <= T_flow <= self.T_flow_max)

    @staticmethod
    def datasheet_cop_table():
        """Real EN14511 MAX-speed COP values, AERO ALM 4-12, p.14
        (Heizleistung / Leistungsaufnahme), for self-test/validation."""
        return {
            (20, 35): round(13.24 / 2.44, 2),
            (7, 35):  round(12.41 / 3.14, 2),
            (2, 35):  round(11.80 / 3.64, 2),
            (-7, 35): round(10.30 / 3.73, 2),
            (7, 55):  round(11.68 / 4.10, 2),
            (2, 55):  round(10.74 / 4.48, 2),
            (-7, 55): round(9.85 / 4.80, 2),
            (2, 70):  round(9.94 / 6.76, 2),
            (-7, 70): round(9.42 / 6.49, 2),
        }


if __name__ == "__main__":
    hp = iDM_AERO_ALM_4_12()
    print("iDM AERO ALM 4-12 - Model Validation vs real EN14511 Datasheet (MAX speed)")
    print("=" * 64)
    print(f"{'Condition':<15} {'Predicted':>10} {'Datasheet':>10} {'Error':>8}")
    print("-" * 64)
    for (T_amb, T_flow), cop_ds in hp.datasheet_cop_table().items():
        cop_pred = hp.compute_COP(T_amb, T_flow)
        err = cop_pred - cop_ds
        label = f"A{T_amb:+d}/W{T_flow}"
        print(f"{label:<15} {cop_pred:>10.2f} {cop_ds:>10.2f} {err:>+8.3f}")

    print()
    print("Operating envelope check:")
    print(f"  Max heating @ A7/W35  : {hp.get_max_heating_capacity(7, 35):.2f} kW (datasheet: 12.41)")
    print(f"  Max heating @ A-7/W55 : {hp.get_max_heating_capacity(-7, 55):.2f} kW (datasheet: 9.85)")
    print(f"  Max heating @ A2/W70  : {hp.get_max_heating_capacity(2, 70):.2f} kW (datasheet: 9.94)")