import numpy as np

_trapz = getattr(np, 'trapezoid', None) or np.trapz

# ── Physical constants (i4b constants.py) ────────────────────────────────────
RHO_WATER    = 997      # kg/m³
RHO_AIR      = 1.225    # kg/m³
C_WATER_SPEC = 4181     # J/(kg·K)
C_AIR_SPEC   = 1005.0   # J/(kg·K)
C_INT_SPEC   = 10000    # J/(m²·K)
V_TS_SPEC    = 5.0      # l/m²   thermal storage volume per floor area
V_UFH_SPEC   = 1.5      # l/m²   UFH water volume per floor area
H_UFH_SPEC   = 4.4      # W/(m²·K)  emitter coupling coefficient


class Building:
    """4R3C thermal building model.

    Parameters
    ----------
    params : dict
        Building parameter dict (e.g. vonovia_model).
    mdot_hp : float
        HP hydraulic circuit mass flow rate [kg/s].
    T_room_set_lower : float
        Lower comfort setpoint [°C].
    T_room_set_upper : float
        Upper comfort setpoint [°C].
    """

    def __init__(self,
                 params,
                 mdot_hp=0.27,
                 T_room_set_lower=20.0,
                 T_room_set_upper=26.0):

        self.params = dict(params)
        self.mdot_hp = mdot_hp
        self.T_room_set_lower = T_room_set_lower
        self.T_room_set_upper = T_room_set_upper
        self.method = '4R3C'

        self.state_keys = ('T_room', 'T_wall', 'T_hp_ret')
        self.input_keys  = ('T_hp_sup', 'T_amb', 'Qdot_gains', 'Qdot_hp_max')

        self._calc_bldg_parameters()

    # ── Parameter derivation ──────────────────────────────────────────────────

    def _calc_bldg_parameters(self):
        p    = self.params
        area = p['area_floor']

        p['volume_air'] = area * p['height_room']
        # H_tr_heavy (opaque: walls/roof/floor) and H_tr_light (windows) are now
        # both authoritative inputs (back-solved from the real design heat load,
        # see derivation notes below `vonovia_model` dict) rather than one being
        # derived from the other.
        p['H_tr']       = p['H_tr_heavy'] + p['H_tr_light']   # kept for reporting/back-compat
        p['H_ve_tr']    = p['H_ve'] + p['H_tr']
        # i4b uses H_UFH_SPEC * area for underfloor heating.
        # This building has radiators → derive from datasheet design conditions:
        # Q=10968 W at T_sup=55°C, T_ret=45°C (T_mean=50°C), T_room=20°C
        # H_rad_con = Q / (T_mean - T_room) = 10968 / (50-20) = 365.6 W/K
        p['H_rad_con']  = 10968.0 / (50.0 - 20.0)  # W/K

        p['C_air']   = RHO_AIR * C_AIR_SPEC * p['volume_air']
        p['C_int']   = C_INT_SPEC * area
        p['C_zone']  = p['C_air'] + p['C_int']
        p['C_bldg']  = p['c_bldg'] * area * 3600
        p['C_wall']  = p['C_bldg'] - p['C_zone']

        volume_water = (V_TS_SPEC + V_UFH_SPEC) * area / 1000
        p['C_water'] = RHO_WATER * C_WATER_SPEC * volume_water

    # ── ODE (called by scipy.integrate.solve_ivp) ────────────────────────────

    def calc(self, t, x, args):
        """4R3C ODE right-hand side.

        Parameters
        ----------
        t    : float  (unused — required by solve_ivp)
        x    : array  [T_room, T_wall, T_hp_ret]
        args : list   [T_hp_sup, T_amb, Qdot_gains, Qdot_hp_max]
        """
        T_room      = x[0]
        T_wall      = x[1]
        T_hp_ret    = x[2]
        T_hp_sup    = args[0]
        T_amb       = args[1]
        Qdot_gains  = args[2]
        Qdot_hp_max = args[3]   # W -- real compressor capacity ceiling at this (T_amb, T_hp_sup)

        rhs = np.zeros(3)

        # The compressor cannot deliver more heat than its real datasheet
        # capacity allows, even if (T_hp_sup - T_hp_ret) implies more.
        # Without this cap, the ODE models an idealized infinite-capacity
        # heat source and the agent could "request" physically impossible
        # heating rates (e.g. high T_hp_sup at very cold T_amb).
        Qdot_hp_requested = self.mdot_hp * C_WATER_SPEC * (T_hp_sup - T_hp_ret)
        Qdot_hp = np.clip(Qdot_hp_requested, 0.0, Qdot_hp_max)

        rhs[0] = (1 / self.params['C_zone'] *
                  (Qdot_gains
                   + self.params['H_rad_con']  * (T_hp_ret - T_room)
                   - 2 * self.params['H_tr_heavy'] * (T_room - T_wall)
                   - self.params['H_tr_light']      * (T_room - T_amb)   # windows: fast path, no thermal mass
                   - self.params['H_ve']            * (T_room - T_amb)))

        rhs[1] = (1 / self.params['C_wall'] *
                  (2 * self.params['H_tr_heavy'] * (T_room - T_wall)
                   - 2 * self.params['H_tr_heavy'] * (T_wall - T_amb)))

        rhs[2] = (1 / self.params['C_water'] *
                  (Qdot_hp
                   - self.params['H_rad_con'] * (T_hp_ret - T_room)))

        return rhs

    # ── Comfort deviation ─────────────────────────────────────────────────────

    def calc_comfort_dev(self, T_room, timestep, t=None):
        """Comfort deviation over one integration step.

        Parameters
        ----------
        T_room   : array-like  intermediate T_room values from solve_ivp
        timestep : float       integration timestep [s]
        t        : array-like, optional
            The solver's actual (non-uniform) time points for T_room, in
            the same units as `timestep`. solve_ivp with an adaptive
            method (e.g. LSODA) does NOT space its internal steps evenly
            -- it samples densely during fast transients and sparsely
            during steady periods. A plain mean/sum over those samples
            therefore over-weights whatever phase of the response the
            solver happened to sample densely. If `t` is provided, this
            uses trapezoidal (time-weighted) integration instead; if not,
            falls back to the old uniform-sample assumption.

        Returns
        -------
        dict : dev_neg_sum [Kh], dev_neg_max [K], dev_pos_sum [Kh], dev_pos_max [K]
        """
        T = np.asarray(T_room)
        dev_neg = np.maximum(self.T_room_set_lower - T, 0.0)
        dev_pos = np.maximum(T - self.T_room_set_upper,  0.0)

        if t is not None and len(t) > 1:
            t = np.asarray(t)
            dev_neg_sum = float(_trapz(dev_neg, t) / 3600.0)   # Kh
            dev_pos_sum = float(_trapz(dev_pos, t) / 3600.0)   # Kh
        else:
            n = len(T)
            dev_neg_sum = float(np.sum(dev_neg) / n * timestep / 3600)
            dev_pos_sum = float(np.sum(dev_pos) / n * timestep / 3600)

        return {
            'dev_neg_sum': dev_neg_sum,
            'dev_neg_max': float(np.max(dev_neg)),
            'dev_pos_sum': dev_pos_sum,
            'dev_pos_max': float(np.max(dev_pos)),
        }

    # ── Info ──────────────────────────────────────────────────────────────────

    def print_params(self):
        p = self.params
        tau = p['C_bldg'] / p['H_ve_tr'] / 3600
        print(f"Building     : {p['name']}")
        print(f"Floor area   : {p['area_floor']} m²")
        print(f"H_ve + H_tr  : {p['H_ve_tr']:.1f} W/K")
        print(f"H_rad_con    : {p['H_rad_con']:.1f} W/K")
        print(f"C_bldg       : {p['C_bldg']/3.6e6:.2f} kWh/K")
        print(f"C_zone       : {p['C_zone']/1e6:.2f} MJ/K")
        print(f"C_wall       : {p['C_wall']/1e6:.2f} MJ/K")
        print(f"C_water      : {p['C_water']/1e3:.1f} kJ/K")
        print(f"τ (time const): {tau:.1f} h")

# ── Building parameter dict ───────────────────────────────────────────────────
#
# H_ve / H_tr_heavy / H_tr_light derivation (replaces earlier TABULA-archetype
# guess, which under-shot the building's real heat loss by ~28%):
#
#   Real geometry (from Vonovia hydraulic-balance sheet, Virchowstr. 6):
#     building width x length x height = 10.4 x 15.755 x 9.0 m
#     footprint (roof + floor/slab)     = 163.85 m^2
#     gross wall area (perimeter*height)= 470.8  m^2
#     window area (4 facades, as below) = 51.0   m^2  ->  net wall = 419.8 m^2
#
#   Targets to match (from project documents):
#     design heat load  = 10.968 kW  @ Hannover norm outdoor temp ~ -12 degC,
#                          T_room = 20 degC  ->  delta_T = 32 K
#                          ->  target total UA = 10968 / 32 = 342.75 W/K
#     annual demand      = 30,828 kWh/a  (cross-check via degree-days, consistent
#                          with UA ~ 340-345 W/K at Hannover HDD ~ 3300-3700 Kd)
#
#   H_ve   = 62 W/K   (kept: 0.34 * n * V, n=0.253 1/h, V=719.8 m^3 heated volume
#                       -- ventilation rate is independent of the envelope-area question)
#   H_tr_light = 36 W/K  (U_window=0.70 W/m^2K * 51 m^2 real window area)
#   H_tr_heavy = target_UA - H_ve - H_tr_light = 342.75 - 62 - 36 ~ 245 W/K
#                (opaque envelope, spread over 747.5 m^2 real opaque area
#                 -> implied U_opaque ~ 0.33 W/m^2K, plausible for a partially
#                 refurbished 1968 masonry MFH -- between EnEV (~0.28) and
#                 unrenovated original stock (~0.8-1.0))
#
vonovia_model = {
    # Thermal properties
    'H_ve':         62.0,     # Ventilation losses [W/K]
    'H_tr_heavy':  245.0,     # Opaque envelope (walls/roof/floor) [W/K] -- thermal-mass coupled
    'H_tr_light':   36.0,     # Window transmission [W/K] -- fast/massless path
    'c_bldg':       35.0,     # Thermal mass [Wh/m²K]

    # Geometry
    'area_floor':  287.92,    # Floor area [m²]
    'height_room':   2.5,     # Room height [m]

    # Heat pump
    'T_offset':     -2.0,     # Temperature offset [K]
    'T_amb_lim':   15.0,     # Outdoor temp above which heating stops [°C]
    'mdot_hp':       0.27,    # HP mass flow rate [kg/s]

    # Windows [m²]
    'windows': {'east': 12.0, 'south': 18.0, 'west': 12.0, 'north': 9.0},
    'tilt':           90.0,
    'frame_fraction':  0.3,

    # Location
    'latitude':   52.44,
    'longitude':   9.74,
    'altitude':   55.0,
    'timezone':   'Europe/Berlin',

    # Name
    'name': 'vonovia_model',
}