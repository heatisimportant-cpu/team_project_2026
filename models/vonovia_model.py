import numpy as np

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
        self.input_keys  = ('T_hp_sup', 'T_amb', 'Qdot_gains')

        self._calc_bldg_parameters()

    # ── Parameter derivation ──────────────────────────────────────────────────

    def _calc_bldg_parameters(self):
        p    = self.params
        area = p['area_floor']

        p['volume_air'] = area * p['height_room']
        p['H_ve_tr']    = p['H_ve'] + p['H_tr']
        p['H_tr_heavy'] = p['H_tr'] - p['H_tr_light']
        
        # Radiator coupling coefficient
        # Assume radiators are sized for peak heat loss at ΔT = 30K (T_amb = -10°C, T_room = 20°C)
        # with a 20% oversizing factor.
        design_heat_load = p.get('design_heat_load', p['H_ve_tr'] * 30.0 * 1.2)
        # Design radiator temperatures: T_sup=55°C, T_ret=45°C -> T_mean=50°C
        # H_rad_con = Q / (T_mean - T_room)
        p['H_rad_con']  = design_heat_load / (50.0 - 20.0)  # W/K

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
        args : list   [T_hp_sup, T_amb, Qdot_gains]
        """
        T_room     = x[0]
        T_wall     = x[1]
        T_hp_ret   = x[2]
        T_hp_sup   = args[0]
        T_amb      = args[1]
        Qdot_gains = args[2]

        rhs = np.zeros(3)

        rhs[0] = (1 / self.params['C_zone'] *
                  (Qdot_gains
                   + self.params['H_rad_con'] * (T_hp_ret - T_room)
                   - 2 * self.params['H_tr']  * (T_room   - T_wall)
                   - self.params['H_ve']       * (T_room   - T_amb)))

        rhs[1] = (1 / self.params['C_wall'] *
                  (2 * self.params['H_tr'] * (T_room - T_wall)
                   - 2 * self.params['H_tr'] * (T_wall - T_amb)))

        rhs[2] = (1 / self.params['C_water'] *
                  (self.mdot_hp * C_WATER_SPEC * (T_hp_sup - T_hp_ret)
                   - self.params['H_rad_con'] * (T_hp_ret - T_room)))

        return rhs

    # ── Comfort deviation ─────────────────────────────────────────────────────

    def calc_comfort_dev(self, T_room, timestep):
        """Comfort deviation over one integration step.

        Parameters
        ----------
        T_room   : array-like  intermediate T_room values from solve_ivp
        timestep : float       integration timestep [s]

        Returns
        -------
        dict : dev_neg_sum [Kh], dev_neg_max [K], dev_pos_sum [Kh], dev_pos_max [K]
        """
        T = np.asarray(T_room)
        n = len(T)
        dev_neg = np.maximum(self.T_room_set_lower - T, 0.0)
        dev_pos = np.maximum(T - self.T_room_set_upper,  0.0)

        return {
            'dev_neg_sum': float(np.sum(dev_neg) / n * timestep / 3600),
            'dev_neg_max': float(np.max(dev_neg)),
            'dev_pos_sum': float(np.sum(dev_pos) / n * timestep / 3600),
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
vonovia_model = {
    # Thermal properties
    'H_ve':         62.0,     # Ventilation losses [W/K]
    'H_tr':        185.0,     # Envelope transmission [W/K]
    'H_tr_light':   36.0,     # Window transmission [W/K]
    'c_bldg':       35.0,     # Thermal mass [Wh/m²K]

    # Geometry
    'area_floor':  287.92,    # Floor area [m²]
    'height_room':   2.5,     # Room height [m]

    # Heat pump
    'T_offset':     -2.0,     # Temperature offset [K]
    'T_amb_lim':    20.0,     # Outdoor temp above which heating stops [°C]
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
    'design_heat_load': 10968.0,
}

sfh_1919_1948_0_soc = {
    'H_ve': 154.0,
    'H_tr': 973.0,
    'H_tr_light': 146.6 + 6.0,
    'c_bldg': 72.0,
    
    'area_floor': 302.5,
    'height_room': 2.5,
    
    'T_offset': 15.0,
    'T_amb_lim': 20.0,
    'mdot_hp': 0.27,
    
    'windows': {'east': 9.1, 'south': 22.0, 'west': 9.1, 'north': 12.2},
    'tilt': 90.0,
    'frame_fraction': 0.3,
    
    'latitude': 48.0252,
    'longitude': 7.7184,
    'altitude': 207.0,
    'timezone': 'Europe/Berlin',
    
    'name': 'sfh_1919_1948_0_soc',
}

sfh_1949_1957_0_soc = {
    'H_ve': 57.0,
    'H_tr': 465.0,
    'H_tr_light': 51.5 + 6.0,
    'c_bldg': 72.0,
    
    'area_floor': 111.1,
    'height_room': 2.5,
    
    'T_offset': 11.0,
    'T_amb_lim': 20.0,
    'mdot_hp': 0.27,
    
    'windows': {'east': 3.2, 'south': 8.6, 'west': 3.2, 'north': 3.3},
    'tilt': 90.0,
    'frame_fraction': 0.3,
    
    'latitude': 48.0252,
    'longitude': 7.7184,
    'altitude': 207.0,
    'timezone': 'Europe/Berlin',
    
    'name': 'sfh_1949_1957_0_soc',
}

sfh_1958_1968_0_soc = {
    'H_ve': 62.0,
    'H_tr': 496.0,
    'H_tr_light': 75.9 + 6.3,
    'c_bldg': 30.0,
    
    'area_floor': 121.2,
    'height_room': 2.5,
    
    'T_offset': 12.0,
    'T_amb_lim': 20.0,
    'mdot_hp': 0.24,
    
    'windows': {'east': 5.7, 'south': 6.3, 'west': 8.9, 'north': 4.1},
    'tilt': 90.0,
    'frame_fraction': 0.3,
    
    'latitude': 48.0252,
    'longitude': 7.7184,
    'altitude': 207.0,
    'timezone': 'Europe/Berlin',
    
    'name': 'sfh_1958_1968_0_soc',
}

sfh_1969_1978_0_soc = {
    'H_ve': 88.0,
    'H_tr': 493.0,
    'H_tr_light': 95.8 + 6.0,
    'c_bldg': 45.0,
    
    'area_floor': 173.2,
    'height_room': 2.5,
    
    'T_offset': 6.0,
    'T_amb_lim': 20.0,
    'mdot_hp': 0.25,
    
    'windows': {'east': 5.0, 'south': 16.6, 'west': 5.0, 'north': 7.6},
    'tilt': 90.0,
    'frame_fraction': 0.3,
    
    'latitude': 48.0252,
    'longitude': 7.7184,
    'altitude': 207.0,
    'timezone': 'Europe/Berlin',
    
    'name': 'sfh_1969_1978_0_soc',
}

sfh_1979_1983_0_soc = {
    'H_ve': 110.0,
    'H_tr': 364.0,
    'H_tr_light': 116.1 + 6.0,
    'c_bldg': 30.0,
    
    'area_floor': 215.6,
    'height_room': 2.5,
    
    'T_offset': 0.0,
    'T_amb_lim': 20.0,
    'mdot_hp': 0.27,
    
    'windows': {'east': 8.7, 'south': 6.0, 'west': 8.7, 'north': 3.6},
    'tilt': 90.0,
    'frame_fraction': 0.3,
    
    'latitude': 48.0252,
    'longitude': 7.7184,
    'altitude': 207.0,
    'timezone': 'Europe/Berlin',
    
    'name': 'sfh_1979_1983_0_soc',
}

sfh_1984_1994_0_soc = {
    'H_ve': 77.0,
    'H_tr': 319.0,
    'H_tr_light': 94.9 + 6.0,
    'c_bldg': 45.0,
    
    'area_floor': 150.2,
    'height_room': 2.5,
    
    'T_offset': 0.0,
    'T_amb_lim': 20.0,
    'mdot_hp': 0.25,
    
    'windows': {'east': 7.4, 'south': 12.7, 'west': 7.4, 'north': 2.1},
    'tilt': 90.0,
    'frame_fraction': 0.3,
    
    'latitude': 48.0252,
    'longitude': 7.7184,
    'altitude': 207.0,
    'timezone': 'Europe/Berlin',
    
    'name': 'sfh_1984_1994_0_soc',
}

sfh_1995_2001_0_soc = {
    'H_ve': 62.0,
    'H_tr': 197.0,
    'H_tr_light': 61.7 + 4.0,
    'c_bldg': 45.0,
    
    'area_floor': 121.9,
    'height_room': 2.5,
    
    'T_offset': -3.0,
    'T_amb_lim': 20.0,
    'mdot_hp': 0.24,
    
    'windows': {'east': 3.6, 'south': 5.0, 'west': 3.6, 'north': 20.3},
    'tilt': 90.0,
    'frame_fraction': 0.3,
    
    'latitude': 48.0252,
    'longitude': 7.7184,
    'altitude': 207.0,
    'timezone': 'Europe/Berlin',
    
    'name': 'sfh_1995_2001_0_soc',
}

sfh_2002_2009_0_soc = {
    'H_ve': 62.0,
    'H_tr': 152.0,
    'H_tr_light': 39.6 + 4.0,
    'c_bldg': 45.0,
    
    'area_floor': 146.5,
    'height_room': 2.5,
    
    'T_offset': -8.0,
    'T_amb_lim': 20.0,
    'mdot_hp': 0.30,
    
    'windows': {'east': 3.9, 'south': 17.3, 'west': 3.9, 'north': 3.1},
    'tilt': 90.0,
    'frame_fraction': 0.3,
    
    'latitude': 48.0252,
    'longitude': 7.7184,
    'altitude': 207.0,
    'timezone': 'Europe/Berlin',
    
    'name': 'sfh_2002_2009_0_soc',
}

sfh_2010_2015_0_soc = {
    'H_ve': 95.0,
    'H_tr': 194.0,
    'H_tr_light': 54.6 + 4.7,
    'c_bldg': 45.0,
    
    'area_floor': 186.8,
    'height_room': 2.5,
    
    'T_offset': -7.0,
    'T_amb_lim': 20.0,
    'mdot_hp': 0.27,
    
    'windows': {'east': 2.7, 'south': 22.6, 'west': 13.0, 'north': 3.7},
    'tilt': 90.0,
    'frame_fraction': 0.3,
    
    'latitude': 48.0252,
    'longitude': 7.7184,
    'altitude': 207.0,
    'timezone': 'Europe/Berlin',
    
    'name': 'sfh_2010_2015_0_soc',
}

sfh_2016_now_0_soc = {
    'H_ve': 95.0,
    'H_tr': 142.0,
    'H_tr_light': 46.2 + 3.4,
    'c_bldg': 45.0,
    
    'area_floor': 186.8,
    'height_room': 2.5,
    
    'T_offset': -8.0,
    'T_amb_lim': 20.0,
    'mdot_hp': 0.22,
    
    'windows': {'east': 2.7, 'south': 22.6, 'west': 13.0, 'north': 3.7},
    'tilt': 90.0,
    'frame_fraction': 0.3,
    
    'latitude': 48.0252,
    'longitude': 7.7184,
    'altitude': 207.0,
    'timezone': 'Europe/Berlin',
    
    'name': 'sfh_2016_now_0_soc',
}

# ── Collection of available building models ───────────────────────────────────
BUILDING_MODELS = [
    vonovia_model,
    sfh_1919_1948_0_soc,
    sfh_1949_1957_0_soc,
    sfh_1958_1968_0_soc,
    sfh_1969_1978_0_soc,
    sfh_1979_1983_0_soc,
    sfh_1984_1994_0_soc,
    sfh_1995_2001_0_soc,
    sfh_2002_2009_0_soc,
    sfh_2010_2015_0_soc,
    sfh_2016_now_0_soc,
]