"""
Building parameters for Virchowstr. 6, Langenhagen

Multi-family house (MFH) built in 1968
Three refurbishment states: Original, EnEV, KfW

Based on detailed calculations from building documentation
and TABULA webtool for German MFH archetypes.
"""


# ============================================================================
# MFH 1968 - EnEV STANDARD (Current state after 2016 refurbishment)
# ============================================================================

mfh_1968_enev = {
    'name': 'MFH 1968 EnEV Standard',
    'location': 'Langenhagen, Germany',
    'type': 'Multi-family house',
    'construction_year': 1968,
    'refurbishment_year': 2016,
    'refurbishment_standard': 'EnEV',

    # Main thermal parameters (calculated)
    'H_ve': 62.0,      # Ventilation heat loss [W/K]
    'H_tr': 185.0,     # Transmission heat loss [W/K]
    'H_tr_light': 36.0,  # Light components (windows) [W/K]
    'c_bldg': 35.0,    # Specific thermal capacity [Wh/(m²·K)]

    # Geometry
    'area_floor': 287.92,  # Heated floor area [m²]
    'height_room': 2.5,    # Average room height [m]

    # Control parameters
    'T_offset': -2.0,      # Heating curve offset [K]
    'T_amb_lim': 20.0,     # Heating limit temperature [°C]
    'mdot_hp': 0.27,       # Heat pump mass flow [kg/s]

    # Window geometry (all orientations)
    'windows': {
        'east': {
            'area': 12.0,      # [m²]
            'tilt': 90,        # [degrees]
            'azimuth': 90,     # [degrees]
            'c_frame': 0.3,    # Frame fraction
        },
        'south': {
            'area': 18.0,
            'tilt': 90,
            'azimuth': 180,
            'c_frame': 0.3,
        },
        'west': {
            'area': 12.0,
            'tilt': 90,
            'azimuth': 270,
            'c_frame': 0.3,
        },
        'north': {
            'area': 9.0,
            'tilt': 90,
            'azimuth': 0,
            'c_frame': 0.3,
        },
    },

    # Location
    'position': {
        'lat': 52.44,
        'long': 9.74,
        'altitude': 55,
        'timezone': 'Europe/Berlin',
    },

    # Heating system
    'heating_type': 'UFH',
}


# ============================================================================
# MFH 1968 - ORIGINAL STATE (Before refurbishment)
# ============================================================================

mfh_1968_original = {
    'name': 'MFH 1968 Original State',
    'location': 'Langenhagen, Germany',
    'type': 'Multi-family house',
    'construction_year': 1968,
    'refurbishment_year': None,
    'refurbishment_standard': 'Original',

    # Main thermal parameters (highest losses)
    'H_ve': 72.0,      # Higher ventilation loss
    'H_tr': 310.0,     # Higher transmission loss
    'H_tr_light': 58.0,  # Higher window losses
    'c_bldg': 35.0,

    # Geometry (same)
    'area_floor': 287.92,
    'height_room': 2.5,

    # Control parameters
    'T_offset': 10.0,      # Higher offset for poor insulation
    'T_amb_lim': 20.0,
    'mdot_hp': 0.30,       # Higher flow for higher losses

    # Windows (same geometry)
    'windows': mfh_1968_enev['windows'],
    'position': mfh_1968_enev['position'],
    'heating_type': 'UFH',
}


# ============================================================================
# MFH 1968 - KfW STANDARD (Best efficiency)
# ============================================================================

mfh_1968_kfw = {
    'name': 'MFH 1968 KfW Standard',
    'location': 'Langenhagen, Germany',
    'type': 'Multi-family house',
    'construction_year': 1968,
    'refurbishment_year': 2016,
    'refurbishment_standard': 'KfW',

    # Main thermal parameters (lowest losses)
    'H_ve': 54.0,      # Best ventilation (possibly with heat recovery)
    'H_tr': 118.0,     # Best envelope insulation
    'H_tr_light': 24.0,  # Best windows
    'c_bldg': 35.0,

    # Geometry (same)
    'area_floor': 287.92,
    'height_room': 2.5,

    # Control parameters
    'T_offset': -7.0,      # Lowest offset for best insulation
    'T_amb_lim': 20.0,
    'mdot_hp': 0.24,       # Lower flow sufficient

    # Windows (same geometry)
    'windows': mfh_1968_enev['windows'],
    'position': mfh_1968_enev['position'],
    'heating_type': 'UFH',
}


# ============================================================================
# HELPER FUNCTION
# ============================================================================

def get_building(variant='enev'):
    """
    Get building parameters for specified variant

    Parameters
    ----------
    variant : str
        Building variant: 'original', 'enev', or 'kfw'

    Returns
    -------
    dict
        Building parameters dictionary
    """
    variants = {
        'original': mfh_1968_original,
        'enev': mfh_1968_enev,
        'kfw': mfh_1968_kfw,
    }

    if variant.lower() not in variants:
        print(f"Warning: Unknown variant '{variant}', using 'enev'")
        variant = 'enev'

    return variants[variant.lower()].copy()


# Default export (EnEV standard - current state)
mfh_1968 = mfh_1968_enev


# ============================================================================
# DETAILED CALCULATION NOTES
# ============================================================================

"""
CALCULATION METHODOLOGY:

1. H_ve (Ventilation heat loss):
   H_ve = 0.34 × n × V
   where:
   - 0.34 = air heat capacity factor [Wh/(m³·K)]
   - n = air change rate [1/h]
   - V = heated volume [m³] = area_floor × height_room

   EnEV: V = 287.92 × 2.5 = 719.8 m³
         n = 0.253 h⁻¹ (refurbished)
         H_ve = 0.34 × 0.253 × 719.8 = 62 W/K

2. H_tr (Transmission heat loss):
   H_tr = Σ(U_i × A_i)

   EnEV example:
   - External walls: 150 m² × 0.28 W/(m²·K) = 42 W/K
   - Roof: 120 m² × 0.20 W/(m²·K) = 24 W/K
   - Floor: 120 m² × 0.25 W/(m²·K) = 30 W/K
   - Other components: 89 W/K
   Total: 185 W/K

3. H_tr_light (Windows):
   H_tr_light = U_win × A_win

   EnEV: A_win = 12 + 18 + 12 + 9 = 51 m²
         U_win = 0.70 W/(m²·K) (double glazing)
         H_tr_light = 0.70 × 51 = 36 W/K

4. c_bldg (Thermal capacity):
   c_bldg = C_building / area_floor

   Assuming C_building ≈ 10,000 Wh/K (masonry building)
   c_bldg = 10,000 / 287.92 ≈ 35 Wh/(m²·K)

COMPARISON OF VARIANTS:

Parameter       Original    EnEV      KfW
------------------------------------------------
H_ve [W/K]      72          62        54
H_tr [W/K]      310         185       118
H_tr_light      58          36        24
mdot_hp [kg/s]  0.30        0.27      0.24
T_offset [K]    10          -2        -7

REFERENCE:
- Building documentation: Virchowstr. 6, Langenhagen
- TABULA webtool: German MFH 1968 archetype
- Heating load: 10.968 kW at ΔT = 20K
"""


if __name__ == '__main__':
    # Print comparison
    print("=" * 70)
    print("BUILDING VARIANTS COMPARISON")
    print("=" * 70)
    print()

    for variant in ['original', 'enev', 'kfw']:
        params = get_building(variant)
        print(f"{params['name']}:")
        print(f"  H_ve:      {params['H_ve']} W/K")
        print(f"  H_tr:      {params['H_tr']} W/K")
        print(f"  H_tr_light: {params['H_tr_light']} W/K")
        print(f"  mdot_hp:   {params['mdot_hp']} kg/s")
        print(f"  T_offset:  {params['T_offset']} K")
        print()
