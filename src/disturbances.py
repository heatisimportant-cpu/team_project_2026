# -*- coding: utf-8 -*-
"""
Internal Heat Gains
===================
Computes internal heat gains (occupancy + appliances) for a building,
following i4b's method based on DIN EN 16798-1.

The profile CSV (data/profiles/InternalGains/ResidentialDetached.csv) gives,
for each hour of the day, specific gains in W/m² with separate weekday/weekend
multipliers. These are scaled by floor area to get absolute watts.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def get_internal_gains(time, profile_path, bldg_area):
    """Compute absolute internal heat gains [W] over a time index.

    Parameters
    ----------
    time : pandas.DatetimeIndex
        Timestamps to compute gains for.
    profile_path : str or Path or pd.DataFrame
        Path to the internal-gains profile CSV (DIN EN 16798-1 format),
        or an already-loaded DataFrame.
    bldg_area : float
        Heated floor area [m²].

    Returns
    -------
    pandas.Series
        Absolute total internal gains [W], indexed by `time`.
    """
    if isinstance(profile_path, (str, Path)):
        profile = pd.read_csv(profile_path, delimiter=';', index_col='hour')
    else:
        profile = profile_path

    df = pd.DataFrame(index=time)
    df['qdot_oc']  = 0.0
    df['qdot_app'] = 0.0

    # Weekday (Mon-Fri) vs weekend (Sat-Sun) split
    mask = time.dayofweek < 5
    wd = time[mask]
    we = time[~mask]

    df.loc[wd, 'qdot_oc']  = (profile['user [W/m^2]'][wd.hour]
                              * profile['workday_user'][wd.hour]).values
    df.loc[wd, 'qdot_app'] = (profile['appliances [W/m^2]'][wd.hour]
                              * profile['workday_appliances'][wd.hour]).values
    df.loc[we, 'qdot_oc']  = (profile['user [W/m^2]'][we.hour]
                              * profile['weekend_user'][we.hour]).values
    df.loc[we, 'qdot_app'] = (profile['appliances [W/m^2]'][we.hour]
                              * profile['weekend_appliances'][we.hour]).values

    # Total specific gains [W/m²] → absolute [W]
    qdot_tot = df['qdot_oc'] + df['qdot_app']
    return (qdot_tot * bldg_area).rename('Qdot_internal')


# ── Solar gains (pvlib, following i4b get_solar_gains) ────────────────────────

# Map orientation names to azimuth angles (pvlib: 0°=N, 90°=E, 180°=S, 270°=W)
_ORIENTATION_AZIMUTH = {
    'north': 0.0,
    'east':  90.0,
    'south': 180.0,
    'west':  270.0,
}

# Standard glazing properties (typical double glazing)
_G_VALUE = 0.6     # solar heat gain coefficient (g-value) [-]
_C_SHADE = 1.0     # shading factor (1.0 = no external shading)


def get_solar_gains(weather, bldg_params, albedo=0.2):
    """Total solar heat gain through all windows [W], following i4b's method.

    Uses pvlib to project irradiance onto each oriented window. GHI is
    decomposed into DNI + DHI via the Erbs model since DWD provides only GHI.

    Parameters
    ----------
    weather : pandas.DataFrame
        DatetimeIndex (tz-aware) with column 'ghi_wm2' [W/m²].
    bldg_params : dict
        Building dict with 'windows' (orientation->area), 'tilt',
        'frame_fraction', and location ('latitude','longitude','altitude','timezone').
    albedo : float
        Ground reflectance (0.2 = grass).

    Returns
    -------
    pandas.Series
        Solar heat gains [W], indexed like `weather`.
    """
    import pvlib

    idx = weather.index
    ghi = weather['ghi_wm2'].clip(lower=0.0)

    location = pvlib.location.Location(
        latitude=bldg_params['latitude'],
        longitude=bldg_params['longitude'],
        tz=bldg_params['timezone'],
        altitude=bldg_params['altitude'],
    )
    solpos    = location.get_solarposition(idx)
    dni_extra = pvlib.irradiance.get_extra_radiation(idx)

    # Use measured DHI (DWD provides it); compute DNI from GHI, DHI, zenith.
    # At low sun angles cos(zenith)→0 makes naive DNI explode, so we:
    #   (1) only compute DNI when the sun is meaningfully above the horizon,
    #   (2) cap DNI at the extraterrestrial normal irradiance (physical max).
    dhi   = weather['dhi_wm2'].clip(lower=0.0)
    zen   = solpos['zenith']
    cos_z = np.cos(np.radians(zen))
    dni = ((ghi - dhi) / cos_z.where(cos_z > 0.0, np.nan)).clip(lower=0.0)
    dni = dni.fillna(0.0)
    # Sun below ~easn, treat direct beam as 0 (all radiation is diffuse)
    dni = dni.where(zen < 85.0, 0.0)
    # Physical cap: DNI cannot exceed the extraterrestrial normal irradiance
    dni = np.minimum(dni, dni_extra)

    tilt          = bldg_params.get('tilt', 90.0)
    frame_frac    = bldg_params.get('frame_fraction', 0.3)
    windows       = bldg_params['windows']

    Qdot_sol = pd.Series(0.0, index=idx)
    for orientation, area in windows.items():
        if area <= 0:
            continue
        azimuth = _ORIENTATION_AZIMUTH[orientation]

        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            solar_zenith=solpos['zenith'],
            solar_azimuth=solpos['azimuth'],
            dni=dni, ghi=ghi, dhi=dhi,
            dni_extra=dni_extra,
            albedo=albedo,
            model='isotropic',   # robust; 'perez' needs airmass
        )
        poa_global = poa['poa_global'].fillna(0.0)

        # Solar gain through this window [W]
        Qdot_sol += area * _G_VALUE * poa_global * (1 - frame_frac) * _C_SHADE

    return Qdot_sol.rename('Qdot_solar').clip(lower=0.0)