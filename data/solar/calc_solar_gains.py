"""
Solar Gains Calculation for MFH 1968 Virchowstr. 6

Processes 10-minute solar irradiance data from DWD station Hannover (ID 2559)
and calculates hourly solar heat gains through building windows.

Input  (measured):  data/solar/hannover_solar_2024_2026.csv  (10-min, Nov 2024 – May 2026)
Input  (synthetic): generated internally for Jan 2024 – Nov 2024
Output:             data/solar/solar_gains_hourly.csv         (hourly, Jan 2024 – May 2026)

Running this script also updates data/profiles/internal_gains.csv in-place:
  - Q_sol       column is replaced with the newly computed values
  - Q_int_total is recalculated as Q_int_specific * AREA_FLOOR + Q_sol

=============================================================================
Unit fix
=============================================================================
The GHI_W_m2 / DHI_W_m2 columns in the source CSV were derived with a 1-min
window instead of the actual 10-min window, introducing a ×10 error:

  Correct:   GHI [W/m²] = global_radiation_J_cm2 × 10 000 / 600  = × 16.667
  As stored: GHI [W/m²] = global_radiation_J_cm2 × 10 000 /  60  = × 166.67

This script recomputes both fields from the raw J/cm² columns.

=============================================================================
Solar gain model (isotropic sky, vertical windows)
=============================================================================
For each vertical window surface:

  I_total  = I_beam + I_diff_sky + I_ground_reflected
  I_beam   = DNI_capped × max(0, cos(AOI))
             where DNI = (GHI − DHI) / sin(elevation), capped at 1000 W/m²
             and cos(AOI) = cos(elev) × cos(azimuth_sun − azimuth_surface)
  I_diff   = DHI × 0.5           (half-sky view factor, vertical)
  I_refl   = GHI × albedo × 0.5  (ground view factor, vertical)

Only computed for solar elevation > 5° (avoids extreme near-horizon DNI
amplification; also removes any measurement artefacts at the horizon).

The solar heat gain through each window:
  Q_win [W] = I_total [W/m²] × A_window [m²] × (1 − c_frame) × g_value

Building parameters (Virchowstr. 6, Langenhagen):
  Location : lat 52.44 °N, lon 9.74 °E
  Windows  : East 12 m², South 18 m², West 12 m², North 9 m²  (c_frame 0.30)
  g-value  : 0.60  (EnEV double glazing)
  Albedo   : 0.20

=============================================================================
Synthetic data for Jan – Nov 2024
=============================================================================
The DWD series starts 2024-11-28.  For the preceding period a synthetic profile
is constructed using:
  - Astronomical solar position (same model as for measured data)
  - Clear-sky GHI ≈ 950 × sin(elevation)
  - Monthly clearness index kt and diffuse fraction fd for Hannover
"""

import os
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT    = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
SOLAR_CSV  = os.path.join(THIS_DIR, 'hannover_solar_2024_2026.csv')
GAINS_CSV  = os.path.join(PROJECT, 'data', 'profiles', 'internal_gains.csv')
OUTPUT_CSV = os.path.join(THIS_DIR, 'solar_gains_hourly.csv')

# ---------------------------------------------------------------------------
# Building / location constants
# ---------------------------------------------------------------------------
LAT_DEG  = 52.44   # latitude  [°N]
LON_DEG  =  9.74   # longitude [°E]
ALT_KM   =  0.055  # elevation above sea level [km]

WINDOWS = {
    'east' : {'area': 12.0, 'azimuth':  90, 'c_frame': 0.30},
    'south': {'area': 18.0, 'azimuth': 180, 'c_frame': 0.30},
    'west' : {'area': 12.0, 'azimuth': 270, 'c_frame': 0.30},
    'north': {'area':  9.0, 'azimuth':   0, 'c_frame': 0.30},
}

G_VALUE        = 0.60   # solar heat gain coefficient (double-glazing, EnEV)
ALBEDO         = 0.20   # ground reflectance
ELEV_MIN_DEG   = 5.0    # minimum solar elevation for beam computation [°]
DNI_MAX        = 1000.0 # hard cap on DNI to prevent horizon artefacts [W/m²]

AREA_FLOOR = 287.92  # [m²] – used when refreshing Q_int_total

# Monthly clearness index and diffuse fraction for Hannover
MONTHLY_KT = {1: 0.25, 2: 0.30, 3: 0.35, 4: 0.43, 5: 0.48,
              6: 0.50, 7: 0.49, 8: 0.49, 9: 0.42, 10: 0.35,
              11: 0.28, 12: 0.22}
MONTHLY_FD = {1: 0.78, 2: 0.68, 3: 0.62, 4: 0.57, 5: 0.52,
              6: 0.47, 7: 0.47, 8: 0.52, 9: 0.57, 10: 0.66,
              11: 0.76, 12: 0.82}


# ===========================================================================
# Solar position
# ===========================================================================

def solar_position(timestamps, lat_deg: float, lon_deg: float):
    """
    Solar elevation [°] and azimuth [°] for an array of timestamps (UTC assumed).
    Azimuth: 0° = North, 90° = East, 180° = South, 270° = West.

    For hourly averages the solar position is evaluated at the hour mid-point
    (timestamp + 30 min) to best represent the mean radiation for that hour.
    """
    ts  = pd.DatetimeIndex(timestamps)
    # Shift to hour midpoint for more representative solar position
    ts_mid = ts + pd.Timedelta(minutes=30)

    lat  = np.radians(lat_deg)
    doy  = ts_mid.day_of_year.values.astype(float)

    B    = np.radians((360.0 / 365.0) * (doy - 81.0))
    decl = np.radians(23.45 * np.sin(B))

    EoT  = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)  # minutes

    utc_h = ts_mid.hour.values + ts_mid.minute.values / 60.0
    LST   = utc_h + lon_deg / 15.0 + EoT / 60.0
    ha    = np.radians(15.0 * (LST - 12.0))

    sin_elev = np.clip(
        np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha),
        -1.0, 1.0
    )
    elevation_deg = np.degrees(np.arcsin(sin_elev))

    cos_zen   = np.sqrt(np.maximum(0.0, 1.0 - sin_elev ** 2))
    denom     = np.where(cos_zen > 1e-6, cos_zen * np.cos(lat), 1e-6)
    # Correct formula: cos(az_from_South) = (sin(elev)*sin(lat) - sin(decl)) / (cos(elev)*cos(lat))
    cos_az_S  = np.clip((sin_elev * np.sin(lat) - np.sin(decl)) / denom, -1.0, 1.0)
    az_from_S = np.degrees(np.arccos(cos_az_S))          # 0–180° from South

    afternoon     = (LST % 24.0) > 12.0
    az_from_N     = np.where(afternoon, 180.0 + az_from_S, 180.0 - az_from_S)
    az_from_N     = az_from_N % 360.0

    return elevation_deg, az_from_N


# ===========================================================================
# Irradiance on vertical window surface
# ===========================================================================

def irradiance_on_vertical(GHI, DHI, elevation_deg, azimuth_sun_deg,
                            azimuth_surf_deg, albedo=ALBEDO,
                            elev_min=ELEV_MIN_DEG, dni_max=DNI_MAX):
    """
    Irradiance on a vertical surface using the isotropic sky model [W/m²].

    Beam component is computed only for solar elevation > elev_min to avoid
    near-horizon amplification artefacts.  DNI is capped at dni_max.
    """
    elev_rad = np.radians(np.clip(elevation_deg, 0.0, 90.0))
    az_sun   = np.radians(azimuth_sun_deg)
    az_surf  = np.radians(azimuth_surf_deg)

    sun_high = elevation_deg > elev_min          # beam only when sun is sufficiently up

    sin_elev = np.sin(elev_rad)
    DNI = np.where(
        sun_high & (sin_elev > 0.09),            # sin(5°) ≈ 0.087
        np.clip((GHI - DHI) / np.maximum(sin_elev, 0.09), 0.0, dni_max),
        0.0,
    )

    # Angle of incidence on vertical surface
    cos_AOI = np.cos(elev_rad) * np.cos(az_sun - az_surf)
    cos_AOI = np.maximum(cos_AOI, 0.0)

    I_beam = np.where(sun_high, DNI * cos_AOI, 0.0)
    I_diff = DHI * 0.5
    I_refl = GHI * albedo * 0.5

    return np.maximum(0.0, I_beam + I_diff + I_refl)


# ===========================================================================
# Building solar gains
# ===========================================================================

def calc_solar_gains_for_building(hourly_df, windows=WINDOWS, g_value=G_VALUE):
    """
    Total solar heat gains through all building windows [W].

    Parameters
    ----------
    hourly_df : pd.DataFrame
        Columns required: GHI_W_m2, DHI_W_m2, elevation_deg, azimuth_deg
    """
    GHI       = hourly_df['GHI_W_m2'].fillna(0.0).values
    DHI       = hourly_df['DHI_W_m2'].fillna(0.0).values
    elevation = hourly_df['elevation_deg'].values
    azimuth   = hourly_df['azimuth_deg'].values

    Q_sol = np.zeros(len(hourly_df))
    for _orient, params in windows.items():
        I_surf    = irradiance_on_vertical(GHI, DHI, elevation, azimuth, params['azimuth'])
        A_glazing = params['area'] * (1.0 - params['c_frame'])
        Q_sol    += I_surf * A_glazing * g_value

    return Q_sol


# ===========================================================================
# Synthetic clear-sky model (pre-measurement period)
# ===========================================================================

def synthetic_ghi_dhi(timestamps, lat_deg, lon_deg):
    """
    Estimate GHI and DHI for periods without measured data.

    Clear-sky model  : GHI_cs ≈ 950 × sin(elevation) for elevation > 0
    Cloud correction : GHI = GHI_cs × kt(month)
    Diffuse fraction : DHI = GHI × fd(month)
    """
    elevation_deg, _ = solar_position(timestamps, lat_deg, lon_deg)

    ts     = pd.DatetimeIndex(timestamps)
    months = ts.month.values
    kt     = np.array([MONTHLY_KT[int(m)] for m in months])
    fd     = np.array([MONTHLY_FD[int(m)] for m in months])

    elev_clipped = np.clip(elevation_deg, 0.0, None)
    GHI_cs = 950.0 * np.sin(np.radians(elev_clipped))
    GHI    = np.maximum(0.0, GHI_cs * kt)
    DHI    = GHI * fd

    return GHI, DHI


# ===========================================================================
# Data loading / resampling
# ===========================================================================

def load_and_fix_measured(path):
    """
    Load measured DWD CSV, fix the ×10 unit error in GHI_W_m2 / DHI_W_m2,
    return a clean 10-minute DataFrame.
    """
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.rename(columns={'datetime': 'timestamp'})
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Recompute from raw J/cm² columns using the correct 10-min interval
    FACTOR = 10_000.0 / 600.0   # 16.6667 W/m² per J/cm²
    df['GHI_W_m2'] = np.maximum(0.0, df['global_radiation_J_cm2'] * FACTOR)
    df['DHI_W_m2'] = np.maximum(0.0, df['diffuse_radiation_J_cm2'] * FACTOR)

    return df[['timestamp', 'GHI_W_m2', 'DHI_W_m2']]


def resample_to_hourly(df_10min):
    """
    Average 10-minute W/m² readings to hourly means.
    NaN gaps (rare) are forward-filled within the resampled output.
    """
    df_10min = df_10min.set_index('timestamp')
    hourly   = df_10min[['GHI_W_m2', 'DHI_W_m2']].resample('1h').mean()
    # Fill any NaN gaps created by missing 10-min rows (e.g. transmission errors)
    hourly   = hourly.fillna(0.0)
    return hourly.reset_index()


def build_full_hourly_solar(start='2024-01-01', end='2026-05-30 23:00'):
    """
    Assemble a complete hourly irradiance series covering start → end.
    - Measured DWD data where available (2024-11-28 onwards)
    - Synthetic (clear-sky × monthly kt) for the earlier period
    """
    df_meas_h      = resample_to_hourly(load_and_fix_measured(SOLAR_CSV))
    measured_start = df_meas_h['timestamp'].min()

    synth_range = pd.date_range(start=start,
                                end=measured_start - pd.Timedelta(hours=1),
                                freq='1h')
    if len(synth_range) > 0:
        GHI_s, DHI_s = synthetic_ghi_dhi(synth_range, LAT_DEG, LON_DEG)
        df_synth = pd.DataFrame({'timestamp': synth_range,
                                 'GHI_W_m2': GHI_s, 'DHI_W_m2': DHI_s})
    else:
        df_synth = pd.DataFrame(columns=['timestamp', 'GHI_W_m2', 'DHI_W_m2'])

    df_all = pd.concat([df_synth, df_meas_h], ignore_index=True)
    df_all = df_all[df_all['timestamp'] <= pd.to_datetime(end)].copy()
    df_all = df_all.sort_values('timestamp').reset_index(drop=True)
    return df_all


# ===========================================================================
# Main processing pipeline
# ===========================================================================

def process():
    """Compute solar gains, save output CSV, and update internal_gains.csv."""

    print("=" * 65)
    print("  Solar Gains Calculation – MFH 1968 Virchowstr. 6")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Irradiance series
    # ------------------------------------------------------------------
    print("\n[1/4] Building hourly solar irradiance series …")
    df_irr = build_full_hourly_solar()
    n_meas = (df_irr['timestamp'] >= '2024-11-28').sum()
    n_synt = len(df_irr) - n_meas
    print(f"      Coverage     : {df_irr['timestamp'].min()}  →  {df_irr['timestamp'].max()}")
    print(f"      Rows         : {len(df_irr):,}  ({n_synt:,} synthetic + {n_meas:,} measured)")
    print(f"      GHI max      : {df_irr['GHI_W_m2'].max():.1f}  W/m²")
    print(f"      GHI mean (day): {df_irr.loc[df_irr['GHI_W_m2']>1,'GHI_W_m2'].mean():.1f}  W/m²")

    # ------------------------------------------------------------------
    # 2. Solar position
    # ------------------------------------------------------------------
    print("\n[2/4] Calculating solar position (hour midpoints) …")
    elev, azim = solar_position(df_irr['timestamp'], LAT_DEG, LON_DEG)
    df_irr['elevation_deg'] = elev
    df_irr['azimuth_deg']   = azim

    # ------------------------------------------------------------------
    # 3. Solar gains
    # ------------------------------------------------------------------
    print("\n[3/4] Calculating window solar heat gains …")
    Q_sol = calc_solar_gains_for_building(df_irr)
    df_irr['Q_sol_W'] = Q_sol

    daytime = Q_sol[Q_sol > 1.0]
    print(f"      Q_sol max    : {Q_sol.max():.0f}  W")
    print(f"      Q_sol mean   : {daytime.mean():.0f}  W  (daytime only)")
    q_annual_kwh = Q_sol.sum() / 1000.0
    print(f"      Total (period): {q_annual_kwh:,.0f}  kWh")

    # ------------------------------------------------------------------
    # 4a. Save solar gains CSV
    # ------------------------------------------------------------------
    out_cols = ['timestamp', 'GHI_W_m2', 'DHI_W_m2',
                'elevation_deg', 'azimuth_deg', 'Q_sol_W']
    df_irr[out_cols].to_csv(OUTPUT_CSV, index=False, float_format='%.3f')
    print(f"\n[4/4] Saved : {os.path.relpath(OUTPUT_CSV, PROJECT)}")

    # ------------------------------------------------------------------
    # 4b. Update internal_gains.csv
    # ------------------------------------------------------------------
    print(f"      Updating: {os.path.relpath(GAINS_CSV, PROJECT)}")
    df_gains = pd.read_csv(GAINS_CSV)
    df_gains['timestamp'] = pd.to_datetime(df_gains['timestamp'])

    solar_lookup           = df_irr.set_index('timestamp')['Q_sol_W']
    df_gains['Q_sol']      = df_gains['timestamp'].map(solar_lookup).fillna(0.0).clip(lower=0.0)
    df_gains['Q_int_total'] = df_gains['Q_int_specific'] * AREA_FLOOR + df_gains['Q_sol']

    df_gains.to_csv(GAINS_CSV, index=False, float_format='%.4f')
    print(f"      Q_sol  range  : {df_gains['Q_sol'].min():.0f} – {df_gains['Q_sol'].max():.0f}  W")
    print(f"      Q_int_total max: {df_gains['Q_int_total'].max():.0f}  W")

    print("\n✅  Done – internal_gains.csv updated with measured + synthetic solar gains.")
    return df_irr


if __name__ == '__main__':
    sys.path.insert(0, PROJECT)
    process()
