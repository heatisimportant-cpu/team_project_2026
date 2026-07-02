# -*- coding: utf-8 -*-
"""
Gains
=====
Builds the combined solar + internal heat gain disturbance series
(`Qdot_gains` [W]) for the building, aligned to a given timestamp index.

Two independent sources are combined:

  Solar gains
  -----------
  From a fixed, pre-cleaned CSV (see solar_gains_MFH_Vonovia_Ref_fixed.csv)
  with hourly Q_sol_W already transposed onto the building's real window
  geometry (see vonovia_model.py windows/g-value derivation notes). That
  file's timestamps are UTC; this module converts them to the target
  index's local time (DST-aware) before aligning, since the building's
  own weather/price data (e.g. train_2021_2023.csv) is in local time
  (confirmed by its DST gap pattern -- a missing 02:00 on the last
  Sunday of March, the standard Europe/Berlin "spring forward" signature).

  Internal gains
  --------------
  From an hourly occupancy/appliance profile (specific gains in W/m^2,
  separate workday vs weekend multipliers per hour), scaled by the
  building's floor area. Workday/weekend is derived from each target
  timestamp's actual day of week, not assumed.

Both are recomputed fresh from source each run (not cached to a merged
CSV), per project decision -- so changes to either source file, the
floor area, or the target index are always reflected without a stale
intermediate file to keep in sync.

IMPORTANT: Qdot_gains is a hidden simulation input only. It is NOT
added to the RL observation (see room_env.py) -- the agent cannot
observe or forecast solar/internal gains directly, by design, since
neither can be reliably predicted in real-time deployment. Instead the
agent is given cyclical time features (hour-of-day, day-of-year,
is_weekend) so it can learn the *statistical pattern* of when gains
tend to be high or low, without depending on knowing their exact value.
"""

import glob
import os
import numpy as np
import pandas as pd

# Fixed location for gain profile data, matching the project's data/profiles/
# folder convention. solar_*.csv files in here (e.g. solar_2021_2023.csv,
# solar_2024_onwards.csv) are auto-discovered and stitched together -- the
# caller doesn't need to know or specify which period file covers which
# date range; whatever combination of files is needed gets picked up
# automatically by reindexing against the target index later.
DEFAULT_PROFILES_DIR = 'data/profiles'
DEFAULT_INTERNAL_GAINS_FILENAME = 'internal_gains.csv'


def discover_solar_files(profiles_dir: str = DEFAULT_PROFILES_DIR) -> list:
    """Find all solar_*.csv files in profiles_dir, sorted by filename
    (so solar_2021_2023.csv loads before solar_2024_onwards.csv etc.)."""
    pattern = os.path.join(profiles_dir, 'solar_*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No solar_*.csv files found in '{profiles_dir}'. Expected e.g. "
            f"'{profiles_dir}/solar_2021_2023.csv', "
            f"'{profiles_dir}/solar_2024_onwards.csv'.")
    return files


def load_all_solar_gains(profiles_dir: str = DEFAULT_PROFILES_DIR,
                          target_tz: str = 'Europe/Berlin') -> pd.Series:
    """Load and concatenate every solar_*.csv found in profiles_dir into
    one continuous Q_sol_W series, converted from UTC to target_tz.

    Multiple period files (e.g. one per train/test split) are combined
    transparently -- whichever rows the caller's target index actually
    needs get pulled out later via reindexing, so the split point between
    files doesn't need to line up with any particular train/test boundary."""
    files = discover_solar_files(profiles_dir)
    parts = [load_solar_gains(f, target_tz=target_tz) for f in files]
    combined = pd.concat(parts)
    combined = combined[~combined.index.duplicated(keep='first')].sort_index()
    return combined


def load_solar_gains(path: str, target_tz: str = 'Europe/Berlin') -> pd.Series:
    """Load the cleaned solar gains CSV and convert from UTC to target_tz.

    Parameters
    ----------
    path : str
        Path to solar_gains_MFH_Vonovia_Ref_fixed.csv (or equivalent).
    target_tz : str
        Timezone to convert into, matching the weather/price data's
        timezone (default 'Europe/Berlin' -- see module docstring for
        how this was confirmed for this project's data).

    Returns
    -------
    pd.Series
        Q_sol_W indexed by tz-aware timestamp in target_tz.
    """
    df = pd.read_csv(path, parse_dates=['timestamp'])
    idx = pd.DatetimeIndex(df['timestamp']).tz_localize('UTC').tz_convert(target_tz)
    s = pd.Series(df['Q_sol_W'].values, index=idx, name='Q_sol_W')
    # DST fall-back creates a duplicate local hour; spring-forward creates a
    # gap. Drop duplicates (keep first) so reindex/interpolation downstream
    # behaves predictably; the gap is handled naturally by reindex+interpolate.
    s = s[~s.index.duplicated(keep='first')]
    return s.sort_index()


def compute_internal_gains(index: pd.DatetimeIndex, internal_csv_path: str,
                            area_floor: float) -> pd.Series:
    """Build hourly internal gains [W] for the building, aligned to `index`.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Target timestamps (e.g. the weather/price data's index). Each
        timestamp's actual hour-of-day and day-of-week are used --
        weekday/weekend is derived from the real calendar, not assumed.
    internal_csv_path : str
        Path to internal_gains.csv (semicolon-separated; hour, user/
        appliances specific gains in W/m^2, workday/weekend multipliers).
    area_floor : float
        Building floor area [m^2] (vonovia_model['area_floor']).

    Returns
    -------
    pd.Series
        Internal gains [W], indexed by `index`.
    """
    profile = pd.read_csv(internal_csv_path, sep=';')
    profile.columns = [c.strip() for c in profile.columns]
    profile = profile.set_index('hour')

    hour_of_day = index.hour
    is_weekend = index.dayofweek.isin([5, 6])   # Sat=5, Sun=6

    user_base = profile['user [W/m^2]'].reindex(hour_of_day).to_numpy()
    appl_base = profile['appliances [W/m^2]'].reindex(hour_of_day).to_numpy()
    wd_user   = profile['workday_user'].reindex(hour_of_day).to_numpy()
    wd_appl   = profile['workday_appliances'].reindex(hour_of_day).to_numpy()
    we_user   = profile['weekend_user'].reindex(hour_of_day).to_numpy()
    we_appl   = profile['weekend_appliances'].reindex(hour_of_day).to_numpy()

    user_factor = np.where(is_weekend, we_user, wd_user)
    appl_factor = np.where(is_weekend, we_appl, wd_appl)

    gain_W_m2 = user_base * user_factor + appl_base * appl_factor
    gain_W    = gain_W_m2 * area_floor

    return pd.Series(gain_W, index=index, name='Qdot_internal_W')


def build_gains_series(index: pd.DatetimeIndex,
                        profiles_dir: str = DEFAULT_PROFILES_DIR,
                        area_floor: float = None,
                        target_tz: str = 'Europe/Berlin',
                        solar_csv_path: str = None,
                        internal_csv_path: str = None) -> pd.Series:
    """Combined solar + internal Qdot_gains [W], aligned exactly to `index`.

    By default, solar data is auto-discovered from every solar_*.csv file
    in `profiles_dir` (stitched together -- see load_all_solar_gains), and
    internal gains from `profiles_dir`/internal_gains.csv. No explicit
    file paths are needed for the normal case of a fixed data/profiles/
    folder; pass solar_csv_path/internal_csv_path to override for one-off
    use (e.g. testing against a file outside the usual folder).

    Parameters
    ----------
    index : pd.DatetimeIndex
        The target disturbance series' index (e.g. T_amb/price data's
        index) that Qdot_gains must align to.
    profiles_dir : str
        Folder containing solar_*.csv file(s) and internal_gains.csv.
    area_floor : float
        Building floor area [m^2]. Required.
    target_tz : str
        Timezone `index` is in (for converting the UTC solar files).
    solar_csv_path, internal_csv_path : str, optional
        Override the auto-discovered paths with a specific single file.

    Returns
    -------
    pd.Series
        Qdot_gains [W], indexed exactly as `index` (same length, same
        timestamps), ready to assign as a 'Qdot_gains' column.
    """
    if area_floor is None:
        raise ValueError("area_floor is required (e.g. vonovia_model['area_floor']).")

    if solar_csv_path is not None:
        solar_raw = load_solar_gains(solar_csv_path, target_tz=target_tz)
    else:
        solar_raw = load_all_solar_gains(profiles_dir, target_tz=target_tz)

    if internal_csv_path is None:
        internal_csv_path = os.path.join(profiles_dir, DEFAULT_INTERNAL_GAINS_FILENAME)

    # `index` (e.g. from train_2021_2023.csv) is naive local civil time --
    # confirmed by its DST gap pattern (missing 02:00 on spring-forward
    # Sundays) -- while solar_raw is tz-aware after UTC->target_tz
    # conversion. Reindexing a naive index against a tz-aware series
    # matches NOTHING (different dtypes), which silently zeroed out solar
    # gains entirely in initial testing. Fix: localize a COPY of the
    # index for matching purposes only, then return values on the
    # ORIGINAL (naive) index so the result drops straight into the
    # caller's existing naive-indexed DataFrame.
    if index.tz is None:
        # nonexistent='shift_forward': spring-forward gap hours don't exist
        # in local time anyway (confirmed absent from the real data), so
        # this only matters if some other naive index ever includes one.
        # ambiguous=False: fall-back's repeated local hour (e.g. 31 Oct
        # 02:00) is treated as the post-transition (standard time)
        # occurrence -- a deterministic convention, since the real data
        # has no duplicate entry to infer the correct side from, and the
        # 1-hour uncertainty this introduces is negligible (1 hour out of
        # tens of thousands in the series).
        match_index = index.tz_localize(
            target_tz, nonexistent='shift_forward', ambiguous=False)
    else:
        match_index = index.tz_convert(target_tz)

    solar_aligned = solar_raw.reindex(match_index)
    solar_aligned.index = index   # restore original (naive) index for merging
    n_missing = solar_aligned.isna().sum()
    if n_missing > 0:
        solar_aligned = solar_aligned.interpolate(method='linear', limit=3)
        solar_aligned = solar_aligned.fillna(0.0)
        print(f"[gains] Note: {n_missing} timestamps in target index had no "
              f"exact solar data match; filled via interpolation.")

    internal = compute_internal_gains(index, internal_csv_path, area_floor)

    total = solar_aligned.to_numpy() + internal.to_numpy()
    return pd.DataFrame({
        'Qdot_gains': total,
        'Q_sol_W':    solar_aligned.to_numpy(),
    }, index=index)