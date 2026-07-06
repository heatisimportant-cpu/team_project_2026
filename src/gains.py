import glob
import os
import numpy as np
import pandas as pd

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

    if area_floor is None:
        raise ValueError("area_floor is required (e.g. vonovia_model['area_floor']).")

    if solar_csv_path is not None:
        solar_raw = load_solar_gains(solar_csv_path, target_tz=target_tz)
    else:
        solar_raw = load_all_solar_gains(profiles_dir, target_tz=target_tz)

    if internal_csv_path is None:
        internal_csv_path = os.path.join(profiles_dir, DEFAULT_INTERNAL_GAINS_FILENAME)

    if index.tz is None:
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