import pandas as pd
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
        Absolute total internal gains [W], indexed by 'time'.
    """
    if isinstance(profile_path, (str, Path)):
        profile = pd.read_csv(profile_path, delimiter=';', index_col='hour')
    else:
        profile = profile_path

    df = pd.DataFrame(index=time)
    df['qdot_oc']  = 0.0
    df['qdot_app'] = 0.0

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

    # Total specific gains [W/m2] to [W]
    qdot_tot = df['qdot_oc'] + df['qdot_app']
    return (qdot_tot * bldg_area).rename('Qdot_internal')