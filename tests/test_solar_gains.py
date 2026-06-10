"""
Unit tests for the solar gains calculation module.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from data.solar.calc_solar_gains import (
    solar_position,
    irradiance_on_vertical,
    calc_solar_gains_for_building,
    WINDOWS, G_VALUE,
)


class TestSuite:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def assert_true(self, condition, name):
        if condition:
            print(f"   {name} PASSED")
            self.passed += 1
        else:
            print(f"   {name} FAILED")
            self.failed += 1

    def assert_close(self, a, b, tol, name):
        self.assert_true(abs(a - b) < tol, name)

    def print_summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*55}")
        print(f"TEST SUMMARY: {self.passed}/{total} passed")
        if self.failed == 0:
            print(" ALL TESTS PASSED!")
        else:
            print(f" {self.failed} tests failed")
        print(f"{'='*55}")


def test_solar_position(suite):
    print("\nTesting solar_position …")

    # Summer solstice noon UTC at lon=9.74° E → LST ≈ 12.65 h
    # Declination ≈ +23.45°, lat=52.44° → elevation should be ~61°
    ts = pd.DatetimeIndex(['2025-06-21 11:30:00'])   # ~midday local time
    elev, azim = solar_position(ts, lat_deg=52.44, lon_deg=9.74)
    suite.assert_true(elev[0] > 55.0, "Summer solstice elevation > 55°")
    # At 11:30 UTC, local solar time is ~12:15 → sun slightly past south, az slightly > 180°
    suite.assert_true(170 < azim[0] < 230, "Summer solstice azimuth near South (170–230°)")

    # Winter solstice noon
    ts_w = pd.DatetimeIndex(['2025-12-21 11:30:00'])
    elev_w, _ = solar_position(ts_w, lat_deg=52.44, lon_deg=9.74)
    suite.assert_true(elev_w[0] > 10.0, "Winter solstice elevation > 10°")
    suite.assert_true(elev_w[0] < 25.0, "Winter solstice elevation < 25°")

    # Midnight → sun below horizon
    ts_n = pd.DatetimeIndex(['2025-06-21 00:00:00'])
    elev_n, _ = solar_position(ts_n, lat_deg=52.44, lon_deg=9.74)
    suite.assert_true(elev_n[0] < 0.0, "Midnight elevation < 0°")


def test_irradiance_on_vertical(suite):
    print("\nTesting irradiance_on_vertical …")

    # Clear midday south window: high irradiance expected
    GHI = np.array([800.0])
    DHI = np.array([100.0])
    elev = np.array([40.0])
    azim_sun = np.array([180.0])  # sun due South

    I_south = irradiance_on_vertical(GHI, DHI, elev, azim_sun, 180.0)
    suite.assert_true(I_south[0] > 400, "South window midday irradiance > 400 W/m²")

    # North window, sun due South → beam = 0
    I_north = irradiance_on_vertical(GHI, DHI, elev, azim_sun, 0.0)
    suite.assert_true(I_north[0] < I_south[0],
                      "North window irradiance < South window irradiance")

    # Nighttime: with zero irradiance inputs → all zero
    elev_night = np.array([-5.0])
    I_night = irradiance_on_vertical(
        np.array([0.0]), np.array([0.0]), elev_night, azim_sun, 180.0
    )
    suite.assert_true(I_night[0] == 0.0, "Night irradiance is zero when GHI=DHI=0")

    # DNI cap: very low elevation should not produce extreme values
    GHI_low = np.array([150.0])
    DHI_low = np.array([20.0])
    elev_low = np.array([3.0])   # below ELEV_MIN_DEG → beam = 0
    I_low = irradiance_on_vertical(GHI_low, DHI_low, elev_low, azim_sun, 180.0)
    # Only diffuse + reflected: 20×0.5 + 150×0.2×0.5 = 10 + 15 = 25
    suite.assert_close(float(I_low[0]), 25.0, 5.0, "Low-elevation irradiance ≈ 25 W/m²")


def test_building_solar_gains(suite):
    print("\nTesting calc_solar_gains_for_building …")

    # Construct a simple hourly DataFrame for midday clear summer day
    df = pd.DataFrame({
        'GHI_W_m2':    [800.0, 0.0],
        'DHI_W_m2':    [100.0, 0.0],
        'elevation_deg': [45.0, -5.0],
        'azimuth_deg': [180.0, 180.0],
    })

    Q_sol = calc_solar_gains_for_building(df)

    # Midday: south window should deliver substantial heat gain
    suite.assert_true(Q_sol[0] > 2000, "Q_sol midday > 2000 W")

    # Midnight: no gains beyond very small diffuse/reflected (GHI=0 → all zero)
    suite.assert_true(Q_sol[1] == 0.0, "Q_sol midnight = 0 W")

    # Sanity: Q_sol never negative
    suite.assert_true(np.all(Q_sol >= 0), "Q_sol always non-negative")

    # Sanity: Q_sol not absurdly large (max ~10 kW for these windows)
    suite.assert_true(Q_sol[0] < 10000, "Q_sol midday < 10 000 W")


def test_internal_gains_csv(suite):
    print("\nTesting updated internal_gains.csv …")

    here    = os.path.dirname(os.path.abspath(__file__))
    project = os.path.abspath(os.path.join(here, '..'))
    path    = os.path.join(project, 'data', 'profiles', 'internal_gains.csv')

    df = pd.read_csv(path)
    suite.assert_true('Q_sol' in df.columns, "Q_sol column present")
    suite.assert_true('Q_int_total' in df.columns, "Q_int_total column present")
    suite.assert_true((df['Q_sol'] >= 0).all(), "Q_sol non-negative")

    # At least some daytime solar gains should be > 0
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    summer_day = df[(df['timestamp'].dt.month == 6) &
                    (df['timestamp'].dt.hour == 12)]
    suite.assert_true((summer_day['Q_sol'] > 0).any(),
                      "Summer noon Q_sol > 0 in at least some hours")

    # Q_int_total should equal Q_int_specific * 287.92 + Q_sol
    diff = (df['Q_int_total'] - (df['Q_int_specific'] * 287.92 + df['Q_sol'])).abs()
    suite.assert_true(diff.max() < 1.0, "Q_int_total = Q_int_specific×area + Q_sol")


def main():
    suite = TestSuite()
    test_solar_position(suite)
    test_irradiance_on_vertical(suite)
    test_building_solar_gains(suite)
    test_internal_gains_csv(suite)
    suite.print_summary()
    return suite.failed


if __name__ == '__main__':
    sys.exit(main())
