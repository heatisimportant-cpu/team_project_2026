"""
Complete test suite for all models and controllers
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from src.models.building import Building
from src.models.heatpump import HeatPump
from src.models.tank import BufferTank
from src.models.integrated import IntegratedSystem
from src.controllers.heatcurve import HeatCurveController
from src.controllers.pid import PIDController
from src.controllers.mpc import MPCController
from data.buildings.mfh_1968 import get_building


class TestSuite:
    """Complete test suite"""

    def __init__(self):
        self.passed = 0
        self.failed = 0

    def assert_true(self, condition, test_name):
        """Assert a condition is true"""
        if condition:
            print(f"   {test_name} PASSED")
            self.passed += 1
        else:
            print(f"   {test_name} FAILED")
            self.failed += 1

    def print_summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY: {self.passed}/{total} passed")
        if self.failed == 0:
            print(" ALL TESTS PASSED!")
        else:
            print(f" {self.failed} tests failed")
        print(f"{'='*60}")


def test_building(suite):
    """Test building model"""
    print("\nTesting Building Model...")

    params = get_building()
    building = Building(params, mdot_hp=0.27, verbose=False)

    # Test initialization
    suite.assert_true(building.params['area_floor'] == 287.92, 
                     "Building floor area")
    suite.assert_true(building.params['H_tr'] == 185.0, 
                     "Building H_tr")
    suite.assert_true(building.params['H_ve'] == 62.0, 
                     "Building H_ve")

    # Test calculated parameters
    suite.assert_true('H_rad_con' in building.params, 
                     "H_rad_con calculated")
    suite.assert_true('C_zone' in building.params, 
                     "C_zone calculated")
    suite.assert_true('C_wall' in building.params, 
                     "C_wall calculated")
    suite.assert_true('C_water' in building.params, 
                     "C_water calculated")

    # Test simulation
    state = np.array([20.0, 20.0, 35.0])
    T_supply = 40.0
    T_amb = 5.0
    Q_gains = 800.0

    rhs = building.calc_4r3c(0, state, T_supply, [T_amb, Q_gains])

    suite.assert_true(len(rhs) == 3, 
                     "Building RHS length")
    suite.assert_true(not np.isnan(rhs).any(), 
                     "Building RHS no NaN")
    suite.assert_true(not np.isinf(rhs).any(), 
                     "Building RHS no Inf")

    # Test comfort calculation
    T_room_series = np.array([20, 21, 22, 23, 24])
    comfort = building.calc_comfort_dev(T_room_series, 3600)
    suite.assert_true('dev_neg_sum' in comfort, 
                     "Comfort dev_neg_sum")
    suite.assert_true('dev_pos_sum' in comfort, 
                     "Comfort dev_pos_sum")


def test_heatpump(suite):
    """Test heat pump model"""
    print("\nTesting Heat Pump Model...")

    hp = HeatPump(P_nom=12000, mdot_nom=0.27)

    # Test initialization
    suite.assert_true(hp.P_nom == 12000, 
                     "Heat pump nominal power")
    suite.assert_true(hp.mdot_nom == 0.27, 
                     "Heat pump mass flow")

    # Test COP calculation
    COP = hp.calc_COP(T_amb=5.0, T_supply=40.0)
    suite.assert_true(2.0 <= COP <= 6.0, 
                     "Heat pump COP in range")
    suite.assert_true(not np.isnan(COP), 
                     "Heat pump COP no NaN")

    # Test power calculation
    P_el, Q_heat, COP = hp.calc_power(T_amb=5.0, T_supply=40.0, T_return=35.0)
    suite.assert_true(P_el >= 0, 
                     "Heat pump P_el positive")
    suite.assert_true(Q_heat >= 0, 
                     "Heat pump Q_heat positive")
    suite.assert_true(COP > 0, 
                     "Heat pump COP positive")

    # Test modulation
    modulation, Q_actual = hp.calc_modulation(Q_demand=8000)
    suite.assert_true(0 <= modulation <= 1, 
                     "Heat pump modulation in range")
    suite.assert_true(Q_actual >= hp.P_min, 
                     "Heat pump Q_actual >= P_min")


def test_tank(suite):
    """Test buffer tank model"""
    print("\nTesting Buffer Tank Model...")

    tank = BufferTank(volume=500, n_zones=3)

    # Test initialization
    suite.assert_true(tank.volume == 0.5, 
                     "Tank volume in m³")
    suite.assert_true(tank.capacity > 0, 
                     "Tank capacity positive")

    # Test temperature change
    dT_dt = tank.calc_dT_dt(T_tank=40.0, Q_in=5000, Q_out=3000, T_amb=20.0)
    suite.assert_true(not np.isnan(dT_dt), 
                     "Tank dT_dt no NaN")

    # Test step simulation
    T_new, Q_loss = tank.step(T_tank=40.0, Q_in=5000, Q_out=3000, dt=3600)
    suite.assert_true(5.0 <= T_new <= 90.0, 
                     "Tank temperature in physical range")
    suite.assert_true(Q_loss >= 0, 
                     "Tank heat loss positive")

    # Test stratification
    T_top, T_mid, T_bottom = tank.calc_stratification(T_tank=40.0, Q_in=5000, Q_out=3000)
    suite.assert_true(T_top >= T_mid >= T_bottom, 
                     "Tank stratification order")


def test_integrated(suite):
    """Test integrated system"""
    print("\nTesting Integrated System...")

    params = get_building()
    system = IntegratedSystem(params, mdot_hp=0.27, mdot_bldg=0.25)

    # Test initialization
    suite.assert_true(system.building is not None, 
                     "Integrated system has building")
    suite.assert_true(system.heatpump is not None, 
                     "Integrated system has heat pump")
    suite.assert_true(system.tank is not None, 
                     "Integrated system has tank")

    # Test step simulation
    state = np.array([20.0, 20.0, 35.0, 40.0])
    state_new, results = system.step(state, T_supply_hp=45.0, T_amb=5.0, 
                                     Q_gains=800.0, dt=3600)

    suite.assert_true(len(state_new) == 4, 
                     "Integrated state length")
    suite.assert_true(not np.isnan(state_new).any(), 
                     "Integrated state no NaN")
    suite.assert_true('T_room' in results, 
                     "Integrated results has T_room")
    suite.assert_true('COP' in results, 
                     "Integrated results has COP")


def test_controllers(suite):
    """Test all controllers"""
    print("\nTesting Controllers...")

    # Test Heat Curve
    hc = HeatCurveController(T_supply_nom=45, T_amb_design=-12, T_room_set=20)
    T_supply = hc.calc_supply_temp(T_amb=5.0)
    suite.assert_true(20 <= T_supply <= 50, 
                     "Heat curve supply in range")

    # Test PID
    pid = PIDController(Kp=2.0, Ki=0.1, Kd=0.05, setpoint=20.0)
    output = pid.update(measured_value=19.5, dt=3600)
    suite.assert_true(20 <= output <= 50, 
                     "PID output in range")

    pid.reset()
    suite.assert_true(pid.integral == 0.0, 
                     "PID reset works")

    # Test MPC
    params = get_building()
    building = Building(params, mdot_hp=0.27, verbose=False)
    mpc = MPCController(building, horizon=24)

    state_init = np.array([20.0, 20.0, 35.0])
    T_amb_forecast = np.ones(24) * 5.0
    Q_gains_forecast = np.ones(24) * 800.0
    price_forecast = np.ones(24) * 0.30

    T_supply_opt, cost = mpc.optimize(state_init, T_amb_forecast, 
                                      Q_gains_forecast, price_forecast)

    suite.assert_true(len(T_supply_opt) == 24, 
                     "MPC horizon length")
    suite.assert_true(np.all(T_supply_opt >= 25), 
                     "MPC supply >= min")
    suite.assert_true(np.all(T_supply_opt <= 50), 
                     "MPC supply <= max")
    suite.assert_true(cost > 0, 
                     "MPC cost positive")


def test_data_loading(suite):
    """Test data files can be loaded"""
    print("\nTesting Data Loading...")

    # Test building parameters
    params = get_building()
    suite.assert_true('area_floor' in params, 
                     "Building params has area_floor")
    suite.assert_true(params['area_floor'] > 0, 
                     "Building area_floor positive")

    # Test weather data
    weather_path = os.path.join(os.path.dirname(__file__), 
                                '../data/weather/hannover_2024_2026.csv')
    if os.path.exists(weather_path):
        weather = pd.read_csv(weather_path)
        suite.assert_true(len(weather) > 0, 
                         "Weather data not empty")
        suite.assert_true('T_amb' in weather.columns, 
                         "Weather has T_amb")

    # Test price data
    price_path = os.path.join(os.path.dirname(__file__), 
                              '../data/prices/electricity_prices_2024.csv')
    if os.path.exists(price_path):
        prices = pd.read_csv(price_path)
        suite.assert_true(len(prices) > 0, 
                         "Price data not empty")
        suite.assert_true('price_eur_kwh' in prices.columns, 
                         "Prices has price_eur_kwh")

    # Test gains data
    gains_path = os.path.join(os.path.dirname(__file__), 
                              '../data/profiles/internal_gains.csv')
    if os.path.exists(gains_path):
        gains = pd.read_csv(gains_path)
        suite.assert_true(len(gains) > 0, 
                         "Gains data not empty")
        suite.assert_true('Q_int_total' in gains.columns, 
                         "Gains has Q_int_total")


def main():
    """Run all tests"""
    print("=" * 60)
    print("COMPLETE TEST SUITE - VIRCHOWSTR. 6")
    print("=" * 60)

    suite = TestSuite()

    test_building(suite)
    test_heatpump(suite)
    test_tank(suite)
    test_integrated(suite)
    test_controllers(suite)
    test_data_loading(suite)

    suite.print_summary()

    return suite.failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
