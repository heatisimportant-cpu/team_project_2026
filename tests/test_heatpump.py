"""Test heat pump model"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.heatpump import HeatPump
import numpy as np


def test_initialization():
    """Test heat pump initialization"""
    hp = HeatPump(P_nom=12000, mdot_nom=0.27)
    assert hp.P_nom == 12000
    assert hp.mdot_nom == 0.27
    print(" Heat pump initialization")


def test_cop_calculation():
    """Test COP calculation"""
    hp = HeatPump()

    # Test at different conditions
    COP1 = hp.calc_COP(T_amb=7.0, T_supply=35.0)
    COP2 = hp.calc_COP(T_amb=-7.0, T_supply=45.0)

    assert 2.0 <= COP1 <= 6.0
    assert 2.0 <= COP2 <= 6.0
    assert COP1 > COP2  # Higher COP at better conditions
    print(" COP calculation")


def test_power_calculation():
    """Test power and heat calculation"""
    hp = HeatPump()

    P_el, Q_heat, COP = hp.calc_power(T_amb=5.0, T_supply=40.0, T_return=35.0)

    assert P_el > 0
    assert Q_heat > 0
    assert COP > 0
    assert abs(Q_heat / P_el - COP) < 0.1  # Q = P * COP
    print(" Power calculation")


def test_modulation():
    """Test modulation calculation"""
    hp = HeatPump(P_nom=12000, P_min=4000)

    # Test at different demands
    mod1, Q1 = hp.calc_modulation(Q_demand=8000)
    mod2, Q2 = hp.calc_modulation(Q_demand=15000)  # Above max
    mod3, Q3 = hp.calc_modulation(Q_demand=2000)   # Below min

    assert 0 <= mod1 <= 1
    assert Q2 == hp.P_nom  # Clipped to max
    assert Q3 == hp.P_min  # Clipped to min
    print(" Modulation calculation")


if __name__ == '__main__':
    print("\nTesting Heat Pump Model...")
    print("=" * 40)
    test_initialization()
    test_cop_calculation()
    test_power_calculation()
    test_modulation()
    print("=" * 40)
    print("All heat pump tests passed!\n")
