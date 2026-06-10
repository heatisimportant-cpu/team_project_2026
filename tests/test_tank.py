"""Test buffer tank model"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.tank import BufferTank
import numpy as np


def test_initialization():
    """Test tank initialization"""
    tank = BufferTank(volume=500, n_zones=3)
    assert tank.volume == 0.5  # m³
    assert tank.capacity > 0
    print(" Tank initialization")


def test_temperature_change():
    """Test temperature change calculation"""
    tank = BufferTank(volume=500)

    # Heating: Q_in > Q_out
    dT_dt1 = tank.calc_dT_dt(T_tank=40.0, Q_in=5000, Q_out=3000, T_amb=20.0)
    assert dT_dt1 > 0  # Temperature increases

    # Cooling: Q_in < Q_out
    dT_dt2 = tank.calc_dT_dt(T_tank=40.0, Q_in=2000, Q_out=4000, T_amb=20.0)
    assert dT_dt2 < 0  # Temperature decreases

    print(" Temperature change calculation")


def test_step_simulation():
    """Test step simulation"""
    tank = BufferTank(volume=500)

    T_init = 40.0
    T_new, Q_loss = tank.step(T_tank=T_init, Q_in=5000, Q_out=3000, 
                              dt=3600, T_amb=20.0)

    assert 5.0 <= T_new <= 90.0  # Physical limits
    assert Q_loss >= 0  # Heat loss always positive
    assert T_new > T_init  # Net heating

    print(" Step simulation")


def test_stratification():
    """Test stratification calculation"""
    tank = BufferTank(volume=500, n_zones=3)

    T_top, T_mid, T_bottom = tank.calc_stratification(T_tank=40.0, 
                                                       Q_in=5000, Q_out=3000)

    assert T_top >= T_mid >= T_bottom  # Stratification order
    assert T_mid == 40.0  # Middle = average

    print(" Stratification calculation")


if __name__ == '__main__':
    print("\nTesting Buffer Tank Model...")
    print("=" * 40)
    test_initialization()
    test_temperature_change()
    test_step_simulation()
    test_stratification()
    print("=" * 40)
    print(" All tank tests passed!\n")
