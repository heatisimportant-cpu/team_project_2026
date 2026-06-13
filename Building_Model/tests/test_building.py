"""
Test building model
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.building import Building
from data.buildings.mfh_1968 import get_building
import numpy as np


def test_building_initialization():
    """Test building can be initialized"""
    params = get_building()
    building = Building(params, mdot_hp=0.27, verbose=False)

    assert building.params['area_floor'] == 287.92
    assert building.params['H_tr'] == 185.0
    assert building.params['H_ve'] == 62.0
    print("Building initialization test PASSED")


def test_building_simulation():
    """Test building can simulate one step"""
    params = get_building()
    building = Building(params, mdot_hp=0.27, verbose=False)

    state = np.array([20.0, 20.0, 35.0])
    T_supply = 40.0
    T_amb = 5.0
    Q_gains = 800.0

    rhs = building.calc_4r3c(0, state, T_supply, [T_amb, Q_gains])

    assert len(rhs) == 3
    assert not np.isnan(rhs).any()
    assert not np.isinf(rhs).any()
    print("Building simulation test PASSED")


def test_building_parameters():
    """Test calculated parameters are correct"""
    params = get_building()
    building = Building(params, mdot_hp=0.27, verbose=False)

    # Check calculated parameters exist
    assert 'H_rad_con' in building.params
    assert 'C_zone' in building.params
    assert 'C_wall' in building.params
    assert 'C_water' in building.params

    # Check reasonable values
    assert building.params['H_rad_con'] > 0
    assert building.params['C_zone'] > 0
    assert building.params['C_wall'] > 0
    print("Building parameters test PASSED")


if __name__ == '__main__':
    print("=" * 60)
    print("TESTING BUILDING MODEL")
    print("=" * 60)
    print()

    test_building_initialization()
    test_building_simulation()
    test_building_parameters()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
