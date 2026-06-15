import math
from .vonovia_model import BUILDING_MODELS as SFH_MODELS

from .mfh_1919_1948 import mfh_1919_1948_0_soc, mfh_1919_1948_1_enev, mfh_1919_1948_2_kfw
from .mfh_1949_1957 import mfh_1949_1957_0_soc, mfh_1949_1957_1_enev, mfh_1949_1957_2_kfw
from .mfh_1969_1978 import mfh_1969_1978_0_soc, mfh_1969_1978_1_enev, mfh_1969_1978_2_kfw

MFH_MODELS = [
    mfh_1919_1948_0_soc, mfh_1919_1948_1_enev, mfh_1919_1948_2_kfw,
    mfh_1949_1957_0_soc, mfh_1949_1957_1_enev, mfh_1949_1957_2_kfw,
    mfh_1969_1978_0_soc, mfh_1969_1978_1_enev, mfh_1969_1978_2_kfw,
]

def calculate_required_pumps(bldg_dict, is_mfh=False):
    """
    Calculates the required number of 13kW cascaded heat pumps based on peak heat loss.
    Peak heat loss is evaluated at -10°C ambient (ΔT = 30K from 20°C room temp).
    """
    area = bldg_dict.get('area_floor', 120.0)
    
    if is_mfh:
        # MFH files define heat loss as W/(m²K), but the physics simulator expects absolute W/K
        H_ve_abs = bldg_dict['H_ve'] * area
        H_tr_abs = bldg_dict['H_tr'] * area
        
        # Fix the building dict so the Simulator uses the correct absolute values!
        bldg_dict['H_ve'] = H_ve_abs
        bldg_dict['H_tr'] = H_tr_abs
    else:
        # SFH files already use absolute W/K
        H_ve_abs = bldg_dict['H_ve']
        H_tr_abs = bldg_dict['H_tr']
        
    peak_heat_loss_W = (H_ve_abs + H_tr_abs) * 30.0
    
    # One iDM AERO ALM 4-12 pump provides ~13kW at peak
    pump_capacity_W = 13000.0
    
    # Calculate required pumps (always at least 1)
    num_pumps = max(1, math.ceil(peak_heat_loss_W / pump_capacity_W))
    
    # Also dynamically scale the water flow rate
    bldg_dict['mdot_hp'] = 0.27 * num_pumps
    
    return num_pumps

# Set type identifier and dynamically calculate required pumps
for sfh in SFH_MODELS:
    sfh['type'] = 'sfh'
    sfh['num_pumps'] = calculate_required_pumps(sfh, is_mfh=False)

for mfh in MFH_MODELS:
    mfh['type'] = 'mfh'
    mfh['num_pumps'] = calculate_required_pumps(mfh, is_mfh=True)

ALL_BUILDING_MODELS = SFH_MODELS + MFH_MODELS
