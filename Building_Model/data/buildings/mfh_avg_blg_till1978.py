"""
Average building model for multi-family houses (MFH) built until 1978, based on the TABULA dataset for Germany. 
The representative building is DE.N.MFH-AB.01-06.Gen.SyAv.002.001, which is a typical multi-family houses with 
medium-heavy construction and old windows. The model includes properties such as ventilation and transmission heat 
losses, thermal mass, floor area, room height, window properties, and geolocation.

Simplified Multi-family House (MFH)

Construction period: until 1978

Dataset: German residential building stock, simplified (IWU/EPISCOPE 2009)

Representative building: DE.N.MFH-AB.01-06.Gen.SyAv.002.001

Nummber of apartments: 6.36
"""

# ------------------------------------------------------------
# State of construction
# TABULA source DE.N.MFH-AB.01-06.Gen.SyAv.002.001
# ------------------------------------------------------------

mfh_simplified = {
    'H_ve': 1.54,         # [W/m²K] floor area related heat transfer coefficient by ventilation
    'H_tr': 0.51,         # [W/m²K] floor area related heat transfer coefficient by transmission
    'H_tr_light': 10.73,  # [W/K] supplemental heat loss due to thermal bridging 
    'c_bldg': 51,         # [Wh/m²K] internal heat capacity per m² reference area for light-heavy construction
    'area_floor': 453.6,  # [m²] reference floor area (conditioned floor area, internal dimensions)
    'height_room': 2.5,   # [m]
    'name': 'mfh_simplified'
}

# Heating system properties
mfh_simplified['T_offset'] = 12
mfh_simplified['T_amb_lim'] = 20
mfh_simplified['mdot_hp'] = 0.45

# Window properties

w_east = {'area': 20.15, 'tilt': 90, 'azimuth': 90, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_south = {'area': 20.15, 'tilt': 90, 'azimuth': 180, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_west = {'area': 20.15, 'tilt': 90, 'azimuth': 270, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_north = {'area': 20.15, 'tilt': 90, 'azimuth': 0, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}

windows = [w_east, w_south, w_west, w_north]

mfh_simplified['windows'] = [w_east, w_south, w_west, w_north]

# Geolocation of Hannover, Germany
mfh_simplified['position'] = {
    'lat': 52.3759,
    'long': 9.7320,
    'altitude': 55,
    'timezone': 'Europe/Berlin'
}
