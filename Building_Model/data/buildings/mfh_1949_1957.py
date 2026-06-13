'''
Multi-family house (MFH)
Construction period: 1949–1978
Location: Hannover, Germany
TABULA dataset: DE.N.MFH.05.Gen.ReEx.001.xxx
'''

# ------------------------------------------------------------
# State of construction
# TABULA source DE.N.MFH.05.Gen.ReEx.001.001
# ------------------------------------------------------------

mfh_1949_1978_0_soc = {
    'H_ve': 330,                 # [W/K] ventilation
    'H_tr': 1860,                # [W/K] transmission
    'H_tr_light': 210 + 12,      # [W/K] internal/light components
    'c_bldg': 55,                # [Wh/m²K] medium-heavy construction
    'area_floor': 820,           # [m²] typical MFH floor area
    'height_room': 2.5,          # [m]
    'name': 'mfh_1949_1978_0_soc'
}

# Heating system properties
mfh_1949_1978_0_soc['T_offset'] = 12
mfh_1949_1978_0_soc['T_amb_lim'] = 20
mfh_1949_1978_0_soc['mdot_hp'] = 0.45

# Window properties (old windows)
w_east  = {'area': 18, 'tilt': 90, 'azimuth': 90,  'g_value': 0.75, 'c_frame': 0.25, 'c_shade': 0.6}
w_south = {'area': 32, 'tilt': 90, 'azimuth': 180, 'g_value': 0.75, 'c_frame': 0.25, 'c_shade': 0.6}
w_west  = {'area': 18, 'tilt': 90, 'azimuth': 270, 'g_value': 0.75, 'c_frame': 0.25, 'c_shade': 0.6}
w_north = {'area': 22, 'tilt': 90, 'azimuth': 0,   'g_value': 0.75, 'c_frame': 0.25, 'c_shade': 0.6}

mfh_1949_1978_0_soc['windows'] = [w_east, w_south, w_west, w_north]

# Geolocation
mfh_1949_1978_0_soc['position'] = {
    'lat': 52.3759,
    'long': 9.7320,
    'altitude': 55,
    'timezone': 'Europe/Berlin'
}


# ------------------------------------------------------------
# Standard refurbishment (EnEV 2009/2014/2016)
# TABULA source DE.N.MFH.05.Gen.ReEx.001.002
# ------------------------------------------------------------

mfh_1949_1978_1_enev = {
    'H_ve': 330,
    'H_tr': 720,
    'H_tr_light': 110 + 6,
    'c_bldg': 55,
    'area_floor': 820,
    'height_room': 2.5,
    'name': 'mfh_1949_1978_1_enev'
}

mfh_1949_1978_1_enev['T_offset'] = -3
mfh_1949_1978_1_enev['T_amb_lim'] = 20
mfh_1949_1978_1_enev['mdot_hp'] = 0.40

# Refurbished windows
w_east  = {'area': 18, 'tilt': 90, 'azimuth': 90,  'g_value': 0.60, 'c_frame': 0.25, 'c_shade': 0.6}
w_south = {'area': 32, 'tilt': 90, 'azimuth': 180, 'g_value': 0.60, 'c_frame': 0.25, 'c_shade': 0.6}
w_west  = {'area': 18, 'tilt': 90, 'azimuth': 270, 'g_value': 0.60, 'c_frame': 0.25, 'c_shade': 0.6}
w_north = {'area': 22, 'tilt': 90, 'azimuth': 0,   'g_value': 0.60, 'c_frame': 0.25, 'c_shade': 0.6}

mfh_1949_1978_1_enev['windows'] = [w_east, w_south, w_west, w_north]

mfh_1949_1978_1_enev['position'] = {
    'lat': 52.3759,
    'long': 9.7320,
    'altitude': 55,
    'timezone': 'Europe/Berlin'
}


# ------------------------------------------------------------
# KfW refurbishment
# TABULA source DE.N.MFH.05.Gen.ReEx.001.003
# ------------------------------------------------------------

mfh_1949_1978_2_kfw = {
    'H_ve': 280,
    'H_tr': 380,
    'H_tr_light': 70 + 4,
    'c_bldg': 55,
    'area_floor': 820,
    'height_room': 2.5,
    'name': 'mfh_1949_1978_2_kfw'
}

mfh_1949_1978_2_kfw['T_offset'] = -8
mfh_1949_1978_2_kfw['T_amb_lim'] = 20
mfh_1949_1978_2_kfw['mdot_hp'] = 0.38

# High-efficiency windows
w_east  = {'area': 18, 'tilt': 90, 'azimuth': 90,  'g_value': 0.50, 'c_frame': 0.25, 'c_shade': 0.6}
w_south = {'area': 32, 'tilt': 90, 'azimuth': 180, 'g_value': 0.50, 'c_frame': 0.25, 'c_shade': 0.6}
w_west  = {'area': 18, 'tilt': 90, 'azimuth': 270, 'g_value': 0.50, 'c_frame': 0.25, 'c_shade': 0.6}
w_north = {'area': 22, 'tilt': 90, 'azimuth': 0,   'g_value': 0.50, 'c_frame': 0.25, 'c_shade': 0.6}

mfh_1949_1978_2_kfw['windows'] = [w_east, w_south, w_west, w_north]

mfh_1949_1978_2_kfw['position'] = {
    'lat': 52.3759,
    'long': 9.7320,
    'altitude': 55,
    'timezone': 'Europe/Berlin'
}
