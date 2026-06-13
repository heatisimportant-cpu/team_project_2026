"""
Multi-family house (MFH)
Construction period: 1958–1968
Location: Hannover, Germany
Reference floor area: 350 m²
TABULA building: DE.N.MFH.05.Gen.ReEx.001
"""

# ------------------------------------------------------------
# Common data (TABULA)
# ------------------------------------------------------------

#Heating properities for original state (0_soc)
H_ve_1        = 0.51      # [W/(m²K)]
H_tr_1        = 1.74      # [W/(m²K)]
H_tr_light_1  = 449.07     # [W/K]

#Heating properities for standard refurbishment (1_enev)
H_ve_2        = 0.51      # [W/(m²K)]
H_tr_2        = 0.61      # [W/(m²K)]
H_tr_light_2  = 449.07     # [W/K]

#Heating properities for deep refurbishment (2_kfw)
H_ve_3        = 0.43      # [W/(m²K)]
H_tr_3        = 0.35      # [W/(m²K)]
H_tr_light_3  = 224.54     # [W/K]

c_bldg      = 45        # [Wh/m²K]
area_floor  = 2844.61       # [m²]
height_room = 2.5       # [m]

# Window areas [m²]
A_Window_East  = 22.245
A_Window_South = 243.24
A_Window_West  = 22.245
A_Window_North = 219.75

# Location: Hannover, Germany
position_hannover = {
    'lat': 52.3759,
    'long': 9.7320,
    'altitude': 55,
    'timezone': 'Europe/Berlin'
}

# ------------------------------------------------------------
# Heat pump data
# ------------------------------------------------------------

Q_hp_design = 37000.0    # [W]
DeltaT_water = 5.0      # [K]
cp_water = 4180.0       # [J/(kg·K)]

mdot_hp = Q_hp_design / (cp_water * DeltaT_water)  # ≈ 0.31 kg/s

# ------------------------------------------------------------
# Heating curve / T_offset
# ------------------------------------------------------------

T_flow_design = 45.0    # [°C]
T_out_design  = -10.0   # [°C]
T_room_set    = 20.0    # [°C]
k_heating     = 0.5

T_offset_base = T_flow_design - (T_room_set + k_heating*(T_room_set - T_out_design))
# ≈ 10°C

# ------------------------------------------------------------
# Building states
# ------------------------------------------------------------

# 0_soc: original state
mfh_1958_1968_0_soc = {
    'H_ve': H_ve_1,
    'H_tr': H_tr_1,
    'H_tr_light': H_tr_light_1,
    'c_bldg': c_bldg,
    'area_floor': area_floor,
    'height_room': height_room,
    'name': 'mfh_1958_1968_0_soc'
}

mfh_1958_1968_0_soc['T_offset'] = T_offset_base + 2.0   # ≈ 12°C
mfh_1958_1968_0_soc['T_amb_lim'] = 20.0
mfh_1958_1968_0_soc['mdot_hp']   = mdot_hp

w_east_0  = {'area': A_Window_East,  'tilt': 90, 'azimuth': 90,  'g_value': 0.75, 'c_frame': 0.3, 'c_shade': 0.6}
w_south_0 = {'area': A_Window_South, 'tilt': 90, 'azimuth': 180, 'g_value': 0.75, 'c_frame': 0.3, 'c_shade': 0.6}
w_west_0  = {'area': A_Window_West,  'tilt': 90, 'azimuth': 270, 'g_value': 0.75, 'c_frame': 0.3, 'c_shade': 0.6}
w_north_0 = {'area': A_Window_North, 'tilt': 90, 'azimuth': 0,   'g_value': 0.75, 'c_frame': 0.3, 'c_shade': 0.6}

mfh_1958_1968_0_soc['windows']  = [w_east_0, w_south_0, w_west_0, w_north_0]
mfh_1958_1968_0_soc['position'] = position_hannover


# 1_enev: standard refurbishment
mfh_1958_1968_1_enev = {
    'H_ve': H_ve_2,
    'H_tr': H_tr_2,
    'H_tr_light': H_tr_light_2,
    'c_bldg': c_bldg,
    'area_floor': area_floor,
    'height_room': height_room,
    'name': 'mfh_1958_1968_1_enev'
}

mfh_1958_1968_1_enev['T_offset'] = T_offset_base        # ≈ 10°C
mfh_1958_1968_1_enev['T_amb_lim'] = 20.0
mfh_1958_1968_1_enev['mdot_hp']   = mdot_hp

w_east_1  = {'area': A_Window_East,  'tilt': 90, 'azimuth': 90,  'g_value': 0.60, 'c_frame': 0.3, 'c_shade': 0.6}
w_south_1 = {'area': A_Window_South, 'tilt': 90, 'azimuth': 180, 'g_value': 0.60, 'c_frame': 0.3, 'c_shade': 0.6}
w_west_1  = {'area': A_Window_West,  'tilt': 90, 'azimuth': 270, 'g_value': 0.60, 'c_frame': 0.3, 'c_shade': 0.6}
w_north_1 = {'area': A_Window_North, 'tilt': 90, 'azimuth': 0,   'g_value': 0.60, 'c_frame': 0.3, 'c_shade': 0.6}

mfh_1958_1968_1_enev['windows']  = [w_east_1, w_south_1, w_west_1, w_north_1]
mfh_1958_1968_1_enev['position'] = position_hannover


# 2_kfw: deep refurbishment
mfh_1958_1968_2_kfw = {
    'H_ve': H_ve_3,
    'H_tr': H_tr_3,
    'H_tr_light': H_tr_light_3,
    'c_bldg': c_bldg,
    'area_floor': area_floor,
    'height_room': height_room,
    'name': 'mfh_1958_1968_2_kfw'
}

mfh_1958_1968_2_kfw['T_offset'] = T_offset_base - 2.0   # ≈ 8°C
mfh_1958_1968_2_kfw['T_amb_lim'] = 20.0
mfh_1958_1968_2_kfw['mdot_hp']   = mdot_hp

w_east_2  = {'area': A_Window_East,  'tilt': 90, 'azimuth': 90,  'g_value': 0.50, 'c_frame': 0.3, 'c_shade': 0.6}
w_south_2 = {'area': A_Window_South, 'tilt': 90, 'azimuth': 180, 'g_value': 0.50, 'c_frame': 0.3, 'c_shade': 0.6}
w_west_2  = {'area': A_Window_West,  'tilt': 90, 'azimuth': 270, 'g_value': 0.50, 'c_frame': 0.3, 'c_shade': 0.6}
w_north_2 = {'area': A_Window_North, 'tilt': 90, 'azimuth': 0,   'g_value': 0.50, 'c_frame': 0.3, 'c_shade': 0.6}

mfh_1958_1968_2_kfw['windows']  = [w_east_2, w_south_2, w_west_2, w_north_2]
mfh_1958_1968_2_kfw['position'] = position_hannover
