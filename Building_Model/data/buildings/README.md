#

This repository contains a parameterized building model for a Multi‑Family House (MFH) from 1919–1948, based on TABULA data and structured for use with i4b‑style thermal and heat‑pump simulations.

The model includes:

Three refurbishment states (0_soc, 1_enev, 2_kfw)

TABULA‑derived thermal parameters

Assumed window optical properties

A building‑sized heat pump (37 kW)

A Vaillant‑based polynomial COP model

A consistent heating‑curve formulation

A reusable Python template

This README explains all modeling assumptions and parameter choices.

1. Building Overview
Type: Multi‑Family House (MFH)

Construction period: 1919–1948

Location: Hannover, Germany

Reference floor area: 350 m²

Dataset: TABULA (DE.N.MFH.03.Gen.ReEx.001)

Heat pump: Generic air‑to‑water HP sized to building load

Performance curve: Fitted polynomial from Vaillant aroTHERM plus VWL 65/5

The building is modeled in three states:

State	Description
0_soc	Original, unrefurbished
1_enev	Standard refurbishment
2_kfw	Deep refurbishment


Each state has its own heat‑loss parameters.

2. TABULA Parameters Used
TABULA provides the following thermal parameters:

Core thermal parameters
Parameter	Meaning
H_ve	Ventilation heat‑loss coefficient (specific)
H_tr	Transmission heat‑loss coefficient (specific)
H_tr_light	Lightweight internal heat transfer
c_bldg	Effective thermal mass


Geometry
Parameter	Meaning
area_floor	Reference floor area
height_room	Room height


Windows
TABULA provides areas only:

A_Window_East

A_Window_South

A_Window_West

A_Window_North

TABULA does not provide optical properties.

3. Window Optical Properties
Since TABULA does not include g‑values or frame fractions, we use standard assumptions:

g‑values
State	Glazing type	g‑value
0_soc	Old double glazing	0.75
1_enev	Modern double glazing	0.60
2_kfw	Triple glazing	0.50


Frame fraction
c_frame = 0.3
Shading coefficient
c_shade = 0.6
These values are consistent with EN ISO 52016 and Fraunhofer IBP defaults.

4. Location Parameters
Hannover, Germany:

lat = 52.3759
long = 9.7320
altitude = 55
timezone = "Europe/Berlin"
Used for weather and solar geometry.

5. Heat Pump Model (Building‑Sized)
The real Vaillant unit is ~6.5 kW, but the TABULA MFH requires:

𝑄
build,design
≈
37
 kW
Therefore, the model uses a building‑sized heat pump:

Code
Q_hp_design = 37000 W
This ensures:

The building reaches setpoint at −10 °C

The HP model is physically consistent

i4b simulations behave realistically

Mass flow rate
𝑚
˙
=
𝑄
hp,design
𝑐
𝑝
⋅
Δ
𝑇
Using:

Q_hp_design = 37000 W

DeltaT_water = 5 K

cp_water = 4180 J/(kg·K)

Result:

mdot_hp ≈ 1.77 kg/s
COP model
Your fitted Vaillant polynomial is used as the performance curve, not as the nominal power.

This is exactly how i4b expects heat pumps to be modeled.

6. Heating Curve and T_offset
Heating curve:

𝑇
flow
=
𝑇
room
+
𝑘
(
𝑇
room
−
𝑇
out
)
+
𝑇
offset
Parameters:

T_room_set = 20°C

k_heating = 0.5

T_flow_design = 45°C at T_out_design = -10°C

Solving gives:

T_offset_base ≈ 10°C
Refurbishment shifts:

State	T_offset
0_soc	12°C
1_enev	10°C
2_kfw	8°C


Better buildings → lower flow temperatures.