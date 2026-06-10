# Data generation – Virchowstr. 6 Winter 2026

This file describes how the input data used in `run_winter_2026.py` is put together. The idea is similar to the i4b project, but reduced to one building and one winter period.

The following files are used:

- `data/buildings/mfh_1968.py`
- `data/weather/hannover_2024_2026.csv`
- `data/prices/electricity_prices_2024.csv`
- `data/profiles/internal_gains.csv`
- `data/solar/hannover_solar_2024_2026.csv`  *(new – measured solar irradiance)*
- `data/solar/solar_gains_hourly.csv`        *(generated – hourly window gains)*

---

## 1. Building data: mfh_1968.py

The file `mfh_1968.py` defines the multi‑family house at Virchowstr. 6:

- year of construction: 1968  
- standard: EnEV  
- location: Langenhagen (near Hannover)  
- floor area: 287.92 m²  
- transmission heat loss coefficient H_tr  
- ventilation heat loss coefficient H_ve  
- effective thermal capacity c_bldg  
- window geometry: East 12 m², South 18 m², West 12 m², North 9 m² (frame fraction 0.30)

From these basic values the model code calculates:

- heat capacities of the air zone, wall/mass and water circuit  
- heat transfer coefficients between room, walls, ambient and heating system

---

## 2. Weather data: hannover_2024_2026.csv

The weather data file contains hourly outdoor temperatures for a longer time period around 2024–2026. For the Winter 2026 run only the slice between 1 December 2025 and 28 February 2026 is used.

Columns:

- `timestamp` – date and time (hourly, UTC)
- `T_amb` – outdoor air temperature in °C

---

## 3. Electricity prices: electricity_prices_2024.csv

The electricity price file is based on Tibber‑like data. One continuous year (2024) with hourly prices is exported to a CSV.

Columns:

- `timestamp` – hourly timestamps for 2024  
- `price_eur_kwh` – electricity price in EUR per kWh

Inside the script, this single‑year series is reused to cover Winter 2025/2026 by shifting all timestamps by +1 year and concatenating.

---

## 4. Internal gains: internal_gains.csv

The file `internal_gains.csv` represents heat gains from occupants, appliances, lighting, **and solar radiation through windows**.

Columns:

- `timestamp` – hourly time stamps  
- `Q_int_specific` – specific internal gains from occupancy/appliances [W/m²]  
- `Q_sol` – solar heat gains through windows [W]  
- `Q_int_total` – total gains used by the building model [W] = `Q_int_specific × area_floor + Q_sol`

The occupancy/appliance profile follows a typical residential daily pattern (evening peaks, night lows). The solar gains are described in section 5 below.

---

## 5. Solar irradiance data and window gains

### 5.1 Source: hannover_solar_2024_2026.csv

Measured 10-minute solar irradiance data from DWD station Hannover (station ID 2559), covering **2024-11-28 to 2026-05-31**.

Raw columns from DWD:

- `global_radiation_J_cm2` – global horizontal irradiance, energy per 10-min interval [J/cm²]  
- `diffuse_radiation_J_cm2` – diffuse horizontal irradiance [J/cm²]  
- `GHI_W_m2`, `DHI_W_m2` – pre-computed power columns (**contain a ×10 unit error** in the source file; the script corrects this)

### 5.2 Unit correction

The stored `GHI_W_m2` column was generated assuming a 1-minute measurement window instead of the actual 10-minute window, making all values 10× too large:

| Conversion | Factor |
|---|---|
| Correct:  `J/cm² × 10 000 / 600 s` | 16.667 W/m² per J/cm² |
| Stored:   `J/cm² × 10 000 / 60 s`  | 166.67 W/m² per J/cm² |

`data/solar/calc_solar_gains.py` always recomputes GHI and DHI from the raw J/cm² columns using the correct factor.

### 5.3 Solar position

Solar elevation and azimuth are calculated analytically for each hour (evaluated at the hour mid‑point for better representativeness):

- Declination from day-of-year  
- Equation of time correction  
- Hour angle from UTC time + longitude offset  
- Location: lat 52.44 °N, lon 9.74 °E

### 5.4 Irradiance on window surfaces (isotropic sky model)

For each vertical window:

```
I_total = I_beam + I_diff_sky + I_ground_reflected

I_beam    = DNI × max(0, cos(AOI))
DNI       = (GHI − DHI) / sin(elevation),  capped at 1 000 W/m²
              only computed for elevation > 5° (avoids horizon artefacts)
cos(AOI)  = cos(elevation) × cos(azimuth_sun − azimuth_surface)

I_diff    = DHI × 0.5        (isotropic half-sky, vertical)
I_ground  = GHI × 0.20 × 0.5 (ground albedo 0.20, vertical)
```

### 5.5 Building solar gains

```
Q_sol [W] = Σ_windows  I_total × A_window × (1 − c_frame) × g_value
```

Window parameters (EnEV state):

| Orientation | Area [m²] | Azimuth | c_frame | g-value |
|---|---|---|---|---|
| East | 12 | 90° | 0.30 | 0.60 |
| South | 18 | 180° | 0.30 | 0.60 |
| West | 12 | 270° | 0.30 | 0.60 |
| North | 9 | 0° | 0.30 | 0.60 |

Total effective glazing area: 35.7 m²  
Peak Q_sol (clear summer day, south window midday): ≈ 8 500 W

### 5.6 Pre-measurement period (Jan – Nov 2024)

For the period before the DWD measurements start, a synthetic clear-sky model is used:

```
GHI_cs = 950 × sin(elevation)   [W/m²]
GHI    = GHI_cs × kt(month)     monthly clearness index for Hannover
DHI    = GHI    × fd(month)     monthly diffuse fraction
```

This gives physically consistent solar gains for the full simulation range (full-year 2024 runs).

### 5.7 Regenerating the solar gains

```bash
python data/solar/calc_solar_gains.py
```

This script:
1. Loads and corrects `hannover_solar_2024_2026.csv`  
2. Generates synthetic data for Jan–Nov 2024  
3. Computes solar position and window irradiances  
4. Writes `data/solar/solar_gains_hourly.csv`  
5. Updates `Q_sol` and `Q_int_total` in `data/profiles/internal_gains.csv`

---

## 6. Thermal storage tank (defined in code)

The thermal storage tank is not stored as a CSV. It is defined directly in the simulation script.

Assumptions:

- volume: 500 L water  
- specific heat of water: about 4.18 kJ/(kg·K)  
- usable temperature spread: roughly 40 K  

This gives an order of magnitude of ~20–25 kWh of usable thermal storage. The script represents the tank by a state of charge (SOC) in percent.

---

## 7. Comfort band and deadband

The comfort definition used in the controllers is:

- setpoint: 21 °C  
- allowed range: 21 ± 1 °C → from 20 to 22 °C

---

## 8. How everything fits together

1. Building parameters from `mfh_1968.py` are used to initialise the 4R3C model.  
2. Weather, prices and gains (internal + solar) are read from CSV and cut to the period.  
3. The baseline and MPC controllers are run over the same time series.  
4. At each hour, the building model updates temperatures based on:  
   - ambient temperature  
   - total gains (`Q_int_total` = occupancy/appliances + **solar gains through windows**)  
   - heat delivered by the heat pump (and possibly the tank)  
   - control decisions  
5. Electricity use is calculated from heating energy and an estimated COP; cost from electricity and hourly prices.  
6. The script outputs CSVs, plots, and a compact summary.

The overall structure and data generation follow the same philosophy as the i4b project but are reduced to a single building and a single winter for clarity and faster iteration.
