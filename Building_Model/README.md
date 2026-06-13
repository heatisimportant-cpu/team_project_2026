# MPC – Winter 2026 Heating System

This repository contains a heating system model for the multi‑family house at Virchowstr. 6 in Langenhagen (MFH 1968, EnEV standard). The goal is to compare a normal weather‑compensated control with a price‑aware control that uses a 500 L buffer tank to shift electricity use to cheaper hours, while keeping the room temperature around 21 °C (21 ± 1 °C).

At the moment the project focuses on one period:

- **Winter 2026**: 1 December 2025 – 28 February 2026 (2137 hours)

The model uses a reduced‑order 4R3C building model, hourly weather data for the Hannover region, hourly electricity prices (Tibber‑like), a residential internal gains profile, and a simple thermal storage tank model.

---

## 1. Quick start

```bash
# install dependencies
pip install -r requirements.txt

# (optional) run tests for the building model
pytest tests

# run the Winter 2026 comparison
python examples/simulation/run_winter_2026.py
```

The script runs two cases:

1. **Baseline**: heating curve controller, no buffer tank  
2. **MPC + tank**: price‑aware controller with 500 L buffer tank and 21 ± 1 °C comfort band

All results are written to `examples/simulation/results`.

---

## 2. What the Winter 2026 simulation does

`examples/simulation/run_winter_2026.py`:

1. Loads the MFH 1968 Virchowstr. 6 building data from `data/buildings/mfh_1968.py`.
2. Loads weather, electricity prices and internal gains from CSV files.
3. Cuts all time series to the period 2025‑12‑01 to 2026‑02‑28.
4. Runs:
   - a **baseline** controller with a heating curve and proportional feedback around 21 °C,
   - an **MPC‑like** controller with a 500 L buffer tank that uses price information plus tank state of charge (SOC) to shift heat pump operation.
5. Computes heat delivered, electricity use, costs and comfort statistics.
6. Saves the full hourly results to CSV and creates time‑series plots for 15‑day windows.

A typical console summary looks like this:

```text
======================================================================
Winter 2026 Heating System Comparison with Thermal Storage
MFH 1968 Virchowstr. 6, Langenhagen
Target room temperature: 21°C
======================================================================

Metric                                Baseline        MPC+Tank      Savings
------------------------------------------------------------------------
Hours simulated                           2137            2137            -
Heat energy (kWh)                      17807.8         15817.5        11.2%
Electricity (kWh)                       5420.8          4772.6        12.0%
Cost (EUR)                             1271.14         1092.31        14.1%
Average COP                               3.58            3.64            -
Avg supply temp (°C)                      42.4            41.7            -
Avg tank SOC (%)                           0.0            15.8            -
Comfort 20-22°C (%)                      100.0           100.0            -
Comfort 20-24°C (%)                      100.0           100.0            -
Avg room temp (°C)                       22.00           21.99            -
Min room temp (°C)                       21.37           21.26            -
Max room temp (°C)                       22.00           22.00            -
Hours too cold (<20°C)                       0               0            -
Hours too hot (>22°C)                        0               0            -
```

Interpretation (current setup):

- Both controllers keep the room in **21 ± 1 °C** all the time.
- The MPC + tank case delivers almost the same useful heat but uses less electricity and reduces cost because some operation is shifted from expensive to cheaper hours.

---

## 3. Controllers

### 3.1 Baseline: heating curve (no tank)

The baseline controller behaves like a typical weather‑compensated system:

- **Heating curve**: supply temperature is a function of outdoor temperature.
- **Feedback**: a proportional term around a 21 °C room setpoint corrects deviations.
- **Deadband**: the controller only reacts when the room tries to leave the 21 ± 1 °C band to avoid unnecessary changes.
- **No buffer tank**: all heat is delivered directly by the heat pump to the building.

This is a reference for a “normal” control without price signals or storage.

### 3.2 MPC + 500 L buffer tank

The MPC‑like controller uses the same building model but adds a tank and the price signal:

- **Tank model**:
  - 500 L water buffer tank.
  - SOC (state of charge) in % (0–100) represents how full the tank is.
  - Simple heat loss term to represent tank standing losses.

- **Price logic** (conceptual):
  - Compute low and high price thresholds from the whole winter price distribution (percentiles).
  - If the current price is **cheap**, future prices are higher and SOC is not full → increase heat pump output and charge the tank.
  - If the current price is **expensive** and SOC is not empty → reduce heat pump output and discharge the tank.

- **Comfort**:
  - Target 21 °C.
  - Deadband ±1 °C → acceptable range 20–22 °C.
  - In the simple model, room temperature is also clipped to [20, 22] °C after each time step.

The aim is not a full optimisation problem as in the original i4b MPC, but a simple, understandable controller that still shows the main effect: shifting some electricity use away from expensive hours.

---

## 4. Building model (4R3C)

The building is represented by a small 4R3C model similar in spirit to the one used in i4b.

States:

- `T_room` – indoor air temperature  
- `T_wall` – lumped wall / building mass temperature  
- `T_hp_ret` – return temperature of the heating circuit

Inputs:

- `T_hp_sup` – supply temperature from the heat pump  
- `T_amb` – outdoor temperature  
- `Q_gains` – internal gains (occupants and appliances, no solar)

The parameters (heat capacities and conductances) are derived from the building description in `data/buildings/mfh_1968.py`. They are typical for a 1968 multi‑family house refurbished to EnEV standard and are in line with the values used in the i4b examples.

If you want the full derivation and the general background, you can look at the original i4b documentation:

- https://github.com/lfrison/i4b/blob/main/README.md  
- https://github.com/lfrison/i4b/blob/main/DATA_GENERATION.md

---

## 5. Data files

### 5.1 Building data

`data/buildings/mfh_1968.py` contains:

- floor area, room height, etc.  
- transmission and ventilation heat loss coefficients (H_tr, H_ve)  
- effective thermal capacity of the building

These are used to set up the 4R3C model.

### 5.2 Weather

`data/weather/hannover_2024_2026.csv` holds hourly outdoor temperatures for a period that contains Winter 2026. The script filters this file to 1 Dec 2025 – 28 Feb 2026.

### 5.3 Electricity prices

`data/prices/electricity_prices_2024.csv` holds hourly electricity prices for a base year. The script shifts this series by one year to get a 2025 series, combines them and then cuts out the Winter 2026 window. This way, hourly price patterns are realistic without needing a separate multi‑year price file.

### 5.4 Internal gains

`data/profiles/internal_gains.csv` holds a residential internal gains profile (no solar). It is used directly as `Q_gains` in the building model.

---

## 6. Output

After a run you will find in `examples/simulation/results`:

- `winter_2026_baseline.csv`  
- `winter_2026_mpc.csv`  
- `winter_2026_days_001_015.png`  
- `winter_2026_days_016_030.png`  
- `winter_2026_days_031_045.png`  
- `winter_2026_days_046_060.png`  
- `winter_2026_days_061_075.png`  
- `winter_2026_days_076_090.png`

The CSV files contain the full hourly time series.  
Each PNG file shows a 15‑day section with six plots:

1. room temperature (and 20–22 °C band)  
2. supply temperature  
3. heat output  
4. COP  
5. tank SOC (for MPC + tank)  
6. outdoor temperature