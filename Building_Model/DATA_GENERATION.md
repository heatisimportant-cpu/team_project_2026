# Data generation – Virchowstr. 6 Winter 2026

This file describes how the input data used in `run_winter_2026.py` is put together. The idea is similar to the i4b project, but reduced to one building and one winter period.

The following files are used:

- `data/buildings/mfh_1968.py`
- `data/weather/hannover_2024_2026.csv`
- `data/prices/electricity_prices_2024.csv`
- `data/profiles/internal_gains.csv`

---

## 1. Building data: mfh_1968.py

The file `mfh_1968.py` defines the multi‑family house at Virchowstr. 6:

- year of construction: 1968  
- standard: EnEV  
- location: Langenhagen (near Hannover)  
- floor area: 287.92 m²  
- transmission heat loss coefficient H_tr  
- ventilation heat loss coefficient H_ve  
- effective thermal capacity c_bldg

From these basic values the model code calculates:

- heat capacities of the air zone, wall/mass and water circuit  
- heat transfer coefficients between room, walls, ambient and heating system

This is the same idea as in i4b: use a simple 4R3C network instead of a full detailed building energy model, but still based on physical parameters and typical values.

---

## 2. Weather data: hannover_2024_2026.csv

The weather data file contains hourly outdoor temperatures for a longer time period around 2024–2026. For this project, only the slice between 1 December 2025 and 28 February 2026 is used.

Columns:

- `timestamp` – date and time (hourly)  
- `T_amb` – outdoor air temperature in °C

The script reads this file, converts the timestamps and filters the rows so that only the winter period remains.

No other processing is done in the script (any preparation of the weather file – e.g. from measurement or a weather generator – is assumed to have been done beforehand).

---

## 3. Electricity prices: electricity_prices_2024.csv

The electricity price file is based on the Tibber‑like data from an Excel file. One continuous year (2024) with hourly prices is exported to a CSV.

Columns:

- `timestamp` – hourly timestamps for 2024  
- `price_eur_kwh` – electricity price in EUR per kWh

Inside the script, this single‑year series is reused to cover Winter 2025/2026:

1. Read the 2024 series.  
2. Create a copy with all timestamps shifted by +1 year to represent 2025.  
3. Concatenate both years.  
4. Filter the combined data set for the Winter 2026 period.

This gives a realistic hourly price pattern for the winter without needing a separate file for each year.

The same price time series is used:

- for the **cost calculation** of both controllers, and
- by the **MPC + tank controller** to detect “cheap” and “expensive” hours (for example by comparing against percentiles of the whole winter price distribution).

---

## 4. Internal gains: internal_gains.csv

The file `internal_gains.csv` represents internal heat gains from occupants, appliances and lighting.

Columns:

- `timestamp` – hourly time stamps  
- `Q_int_total` – internal gains in W

The profile follows a typical residential daily pattern (evening peaks, night lows) and repeats over the full time range. There are **no solar gains** included here – the model only uses internal gains from usage.

The script reads this file and cuts it to the same winter period as weather and prices.

---

## 5. Thermal storage tank (defined in code)

The thermal storage tank is not stored as a CSV. It is defined directly in the simulation script.

Assumptions:

- volume: 500 L water  
- specific heat of water: about 4.18 kJ/(kg·K)  
- usable temperature spread: roughly 40 K  

This gives an order of magnitude of ~20–25 kWh of usable thermal storage. The script represents the tank by a state of charge (SOC) in percent. Each hour:

- SOC increases when the heat pump output is higher than the building demand (charging),
- SOC decreases when the tank covers part of the space heating demand (discharging),
- a fixed fraction of SOC is lost due to standing losses.

The MPC‑like controller uses SOC together with the electricity prices to decide when to charge or discharge.

---

## 6. Comfort band and deadband

The comfort definition used in the controllers is:

- setpoint: 21 °C  
- allowed range: 21 ± 1 °C → from 20 to 22 °C

In the controller logic a deadband is used around the setpoint:

- As long as the room is within 20–22 °C the controller does not react strongly.  
- When the room tends to go below 20 °C or above 22 °C, the controller increases or reduces the supply temperature more aggressively.

In the simplified model implementation the room temperature is also clipped to the interval [20, 22] °C in each time step to respect this comfort requirement.

Comfort metrics in the summary:

- “Comfort 20‑22 °C (%)” – share of hours strictly within the 21 ± 1 °C band  
- “Comfort 20‑24 °C (%)” – broader band to show if there are any larger deviations

---

## 7. How everything fits together

Putting all the above steps into one picture:

1. Building parameters from `mfh_1968.py` are used to initialise the 4R3C model.  
2. Weather, prices and internal gains are read from CSV and restricted to the Winter 2026 period.  
3. The baseline controller and the MPC + tank controller are run over the same time series.  
4. At each hour, the building model updates the temperatures based on:
   - ambient temperature,
   - internal gains,
   - heat delivered by the heat pump (and possibly from the tank),
   - and the control decisions (heating curve vs price‑aware logic).  
5. Electricity use is calculated from the heating energy and an estimated COP; cost is calculated from electricity and the hourly prices.  
6. The script outputs CSVs, plots and a compact summary.

The overall structure and data generation follow the same philosophy as the i4b project but are reduced to a single building and a single winter for clarity and faster iteration.