# 🗺️ RL Roadmap v3 — Multi-Family House Integration

> **Starting point:** v14 champion model on `training-houses` branch (linear+quadratic penalty, 3M steps, ~95% comfort)
> **Current branch:** `training-multi-houses`
> **New assets:** 4 MFH model files (12 building states) + cascaded heat pump code
> **Goal:** A single generalised RL agent that controls SFH (1 pump) and MFH (2–4 cascaded pumps) across all refurbishment states

---

## Current Inventory

### Single-Family Houses (SFH) — from `vonovia_model.py`
| # | Building | Floor Area | H_ve+H_tr | Refurb States | Pumps |
|---|----------|-----------|-----------|---------------|-------|
| 1 | vonovia_model | ~120 m² | ~3.5 W/(m²K) | 1 (original) | 1 |
| 2–11 | sfh_1919…2016 | ~120 m² | varies | 1 (original) | 1 |

**After filtering:** 6 buildings pass the 12kW feasibility check (H_ve+H_tr ≤ 400 W/K).

### Multi-Family Houses (MFH) — NEW
| # | File | Floor Area | Refurb States | H_tr range |
|---|------|-----------|---------------|------------|
| 1 | `mfh_1919_1948.py` | 350 m² | 3 (0_soc, 1_enev, 2_kfw) | 0.47 – 2.99 W/(m²K) |
| 2 | `mfh_1949_1957.py` | 574.8 m² | 3 (0_soc, 1_enev, 2_kfw) | 0.45 – 2.54 W/(m²K) |
| 3 | `mfh_1958_1968.py` | 2844.61 m² | 3 (0_soc, 1_enev, 2_kfw) | 0.35 – 1.74 W/(m²K) |
| 4 | `mfh_1969_1978.py` | 426.01 m² | 3 (0_soc, 1_enev, 2_kfw) | 0.42 – 1.85 W/(m²K) |

### Heat Pump — `heatpump_model.py`
- **Model:** iDM AERO ALM 4-12 (single unit: 4–13 kW thermal, COP 1.39–5.42)
- **Cascade class:** `Cascaded_iDM_AERO_ALM_4_12(num_units=N)` — scales P_el_max, P_el_min, heater_power by N

---

## Critical Analysis: What Must Change

### Problem 1: MFH buildings are WAY too big for 1 heat pump
| Building | State | Peak Heat Loss at -10°C | 1 Pump Capacity | Pumps Needed |
|----------|-------|------------------------|-----------------|-------------|
| mfh_1919_1948 | 0_soc (original) | 350 × (0.51+2.99) × 30 = **36,750 W** | ~13 kW | **3** |
| mfh_1919_1948 | 2_kfw (deep refurb) | 350 × (0.43+0.47) × 30 = **9,450 W** | ~13 kW | **1** |
| mfh_1949_1957 | 0_soc | 574.8 × (0.51+2.54) × 30 = **52,614 W** | ~13 kW | **4** |
| mfh_1958_1968 | 0_soc | 2844.6 × (0.51+1.74) × 30 = **192,010 W** | ~13 kW | **15** ⚠️ |
| mfh_1958_1968 | 2_kfw | 2844.6 × (0.43+0.35) × 30 = **66,563 W** | ~13 kW | **5** |
| mfh_1969_1978 | 0_soc | 426 × (0.51+1.85) × 30 = **30,161 W** | ~13 kW | **3** |

> [!WARNING]
> **`mfh_1958_1968` is a monster building** (2,844 m²). Even deep-refurbished, it needs 5 cascaded pumps. In original state it needs **15 pumps**. This building may need special treatment or exclusion from the initial training set.

### Problem 2: The `BUILDING_MODELS` list only has SFH buildings
The MFH models are defined in their own files but are NOT imported or registered anywhere. We need to import them into a centralised registry.

### Problem 3: The observation space uses one-hot building IDs
Currently the agent sees `[0,0,1,0,0,0]` to know which building it's in. With 6 SFH + 12 MFH states = **18 buildings**, the one-hot vector would be 18 dims long and the agent wouldn't generalise to new buildings. We need **contextual parameters** instead.

### Problem 4: The environment creates a single HP per building
`room_env.py` line 168: `self.hp_model = self.current_heatpump_class()` — always creates one pump. For MFH we need `Cascaded_iDM_AERO_ALM_4_12(num_units=N)`.

### Problem 5: The `Q_hp_design` and `mdot_hp` in MFH files are wrong
All 4 MFH files hardcode `Q_hp_design = 37000.0 W` and derive `mdot_hp` from it. This is a placeholder — we should derive it from the actual cascade size.

---

## Implementation Phases

### Phase 1: MFH Model Integration (v15)
> **Goal:** Get all 18 building profiles importable and registered. No RL changes yet.

#### Step 1.1 — Add `num_pumps` to every building dict
Each building needs to declare how many heat pumps it requires. Calculate this automatically from peak heat loss vs single-pump capacity.

**Files:** `vonovia_model.py` (add `num_pumps=1` to all SFH), all 4 `mfh_*.py` files

#### Step 1.2 — Fix `mdot_hp` in MFH files
Remove the hardcoded `Q_hp_design = 37000` from MFH files. Instead, derive `mdot_hp` from the cascade:
```python
# mdot per pump ≈ 0.27 kg/s (from datasheet ΔT=5K at max capacity)
mdot_hp = 0.27 * num_pumps
```

#### Step 1.3 — Create unified `ALL_BUILDING_MODELS` list
**File:** `models/__init__.py`
Import all SFH + MFH dicts and combine into one master list. Add a `'type'` field (`'sfh'` or `'mfh'`) for identification.

#### Step 1.4 — Verify: Run self-test
```python
python -c "from models import ALL_BUILDING_MODELS; print(f'{len(ALL_BUILDING_MODELS)} buildings loaded')"
```

---

### Phase 2: Cascaded Heat Pump in Simulator (v16)
> **Goal:** The Simulator creates the correct number of cascaded pumps per building.

#### Step 2.1 — Update `room_env.py` to use cascade
In `_select_models()`, read `num_pumps` from the building dict and instantiate:
```python
num_pumps = self.current_building_params.get('num_pumps', 1)
self.hp_model = Cascaded_iDM_AERO_ALM_4_12(num_units=num_pumps)
```

#### Step 2.2 — Update `mdot_hp` per building
Pass the building's `mdot_hp` (which now scales with cascade size) through to the `Building` class.

#### Step 2.3 — Update the building filter
The current filter (`H_ve + H_tr <= 400.0`) was designed for single-pump SFH. For MFH with cascaded pumps, the filter logic needs to check:
```python
peak_heat_loss = (b['H_ve'] + b['H_tr']) * b['area_floor'] * 30.0  # W at -10°C
max_cascade_capacity = 13000.0 * b.get('num_pumps', 1)              # W
b passes if peak_heat_loss <= max_cascade_capacity * 1.1            # 10% margin
```

#### Step 2.4 — Verify: Physics sanity check
Run a single MFH building (e.g., `mfh_1919_1948_2_kfw` with 1 pump) for 90 days and confirm:
- Room temperature stays within [18, 24] °C
- HP cycles are reasonable
- No ODE solver failures

---

### Phase 3: Contextual Observation Space (v17)
> **Goal:** Replace one-hot building IDs with physics parameters so the agent generalises.

#### Step 3.1 — Define contextual features
Replace the one-hot vector with normalised physics parameters:

| Feature | Range | Why the agent needs it |
|---------|-------|----------------------|
| `H_tr` (normalised) | [0, 1] | How fast the building loses heat |
| `H_ve` (normalised) | [0, 1] | Ventilation losses |
| `c_bldg` (normalised) | [0, 1] | Thermal mass (how long heat is stored) |
| `area_floor` (normalised) | [0, 1] | Building size |
| `num_pumps` (normalised) | [0, 1] | Available heating capacity |

**Normalisation:** Use min-max across all registered buildings so every feature falls in [0, 1].

#### Step 3.2 — Update `_build_obs_space()` and `_build_obs()`
Replace the one-hot block (currently 6–11 dims) with the 5 contextual features above.

**New obs dimensions:** 3 (states) + 2 (current weather/price) + 24 (price forecast) + 24 (weather forecast) + 4 (time encoding) + 1 (prev action) + 5 (context) = **63 dims**

#### Step 3.3 — Verify: Observation space is correct
```python
env = RoomHeatEnv(disturbances=df, days=30, random_init=True)
print(f"Obs shape: {env.observation_space.shape}")  # should be (63,)
```

---

### Phase 4: Training & Evaluation (v18)
> **Goal:** Train the unified agent and verify it handles all building types.

#### Step 4.1 — First training run (feasible buildings only)
Exclude `mfh_1958_1968` (the 2844 m² monster) for now. Train on remaining buildings:
- 6 SFH × 1 state = 6 profiles
- 3 MFH × 3 states = 9 profiles (excluding mfh_1958_1968)
- **Total: 15 building profiles**

```bash
python train_SAC.py \
  --data data/train_2021_2023.csv \
  --eval_data data/test_2024_onwards.csv \
  --days 30 \
  --timesteps 3000000 \
  --comfort_weight 10.0 \
  --cycle_weight 0.5 \
  --run_name sac_hp_v18_multi_house \
  --seed 42
```

#### Step 4.2 — Evaluate across all building types
Run evaluation separately for SFH (1 pump) and MFH (2–4 pumps) to compare performance.

#### Step 4.3 — Verify generalisation
Pick a building the agent has NEVER seen during training (e.g., hold out `mfh_1969_1978_1_enev`). Evaluate on it. If comfort > 85%, the contextual approach is working — the agent has learned physics, not memorised buildings.

---

### Phase 5: Bring Back mfh_1958_1968 (v19)
> **Goal:** Handle the 2844 m² building that needs 5–15 pumps.

#### Step 5.1 — Decision point
- **Option A:** Cap cascading at 6 pumps (78 kW). Only train on the deep-refurbished state (needs 5 pumps). The original state is physically unreasonable for air-source heat pumps.
- **Option B:** Allow up to 15 pumps but this changes the `num_pumps` normalisation range significantly.

#### Step 5.2 — Retrain with the expanded set

---

## Summary Timeline

| Version | What | Key Files | Est. Effort |
|---------|------|-----------|------------|
| v15 | MFH model integration | `vonovia_model.py`, `mfh_*.py`, `models/__init__.py` | 1 hour |
| v16 | Cascaded HP in simulator | `room_env.py`, `simulator.py` | 1 hour |
| v17 | Contextual observation space | `room_env.py` | 1 hour |
| v18 | Train & evaluate unified agent | `train_SAC.py`, `evaluate.py` | 2 hours (training) |
| v19 | Big MFH building support | All | TBD |

**Total implementation time:** ~3 hours of coding + training time

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| MFH buildings have wildly different thermal dynamics than SFH | Agent struggles to converge | Start with deep-refurbished MFH only (closest to SFH), add original states later |
| Cascaded HP scaling creates reward imbalance (MFH costs 4× more than SFH) | Agent avoids MFH buildings | Normalise electricity cost by floor area in reward function |
| `mfh_1958_1968` at 2844 m² is an outlier | ODE solver instability, unreasonable pump counts | Exclude from initial training, add in Phase 5 |
| Observation space change breaks old models | Can't compare to v14 | Expected — this is a fresh architecture. Keep v14 as baseline reference. |
