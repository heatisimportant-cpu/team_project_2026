# 🗺️ RL Roadmap v2 — From v9 Onwards

> **Starting point:** v9 champion model (tag `v9.0` on branch `training-houses`)
> **Current best:** 87–98% comfort | 19.0–19.5°C minimums | ~3-4 HP cycles/day
> **Target:** ≥95% comfort across all buildings | Quantified cost savings vs baseline

This roadmap picks up where Roadmap v1 left off. All Phase 1 and Phase 2 items from the original roadmap are **completed** (reward fix, observation engineering, multi-building training, environment sanitization, reward engineering). This document defines the path forward.

---

## Summary of Completed Work (v4 → v9)

| Version | Change | Result |
|---------|--------|--------|
| v4 | Removed `1/1000` scaling, added asymmetric penalty | Comfort 76.9% → 85% |
| v5 | Added time features (sin/cos hour, day-of-week), previous action, forecast | Comfort → 88% |
| v6 | Proper train/test split, VecNormalize, longer training | Comfort → 90% |
| v7 | Multi-building training (11 buildings, one-hot encoding) | 5 buildings catastrophically fail |
| v8 | Environment sanitization (filter to 6 feasible buildings) | All buildings ≥77%, no failures |
| v9 | Aggressive asymmetric penalty (-5.0 → -20.0) | All buildings ≥87%, best 98.1% |
| v10 | Pre-heating bonus (+1.0) — **REVERTED** | Reward hacking, performance degraded |

---

## Phase 4: Physics Fidelity (v11–v12)

> **Goal:** Make the simulation physically accurate so the trained agent transfers to real hardware.

### Step 4.1 — Fix `H_rad_con` Per Building (v11)
**Priority: 🔴 Critical**
**Files:** `models/vonovia_model.py`, `models/vonovia_model.py` (Building class)

**Problem:** All 11 buildings share the same radiator coupling coefficient `H_rad_con = 365.6 W/K`, which was derived from the Vonovia building's design conditions (10,968 W at ΔT=30K). A small 111 m² house from 1949 does not have the same radiator capacity as a 288 m² modern building.

**What to do:**
1. For each building dict in `vonovia_model.py`, add a `design_heat_load` key (in Watts) based on `(H_ve + H_tr) * 30` (the peak heat loss we already calculated).
2. In `Building._calc_bldg_parameters()`, compute `H_rad_con = design_heat_load / (50 - 20)` per building instead of hardcoding.
3. Re-evaluate the v9 model on the corrected physics to see if comfort scores change.

**Verification:** Run the v9 evaluation loop on the 6 feasible buildings with corrected `H_rad_con`. Compare results to the v9 baseline.

---

### Step 4.2 — Add Solar Gains (v12)
**Priority: 🟡 Medium**
**Files:** `src/room_env.py`, possibly new file `models/solar_model.py`

**Problem:** The building models have detailed window specifications (`windows`, `tilt`, `frame_fraction`, `latitude`, `longitude`) but solar gains are never computed. `Qdot_gains` defaults to `0.0`. On a sunny winter day, south-facing windows can contribute 2–5 kW of free heating.

**What to do:**
1. Add a simple clear-sky solar irradiance model (or use `pvlib`) that computes global horizontal irradiance from timestamp + latitude/longitude.
2. For each window orientation (east, south, west, north), compute the incident irradiance.
3. Compute `Qdot_solar = Σ (window_area * (1 - frame_fraction) * g_value * irradiance)` per orientation.
4. Pass `Qdot_solar` as the `Qdot_gains` disturbance into the simulator.

**Verification:** Run a 365-day simulation and verify that:
- Solar gains peak at noon.
- South-facing windows contribute the most in winter.
- The agent learns to reduce HP output during sunny hours.

---

## Phase 5: Training Improvements (v13–v15)

> **Goal:** Squeeze the last 5-10% of comfort improvement through better training methodology.

### Step 5.1 — Longer Training (v13)
**Priority: 🔴 Critical**
**Files:** `train_SAC.py` (just change `--timesteps`)

**Problem:** Training logs show the reward was still improving at 1M steps. The policy has not fully converged.

**What to do:**
1. Train for 2M steps: `--timesteps 2_000_000`
2. Compare the final eval reward and comfort % against the 1M baseline.
3. If still improving, try 3M.

**Verification:** Compare v13 (2M steps) comfort % against v9 (1M steps) on identical evaluation.

---

### Step 5.2 — Fix Random Initialization Range (v14)
**Priority: 🟢 Low**
**Files:** `src/room_env.py` → `_initial_state_dict()`

**Problem:** Random init samples `T_room` from `[5.0, 60.0]`. Starting at 5°C or 55°C is physically impossible and wastes the replay buffer on meaningless transitions.

**What to do:**
```python
def _initial_state_dict(self) -> Dict:
    if self.random_init:
        return {
            'T_room':   float(self.np_random.uniform(18.0, 24.0)),
            'T_wall':   float(self.np_random.uniform(16.0, 24.0)),
            'T_hp_ret': float(self.np_random.uniform(20.0, 40.0)),
        }
    return {k: 20.0 for k in self.bldg_model.state_keys}
```

**Verification:** Compare training convergence speed (reward curve) against v13.

---

### Step 5.3 — Update Stale Docstring (v14)
**Priority: 🟢 Low**
**Files:** `src/room_env.py` → `_reward()` docstring

**Problem:** The docstring still says `-5.0 * (T_lower - T_room)²` but the code uses `-20.0`. This will confuse anyone reading the code.

**What to do:** Update the docstring to match the actual `-20.0` multiplier.

---

### Step 5.4 — Curriculum Learning (v15)
**Priority: 🟡 Medium**
**Files:** `train_SAC.py`, `src/room_env.py`

**Problem:** The agent trains on all 6 buildings from the start. Buildings have very different thermal time constants (τ ranges from ~15h to ~50h), which makes convergence harder.

**What to do:**
1. For the first 500k steps, train on only Building 0 (vonovia_model, the easiest).
2. At 500k, introduce Buildings 7 and 8 (next easiest).
3. At 750k, introduce all remaining buildings.
4. Continue to 2M total steps.

This is called curriculum learning and typically improves final performance by 10-20%.

**Verification:** Compare v15 comfort % against v13 (all buildings from the start).

---

## Phase 6: Evaluation & Benchmarking (v16–v17)

> **Goal:** Quantify how much better the RL agent is compared to a simple rule-based controller.

### Step 6.1 — Baseline Controller (v16)
**Priority: 🟡 Medium**
**Files:** New file `baseline_controller.py` or add `--baseline` flag to `evaluate.py`

**Problem:** We have no reference point to quantify the RL agent's value. "87% comfort" means nothing if a simple thermostat achieves 85%.

**What to do:**
1. Implement a simple heating-curve controller:
   ```
   if T_amb < T_amb_lim:
       T_sup = 20 + 1.5 * (20 - T_amb)   # standard heating curve
       T_sup = clip(T_sup, 25, 65)
   else:
       T_sup = T_hp_ret  # off
   ```
2. Run this controller through the same evaluation loop.
3. Compare comfort %, energy cost, and HP cycles side-by-side.

**Verification:** The RL agent should achieve comparable comfort at **lower energy cost** (that's the whole point of RL — it shifts load to cheap hours).

---

### Step 6.2 — Enhanced Evaluation Plots (v17)
**Priority: 🟢 Low**
**Files:** `evaluate.py`

**What to do:**
1. Add a **4th panel** showing instantaneous COP over time.
2. **Overlay electricity price** on the supply temperature panel (dual y-axis) to visualize load-shifting.
3. Add a **365-day evaluation mode** to verify summer shutdown behavior.
4. Add a summary comparison table when `--baseline` is used.

---

## Phase 7: Deployment Readiness (v18+)

> **Goal:** Prepare the model for real-world deployment or hardware-in-the-loop testing.

### Step 7.1 — Defrost Cycle Modelling
Apply a COP penalty factor (`COP *= 0.85`) when `0 < T_amb < 5°C` to model the defrost energy loss.

### Step 7.2 — Backup Heater Activation
The simulator should activate the 6 kW electric backup heater (at COP=1.0) when `Qdot_th` exceeds the HP's maximum capacity.

### Step 7.3 — Add `.gitignore`
```
__pycache__/
*.pyc
.DS_Store
.env
runs/
teamproject/
```

### Step 7.4 — Model Export for Edge Deployment
Export the trained SAC policy to ONNX format for deployment on an edge controller (e.g., Raspberry Pi or PLC).

---

## Quick Reference: Version Plan

| Version | Phase | Change | Key Metric |
|---------|-------|--------|------------|
| **v9** | ✅ Done | Aggressive asymmetric penalty | 87-98% comfort |
| v11 | 4.1 | Fix H_rad_con per building | Correct physics |
| v12 | 4.2 | Add solar gains | -10-15% energy |
| v13 | 5.1 | 2M training steps | +5-10% comfort |
| v14 | 5.2-5.3 | Fix random init + docstring | Faster convergence |
| v15 | 5.4 | Curriculum learning | +10-20% comfort |
| v16 | 6.1 | Baseline controller comparison | Quantify RL value |
| v17 | 6.2 | Enhanced eval plots | Better insights |
| v18+ | 7.x | Deployment readiness | Production ready |
