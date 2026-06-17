# Version 11 (v11) — The Radiator Physics Fix

## What we did
We discovered a major physics flaw: every single building was forced to use the exact same tiny radiator (`H_rad_con = 365.6 W/K`), regardless of whether the house was 111 m² or 288 m².
We changed the code (`models/vonovia_model.py`) to dynamically calculate and install mathematically correct, appropriately sized radiators for each house based on their design heat load.

## Why it was a success
In `v9`, the agent had to rapidly cycle the heat pump 400+ times to keep the house warm because the radiators were too small. After the `v11` fix, the extreme cycling vanished (dropping to ~30 to 100 cycles over 90 days), and comfort scores tightened up beautifully, with several buildings scoring **≥ 95% comfort**.
