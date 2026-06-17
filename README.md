# Version 9 (v9) — The Aggressive Penalty Success

## What we did
We introduced the **Aggressive Asymmetric Penalty** into the reward function (`src/room_env.py`).
We changed the underheating penalty from `-5.0` to `-20.0`, making it 4x more painful for the agent to let the house drop below 20°C compared to overheating.
We also sanitized the training environments to only include 6 "feasible" buildings where the peak heat loss didn't overwhelm the 12kW heat pump.

## Why it was a success
This was the first time we saw **zero catastrophic failures**. Previous versions were dropping to 6.8°C. In `v9`, comfort scores skyrocketed to between **87% and 98%** across all 6 buildings.
