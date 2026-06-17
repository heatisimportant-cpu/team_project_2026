# Version 14 (v14) — The Linear Penalty Loophole Fix

## What we did
When we trained `v13` for a massive 5 million steps, the agent got "too smart". It realized that a purely quadratic penalty meant dropping slightly to 19.5°C barely punished it, so it traded comfort for lower electricity bills.
In `v14`, we closed this loophole by adding a **Linear Term** to the penalty (`-20*|T_lower - T| - 20*(T_lower - T)²`) in `src/room_env.py`.

## Why it was a success
This was arguably the best, most robust model we ever produced on the 6 sanitized buildings. The linear term acted like a physical wall. Minimum temperatures locked tightly at **19.3°C – 19.7°C**, comfort sat near a perfect **95% average**, and compressor cycles were extremely healthy (2-3 starts per day).
