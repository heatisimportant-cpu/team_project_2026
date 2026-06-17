# Version 16 (v16) — Contextual RL & "Big Brain" Network

## What we did
1. **Contextual RL**: We stripped the old one-hot building ID from the observation space and injected 5 actual physical parameters: Envelope heat loss (`H_tr`), Ventilation loss (`H_ve`), Thermal mass (`c_bldg`), Floor area, and Number of pumps. The agent now learns thermodynamics instead of memorizing IDs!
2. **Physics Exploit Fix**: Added a hard limit in the simulator so the RL agent cannot request more heat than the cascaded heat pumps can physically output.
3. **Big Brain Network**: Because the agent now has to manage 20 completely diverse buildings, we exponentially increased the SAC neural network size from the default `[256, 256]` to a massive 3-layer `[512, 512, 512]` architecture (nearly 3 million parameters) and trained it for 3,000,000 timesteps.

## Results & Analysis
The massive neural network perfectly fine-tuned the highly refurbished, well-insulated buildings (`kfw` models), achieving **94.2% comfort** with incredibly tight temperature ranges.

However, the agent catastrophically broke down on the leaky, unrefurbished 1949 Multi-Family Houses (comfort crashed to 18.3%). It "forgot" how to handle the mid-range houses. This was caused by **neural network overfitting** and the agent exploiting the asymmetric reward function (panicking in leaky houses because the overheating penalty is too weak).
