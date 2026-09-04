# Current experiment

Active plan: `2026_09_04_23_54_10` — R&B (Radon Bridge), restored backbone/head/bridge learning rates 3e-5 / 1e-4 / 1e-4. Fix stage3, M=32, S=64 and compare rho=1/16, 1/8, 1/4, 1/2. Use seed3416 for the matched sweep; explicitly reuse its completed no-bridge, rho=1/16 and rho=1/8 references, and add rho=1/4 and rho=1/2 within the inherited global budget.

The lower-native-LR study `2026_09_04_22_32_30` completed all four trials and is retained as a historical negative result. It is no longer the active default. Per-source M/rho support remains available; this study uses equal scalar values. All prior run artifacts remain immutable.
