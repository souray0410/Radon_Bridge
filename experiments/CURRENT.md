# Current experiment

Active: `2026_09_04_23_54_10` — R&B (Radon Bridge).

- Backbone LR: 3e-5 and 6e-5; head/bridge LR: 1e-4 / 1e-4.
- Fixed stage3, M=32, S=64; uniform rho=1/16, 1/8, 1/4.
- Seeds3416/3417: each LR has three rho arms plus its own no-bridge control (16 fresh trials).
- Existing independent best checkpoints; batch16, full fine-tuning, BN updates and validation plateau rules unchanged.
- GPU time budgets explicitly removed by the user. Continue actual time accounting; keep the 10 GiB per-GPU project memory limit and other LOOK workloads.

The lower-native-LR study `2026_09_04_22_32_30` remains historical negative evidence, not an active default. Per-source M/rho support remains available; this study uses equal scalar values.

ρ=1/2 was deferred by the user after exceeding the 10 GiB profile limit; it is excluded from this experiment matrix.


Queued follow-up: `2026_09_05_00_05_58` adds Gaussian projection, spatially scrambled Radon, self-only and pooled communication for both backbone LRs and seeds (16 additional trials). Match standard R&B and no-bridge references from the active study at M32/S64/rho1_8; pooled M/S are inapplicable and capacity differs. Combined final report: 32 trials. Deploy only after the active remote queue releases its lock.
