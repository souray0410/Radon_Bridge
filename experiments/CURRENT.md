# Current experiment sequence

`2026_09_04_21_41_26` runs overnight ablations, then `2026_09_04_21_45_45` fills the remaining self-only/pooled mechanism pair.

Both original queues stopped at their group-budget gates. The detached continuation
`2026_09_04_22_24_45` completes seed3417 and mechanism controls within one inherited
global budget. The next study `2026_09_04_22_32_30` waits for it, then reruns the
seed3416/3417 matched no-bridge/M32 pairs at lower backbone/head learning rates.
It retains scalar M32/rho1_8; participant-specific M/rho is supported but not used
in this learning-rate comparison. Runtime status and accounting live under each
run directory on ws02.
