# Native parent route correction — 2026-09-17

Implementation branch: `2026_09_17_12_46_00`. This is an independently tested
preparation change; healthy scientific workers and the active controller remain
on their existing snapshots. It does not certify a qualified 3D parent.

The project-order compiler previously looked up both modalities using the same
architecture name. This works for ResNet, but cannot find the approved
`densenet121` / `monai_densenet121_3d` or `swin_b` /
`swin_unetr_encoder_3d` pair. A complete parent pair could therefore remain
blocked even after training finished. These are distinct model implementations,
not aliases for one network.

`studies/parent_routes.py` now specifies these pairs explicitly. Each family is
counted once; an unknown architecture fails closed. Incomplete parents remain
waiting. The compiler retains native verification, runtime integration gates,
per-seed release and accepted pilot-report requirements for later seeds.
The scientific arms, losses, data, selection and stopping rules are unchanged.

Ibex isolated CPU validation: 16 targeted tests passed, including actual
compiler calls for all three families, unavailable partners, unknown models,
per-seed release, idempotent bindings and forged pilot-acceptance rejection.
Evidence: operations `2026_09_10_11_11_31/radon_routes_20260917/tests.log`.
Local structure and diff checks passed. These tests use synthetic parent
receipts; real-model GPU acceptance is still required and is not implied.

At 09:45 UTC the active parent-controller configuration still has no `project`
binding. Do not merely insert one or fabricate a runtime-gate receipt. Next:
complete checkpoint/graph replay and resource qualification in a separate
admitted task, then bind the accepted project executor using the corrected
compiler. Intermediate checkpoints may support engineering preflight, but do
not pass the formal trained-parent gate. No new experiment or GPU request was
created by this correction. LOOK remains running independently.
