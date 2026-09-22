# Historical reproduction

The boundary is the naming and architecture transition on 2026-09-09. [Pre-rename source](https://github.com/souray0410/Radon_Bridge/tree/archive/2026_09_09_10_30_34_before_rename) preserves commit `87c6fb0` and its original directory layout, experiment records and dependencies. Existing older study branches remain available. Use the source version recorded by each experiment for reproduction; do not substitute current main for an old serialized Python model.

`main` contains one consolidated architecture migration. Today's temporary preparation and partial-layout branches are superseded and retired after acceptance. This does not delete historical research data, predictions, checkpoints or reports from their authorized archives.

State_dict numerical compatibility is checked separately from Python pickle import-path compatibility. New data, training and test access require their own protocol; no old queue resumes automatically.

V5 removes the unused historical `analysis.cli` shared-head diagnostic and `training.pretrain` positional PilotGraph pilot from the current wheel. Neither matches the current constructor, and neither has a current execution caller. Original source remains in Git and frozen experiment snapshots; historical checkpoints are not routed through a V4 compatibility mode. Current independent/cohort training is in `training.trainer` and `studies.cohort_case`, with explicit offline conversion and accepted replay. The module migration map retains the earlier rename history and marks these two current modules retired.
