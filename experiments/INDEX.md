# Experiment ledger

**Status correction:** shared-head results in 001–003 do not test the user's separate-predictor formulation. Unimodal and late-fusion outcomes remain historical auxiliary results, not macro-F1-selected corrected baselines. See [specification audit](specification_audit/AUDIT.zh-CN.md). No new experiment was run during this audit.


GitHub main is always the latest code and experiment record. Timestamp branches
archive the previous main before an update; timestamps use Asia/Riyadh. Historical
branches use timestamps only. Experiment identifiers remain in directories, not
branch names. See [migration map](BRANCH_ARCHIVE.json). Explicit
source tags remain the immutable link from results to the actual executable code.
ws keeps one checkout on main, with independent data/cache/result storage.

| Experiment | Question | Execution source | Result record |
|---|---|---|---|
| 001 | Initial large handoff | pilot/001-source | [First pilots](../PILOT_REPORT.zh-CN.md) |
| 002 | Smaller handoff, frozen continuation backbone | pilot/002-source | [First pilots](../PILOT_REPORT.zh-CN.md) |
| 003 | CFP resolution, unimodal baselines, two seeds | run/003-source | [Report](003-cfp-resolution/REPORT.zh-CN.md) |
| 004 | Fixed late fusion without retraining | run/004-source | [Report](004-fixed-late-fusion/REPORT.zh-CN.md) |

001–003 do not establish an R&B advantage. 003 suggests retaining original-source
CFP 224 for stronger baselines; 004 gives a potentially useful fixed late-fusion
baseline (validation AUROC 0.6892) whose paired uncertainty intervals still cross
zero. These are exploratory results, with no test/clinical claim.

Next scientific gate: enlarge the training cohort and stabilize the pretrained
baselines, retain trained unimodal and fixed late fusion comparisons, then revisit
geometry against generic learned projections and paired shuffling. Reserve fresh
validation/test data for a frozen protocol instead of repeatedly tuning on the
same 128 participants. Do not select medical endpoints by the apparent R&B gain.

## 005 — corrected separate task paths

[Protocol](005-separate-tasks/README.md) · [Report](005-separate-tasks/REPORT.zh-CN.md). Source 88882f4; completed70epochs /5.08min. Per-task macro-F1; no stable R&B gain. Historical shared-head runs remain separate.

## 006 — implementation and optimization audit

[Report](006-implementation-audit/REPORT.zh-CN.md). Real checkpoint diagnostics found large train/validation gaps and coupled gradient clipping; new optimizer independence gate added. No new performance training.

## 007 — bounded combination sweep

[Protocol and acceptance](007-combination-sweep/README.md). 8 baseline recipes +20 communication configurations +10 seed-replication runs, each8epochs. Fixed within-trial optimization; five-minute monitor. Results pending execution.

## 008 — expanded cohort and rigorous development validation

[Execution plan](008-expanded-validation/PLAN.zh-CN.md). Verified1264train/296development-validation, sealed290test not evaluated. CPU cache preparation first; larger backbone/5-seed controls pending implementation after007. Thirty-minute monitoring.

007 completed:38trials/304epochs;76prediction-task checks passed. Expanded1264/296cache prepared. Qualification protocol forResNet18/34 × stage4/stage3+4 fixed before launch.
