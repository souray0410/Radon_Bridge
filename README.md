# R&B / RadonBridge

最新：[007组合筛查方案](experiments/007-combination-sweep/README.md)，38次短跑，含修复后的独立基线、桥接组合和两个新种子复核。

Current implementation preserves two complete native task paths and uses intermediate
feature communication through explicit MHD V4 forward/backward. Read [requirements](REQUIREMENTS.md),
[full source specifications](specification/README.md), [legacy audit](experiments/specification_audit/AUDIT.zh-CN.md)
and [corrected pilot 005](experiments/005-separate-tasks/README.md).

[005结果](experiments/005-separate-tasks/REPORT.zh-CN.md)：70 epoch正常完成，约5.08分钟；未见稳定桥接增益，续训稳定性是下一项诊断。

## Current pilot

- UKB participant record-derived glaucoma; 256 training / 128 validation, both eyes.
  CFP224 and volumetric OCT32x96x96. No test evaluation.
- ImageNet ResNet18 and a 3D inflated ResNet18, each with its own head, CE and metrics.
- Independent warmup, then matched no-bridge / one-bridge / two-bridge / scrambled /
  self-only continuations. Stage3/4 and heads adapt; earlier stages and BN stats frozen.
- FeatureSpec / attach_group accept heterogeneous K inputs, retain identities, and
  implement M/S/H reference controls. Linear mixer, Householder, adjoint, direct residual.
- Per-task macro-F1 primary, macro precision/recall and auxiliary AUROC reported separately.
- Dense reference operators support small nD features; scalable high-resolution kernels
  and evidence of clinical effectiveness are not claimed.

## Run and limits

Use the pinned submodule and compatible PyTorch/torchvision. Source data, predictions and
checkpoints remain on ws. One GPU; allocator cap8GiB, own-process stop9.5GiB; 15minute budget.
No automatic interrupted-run resume; existing output directories cannot be overwritten.

```bash
git submodule update --init
export PYTHONPATH="$PWD:$PWD/third_party/MHD_Project"
python tests/check_core.py
python tests/check_requirements.py
CUDA_VISIBLE_DEVICES=1 python -m radonbridge.separate_pilot \
  --data CACHE_ROOT --output RUN_ROOT \
  --protocol experiments/005-separate-tasks/protocol.json --lock LOCK_PATH
```

## Legacy experiments

001–004 used a shared output in the paired arms. They do not validate the corrected
separate-task formulation and remain historical diagnostics only. The old training CLI
is retired; its module remains available for reconstructing old checkpoint diagnostics.

## Interpretation and next gates

The warmup screen asks whether training AUROC >=0.70 and validation AUROC >=0.60.
These are inexpensive engineering checks, not publication criteria. All arms
may still run as diagnostics if the screen fails, but their comparison cannot
be interpreted as evidence for or against R&B's general effectiveness.

Validation checkpoint selection makes the reported paired-bootstrap interval
exploratory and optimistic. It does not replace independent model selection,
multiple seeds, trained unimodal baselines, clinical covariate controls,
patient-matched shuffling, or an external/test cohort. Input occlusion is only
an input-use diagnostic, not a unimodal baseline.

Next decisions depend on these observations: strengthen weak baselines; repeat
promising matched comparisons; then evaluate logMAR/age/structural tasks and
generic learned projection/cross-attention. Do not select a medical endpoint
solely because it maximizes R&B's apparent gain.

The first feedback run used Handoff 0.25 and showed immediate overfitting. The
current default is 0.03125. After warmup, backbone stage 4 is fixed and matched
arms train only the classifier, plus bridge parameters where present. This is
an evidence-driven pilot change; it is not a selected final hyperparameter.

## Experiment history and deployment

See [project management](PROJECT_MANAGEMENT.zh-CN.md) for permanent experiment
branches and the single ws checkout. The first two pilots are documented in
[PILOT_REPORT.zh-CN.md](PILOT_REPORT.zh-CN.md). The next fixed protocol is
[experiment 003](experiments/003-cfp-resolution/PROTOCOL.md): original CFP at
96 versus 224, independent unimodal baselines, and two seeds of matched bridges.
New runs enable deterministic algorithms, include epoch-zero checkpoints, and
report fixed-last-epoch metrics alongside validation-selected metrics.

[Experiment 004](experiments/004-fixed-late-fusion/PROTOCOL.md) reuses fixed-epoch
unimodal predictions for equal-weight late fusion, without additional training.
`main` always holds the latest code and records; timestamp branches archive the
previous main before each executable update. ws follows main after running jobs finish.

See the [experiment ledger](experiments/INDEX.md) for all source tags and reports.
Latest completed feedback: 7 training runs / 19 continuations took 11.07 minutes
on one GPU with a 2064 MiB peak. R&B showed no reliable added benefit. A subsequent
zero-training equal-probability fusion reached exploratory validation AUROC
0.6892; paired uncertainty intervals still cross zero. This practical baseline
will be retained in subsequent studies.

## Training independence correction

[005复核](experiments/006-implementation-audit/REPORT.zh-CN.md) identified coupled global gradient clipping and an unvalidated baseline. New training protocols must explicitly set `loss_reduction: "sum"`, `clip_policy: "per_task"`, `warmup_adapt_stages`, and `adapt_stages`. The archived005 protocol is deliberately rejected by the new trainer; use its timestamp source to reproduce the historical run. Run `python tests/check_optimization.py` before new training. No improved performance is claimed from this correction alone.
