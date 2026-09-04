# R&B / RadonBridge

Experimental Householder–Radon communication between native 2D and 3D neural
features, represented explicitly with MHD V4 forward and backward levels.

This is a **bounded feasibility pilot**, not a validated clinical model.
The first question is whether the implementation is correct and whether a small
paired experiment produces a useful direction for a subsequent study.

## Current pilot

- UKB participant-level, record-derived glaucoma labels; existing train and
  validation assignments; no test evaluation.
- 256 training and 128 validation participants, balanced within each split;
  both eyes; CFP 96x96 and 32 ordered OCT slices spanning the full scan range,
  each resized to 96x96. This is a coarse 3D input, not central-slice OCT.
- ImageNet ResNet18 for CFP; the same filters inflated into 3D for OCT.
  Inflation averages a 2D kernel along the new depth axis and sums the first
  layer's RGB kernels for a grayscale input. It is not OCT-specific pretraining.
- First three stages frozen, fourth stage and classifier adapted. BatchNorm
  running statistics frozen. This deliberately measures a cheap adaptation
  regime rather than full-network optimization.
- Baseline warmup, then four matched continuations from the same baseline:
  no bridge, Radon bridge, spatially scrambled Radon operator, and self-only
  projection filtering. All use identical participant order and training budget.
- Mixer is bias-free linear Conv1d, initialized to zero; direct residual return.
  Handoff compression follows projection. No nonlinear or adaptive gate.
- Dense reference operators operate on small stage features. nD mathematical
  interface and 1D–4D tests; no claim of scalable arbitrary-dimensional kernels.

## Resource limits

Use one available GPU for the initial run. The pilot limits PyTorch allocator
memory to 8 GiB and stops if own-process `nvidia-smi` usage exceeds 9.5 GiB.
This is a practical guard with headroom, not a hardware-isolated memory quota.
Other GPU users must not be interrupted. Initial training budget: 60 minutes;
24 GPU hours is only the previously agreed outer ceiling, not a target.

## Run

Initialize the pinned MHD submodule and use an environment containing the
dependencies in `pyproject.toml`, including torchvision compatible with PyTorch.
Install this package, or expose the two source roots through `PYTHONPATH`:

```bash
git submodule update --init
export PYTHONPATH="$PWD:$PWD/third_party/MHD_Project"
python tests/check_core.py
python -m radonbridge.data --labels LABELS_CSV --image-root EXPORTED_CFP_ROOT \
  --source-root RAW_UKB_ROOT --output CACHE_ROOT
CUDA_VISIBLE_DEVICES=0 python -m radonbridge.pilot --data CACHE_ROOT --output RUN_ROOT
```

Source media are only read. Participant IDs, selected records, images,
checkpoints and predictions stay on ws, outside Git. Local copies are temporary.
The configuration captures commits, pretraining provenance, hardware and limits;
the data audit records source-label and selected-manifest hashes. Each arm saves
best and last model/optimizer state plus its epoch. Automatic interrupted-run
resume is not implemented in this initial pilot; checkpoints permit explicit
recovery, and existing completed results are never overwritten.

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
