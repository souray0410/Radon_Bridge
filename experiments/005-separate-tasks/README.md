# 005: separate native tasks with intermediate communication

Protocol is frozen before launch in protocol.json. This is a single-seed exploratory
experiment, not a significance or clinical validation study. No test set is opened.

256 train / 128 validation participants; CFP224, OCT32x96x96, both eyes. Participant
record-derived glaucoma classification. Each branch has its own ResNet18 task head.
CFP ImageNet initialization; OCT uses inflated ImageNet filters, not OCT pretraining.
Independent warmup: 10 epochs, stage4/head adapted, earlier stages and BN stats frozen.
Matched continuations: 10 epochs, stage3/4, heads and bridge trainable, stage1/2 frozen.
AdamW: backbone 1e-5, heads/bridge 1e-4; batch4; identical order and seed3407.
Warmup checkpoint selected per-task macro-F1; fixed final continuation epoch is primary.
Selected checkpoints (mean of task macro-F1) are supplementary and explicitly labelled.

Arms: independent, bridge stage3, bridge stage2+3, scrambled stage2+3, self-only stage2+3.
M0=[8] CFP / [4,4] OCT; upsilon=(1,1,1/32). S0 from full interpolation support and spacing,
max across group. Stage2 H=32/64, stage3 H=64/128. Linear kernel3 mixer; zero initialization;
no gate, direct residual, adjoint backprojection. The chosen M0 is a small pilot reference,
not a claim of sufficient angular sampling. Operator geometry logs are saved per model.

Report each output's macro-F1, precision, recall, confusion matrix, and auxiliary AUROC.
Paired participant bootstrap intervals compare final predictions, 1000 draws; exploratory,
no multiplicity correction. Validation has already been inspected in older pilots.
Own allocator cap8GiB, own-process stop9.5GiB, training budget15minutes; other tasks untouched.

Acceptance: tests/check_core.py and tests/check_requirements.py must pass on ws before launch.
Execution completed successfully. See REPORT.zh-CN.md and summary.json for results; the protocol above remains unchanged.
