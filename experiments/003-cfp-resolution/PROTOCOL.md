# Experiment 003: resolution and independently trained unimodal baselines

Written before new outcomes are inspected. The same 256 training and 128 validation
participants and fixed OCT cache are used throughout; the test set remains unused.
CFP 224 is decoded from originals, not enlarged from the 96 cache. Every reconstructed
96 image must match the old cache pixel for pixel. The only preprocessing change is
the final CFP output size. Stage-3 CFP features become 14x14 rather than 6x6.

Questions:
1. Does CFP 224 improve the independently trained CFP baseline over CFP 96?
2. Does joint 224 improve the matched joint 96 baseline?
3. Does R&B beat no bridge, scrambled geometry, and self-only filtering at either
   resolution, and is its direction consistent across two fixed seeds?

Seven runs are fixed in protocol.json: four paired joint runs (two resolutions x
two seeds), two CFP-only runs, and one OCT-only run. Joint runs each have four
continuations. Backbones use the same pretrained initialization and adaptation
policy as pilot 002. Modality-only models are independently trained MHD graphs.
All runs use ten warmup and ten continuation epochs. No outcome-dependent extension.
Deterministic CUDA algorithms are enabled; seeds change head/bridge initialization
and minibatch order, not participant splits.

Primary pilot comparison: validation AUROC at the fixed final continuation epoch.
Also report validation-selected checkpoint AUROC (including epoch zero) and log loss.
This is an exploratory comparison, not an independent efficacy estimate. Two seeds
are a direction check, not a reliable estimate of training variance. The existing
small balanced cohort and record-derived labels limit generalization. Different
resolution changes pooling discretization and bridge sampling as well as retained
image information; no single mechanism is proven by improvement.

Run sequentially on available GPU 1; 8 GiB allocator cap and own-process 9.5 GiB stop.
Batch wall-time ceiling 25 minutes, individual run ceiling 8 minutes. Keep all
participant data and predictions on ws. Only aggregate reports enter Git.
