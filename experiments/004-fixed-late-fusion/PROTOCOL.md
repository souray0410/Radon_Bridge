# Experiment 004: fixed late fusion, without retraining

Motivation: experiment 003 independently trained CFP 224 and OCT models have
similar validation AUROC, while the joint model does not clearly outperform them.
This follow-up is designed after those unimodal outcomes were seen, but before
computing late-fusion outcomes. It is an exploratory diagnostic, not confirmatory.

Use only seed 3407 fixed-last-epoch predictions from experiment 003. Keep the same
validation participants. Primary new model: arithmetic mean of CFP 224 and OCT
probabilities, equal weights fixed at 0.5. Secondary controls: CFP 96/OCT probability
mean and CFP 224/OCT logit mean. No fitted fusion weight, calibration, threshold,
retraining, hyperparameter search, test cohort, or individual data export.

Report AUROC, average precision, cross-entropy, Brier score and fixed-threshold
accuracy; paired bootstrap differences against both individual models and the
trained joint baseline/R&B. These are unadjusted exploratory intervals. A fusion
benefit would motivate stronger late-fusion baselines; it would not validate R&B.
Logit averaging can amplify poorly calibrated predictions. The models were
trained independently and their total parameter count differs from the joint
model: this is a practical baseline, not a capacity-matched mechanism control.

Store source/configuration/aggregate results on this experiment's GitHub branch;
individual predictions remain under the ws result directory. Execute under the
same checkout lock. No CUDA context or GPU allocation is needed.
