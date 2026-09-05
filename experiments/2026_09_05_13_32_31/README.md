# R&B: learned native channel encoder and decoder

User-authorized addition: 36 second-stage trainings after the centered-fit queue finishes. Three seeds3416–3418, rho1/16,1/8,1/4; Radon/self/scrambled/linear_resample. Reuse93 references, total129 results. No extra native pretraining or basis fitting.

The new `learned_channel` path uses independently learned bias-free pointwise C-to-r and r-to-C convolutions around geometry and the existing zero-initialized kernel3 mixer. Initialize encoder/decoder with the same QR as the fixed random control (transpose/forward), then untie and train without orthogonality constraints. Native networks continue full training and BN updates. Same stage3/M32/S64, LR6e-5/1e-4, batch16, plateau policy and 10GiB per-GPU project limit.

The deferred runner accepts the completed93-result report, obtains the project lock, profiles all four modes at rho1/4 with full forward/backward/optimizer and read-only full diagnostics, then deploys only on success. Failures require attention; no automatic hyperparameter changes. Source must remain immutable during active queues.

Entry points: `scripts/run_learned_channel_study.py --root <unique-run-directory> --deploy-repo /home/mengh/RadonBridge`; `scripts/collect_learned_channel_report.py --root <remote-run-directory> --destination <local-directory>`.

Reporting includes all129 results, learned-channel-minus-matched-reference comparisons, shared-index participant bootstrap10000, individual seeds and new3418, full parameter/cost/epoch/hash provenance. Learned encoder amplification is separated from orthogonal row-space energy/variance retention. Current evidence remains development-set exploration. See REQUIREMENTS.md for the accepted protocol and cpu_acceptance.json for completed checks. GPU acceptance remains deferred.
