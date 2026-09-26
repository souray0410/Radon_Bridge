# Modern CFP2D/OCT3D communication: MMTM task adaptation

The approved R&B matrix uses CFP ConvNeXt-Base or DINOv2 ViT-Base, true 3D OCT SwinUNETR encoder, cataract/glaucoma/macular degeneration, and complete MMTM/MBT methods. Frozen-backbone and full-finetuning regimes are separate identities. B-scan 2D weights do not replace the 3D branch. The first seed is 3416, followed by 3417/3418 independent of outcome.

## Author mechanism and current implementation

`ModernMMTMHost` is the new multi-stage classifier, using corresponding stages 2/3/4, spatial pooling, concatenate/squeeze/ReLU, separate sigmoid gates and mean branch logits with one classification loss. Gate scale and initialization follow [author code at 1c81cfefad5532cfb39193b8af3840ac3346e897](https://github.com/haamoon/mmtm/blob/1c81cfefad5532cfb39193b8af3840ac3346e897/mmtm.py). The medical task, backbone pair and observed-eye participant pooling are explicit adaptations; this is not a reproduction of the original action-recognition dataset.

The historical `mmtm` single-site identity-initialized adapter is unchanged. The new `mmtm_author` identity uses sigmoid scale 1 and normal Linear initialization. No historical result is relabeled as a complete modern method.

## Matched scientific execution

For every accepted A, compare selected A, continued A, A+Radon and A+ordinary linear bridge under the same data, initialization ancestry and registered optimization budget. Corresponding feature levels are enumerated; arbitrary cross-level pairs are not part of this package. Report measured cost/parameter differences rather than claiming exact budget equality without evidence. Frozen and finetuned regimes must retain their own parent identities. Train/dev are full cohorts and test stays sealed.

## Evidence and remaining deployment gates

2026-09-26 CPU PyTorch 2.8: 28 method/model tests passed, including author-formula output/input/parameter gradients, native versus V5 continuous three updates, optimizer/model reload, and frozen parent parameters/BatchNorm buffers unchanged from initialization. FP32 rtol=1e-5, atol=1e-6; author float64 gradients 1e-12. This is bounded CPU engineering evidence, not actual modern-pair GPU qualification or research results.

Next: qualify actual modern 2D/3D parent shapes, memory, gradients, save and new-process resume on Ibex under existing claims; then deploy full cohort training and matched successors. MBT, modern parent coverage, additional Radon/linear augmentation on this complete multi-stage host and independent scientific acceptance remain open. Do not hot-edit healthy legacy runs or deployed source snapshots.

## Training-entry integration increment

The multi-stage host now reuses NativePair's source-owned optimizer groups, gradient clipping and node identity interface. In frozen mode only communication parameters enter optimizer groups; in finetuning mode every trainable parent/head/communication parameter belongs to exactly one group. Parent node maps remain available for migration and downstream binding. Both regimes have targeted coverage; the related CPU suite is now30passed. These are tiny 2D/3D graph fixtures, not the full ConvNeXt/Swin volume GPU lifecycle.

## 2026-09-26 pre-deployment semantic review

The actual complete MMTM classifier is the **mean of branch logits**, so its development macro-F1 selects A and matched successors. The legacy two-independent-head mean-F1 remains a separately named diagnostic. Evaluation now saves the joint prediction as well as both branches; it must not substitute mean branch F1 or mean probability fusion for the registered joint classifier. This mismatch was found in the source review before any modern scientific update. FP32 prediction replay uses rtol1e-5/atol1e-6. Tiny fixture coverage checks that the different selection rules are distinguishable.

Multi-site Radon/linear additions and strict selected-A transfer now have CPU output/gradient/three-update/recovery tests. They are implemented but still need actual full-size GPU qualification and the complete matched-training execution package.
