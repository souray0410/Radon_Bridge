# 待当前队列完成后追加：2026_09_05_10_09_52

中心化拟合基36项配对补充，最终93项结果。当前2026_09_05_09_29_16继续执行；新版本仅在前序报告完成、项目锁释放且预检通过后部署。

# 当前追加工作：2026_09_05_09_29_16

三种子×三档ρ，57项完整配对结果；18项核验复用，39项第二阶段＋1次双分支预训练。旧五种子队列因诊断OOM已停，3419/3420取消。新隔离版本修复诊断并做完整预检后部署。状态以新目录 queue_status.json 为准。

# 当前追加工作：2026_09_05_09_03_55

五种子八臂机制验证：新增32项第二阶段＋3项独立预训练，复用8项历史结果。先做GPU预检，再部署执行；状态以新运行目录queue_status.json/status.json和报告为准。详见[计划](2026_09_05_09_03_55/PLAN.zh-CN.md)。旧44项已全部完成。

# 当前追加工作：2026_09_05_01_29_50

准备固定SVD通道压缩12项配对实验，等待现有32项全部结束后拟合与GPU验收，通过后自动部署运行。未声称当前已通过GPU验收。详见[计划](2026_09_05_01_29_50/PLAN.zh-CN.md)。

以下保留前轮记录：

# Current experiment

Active: `2026_09_04_23_54_10` — R&B (Radon Bridge).

- Backbone LR: 3e-5 and 6e-5; head/bridge LR: 1e-4 / 1e-4.
- Fixed stage3, M=32, S=64; uniform rho=1/16, 1/8, 1/4.
- Seeds3416/3417: each LR has three rho arms plus its own no-bridge control (16 fresh trials).
- Existing independent best checkpoints; batch16, full fine-tuning, BN updates and validation plateau rules unchanged.
- GPU time budgets explicitly removed by the user. Continue actual time accounting; keep the 10 GiB per-GPU project memory limit and other LOOK workloads.

The lower-native-LR study `2026_09_04_22_32_30` remains historical negative evidence, not an active default. Per-source M/rho support remains available; this study uses equal scalar values.

ρ=1/2 was deferred by the user after exceeding the 10 GiB profile limit; it is excluded from this experiment matrix.


Queued follow-up: `2026_09_05_00_05_58` adds Gaussian projection, spatially scrambled Radon, self-only and pooled communication for both backbone LRs and seeds (16 additional trials). Match standard R&B and no-bridge references from the active study at M32/S64/rho1_8; pooled M/S are inapplicable and capacity differs. Combined final report: 32 trials. Deploy only after the active remote queue releases its lock.
