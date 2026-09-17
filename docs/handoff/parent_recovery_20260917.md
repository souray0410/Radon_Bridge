# 2026-09-17：3D父模型接续与吞吐审计

本次只访问既定train/dev证据；未改变batch、BN、优化器、候选清单或停止规则，未读取test。15项OCT3D候选尚无训练验收，正式六臂研究尚未放行。不能把“控制器恢复”写成“R&B实验已完成”。

## 已修复并核验

1. `native_prerequisites.audit`在当前代码中只建立ResNet组，忽略已有catalog的`registered_groups`，会使包含DenseNet/Swin的真实73候选清单产生KeyError。改为共用显式二维/三维架构路线，核验登记组与候选归属；未知架构或错误模态组直接拒绝。完整catalog的原文件和SHA不改写。
2. Ibex独立源码20项针对测试通过，包括完整Controller.tick穿过DenseNet/Swin数据流；真实73候选使用原训练完成验证器重放：18组、54项accepted、0项校验异常，耗时54.31秒。缺失3D模型仍保持等待，测试没有制造接受标记。
3. ResNet50/OCT3D/青光眼加权CE运行`2026_09_10_14_34_24_788954`因旧profiler导入不存在的`training_components`而在正式训练前失败，claim一直是`liveness_needs_review`。已核验51849530及其step3终止、当前Slurm无该allocation、无正式last.pt或accepted；新pinned profiler API核验通过。经原Claims接口恢复为paused，原run/spec/seed不变，仍须完整GPU资源预检。现有dispatcher下游可执行API项由14增至15，无拒绝；尚未重新获得GPU运行证据。

上述claim证据在授权operations目录`radon_parent_review_20260917/claim_recovery_accepted.json`；完整catalog验收在同目录`current_catalog_audit.json`。受限路径与原记录留在服务器，GitHub只存汇总。

## 不能再把慢简单解释为3D

| 已有完整预检 | 墙钟 | GPU峰值 | step内存峰值 |
|---|---:|---:|---:|
| ResNet50 3D，glaucoma，断点接续 | 7361.77秒（2.04小时） | 38.85 GiB | 100.02 GiB |
| DenseNet121 3D，AMD，新运行 | 4947.68秒（1.37小时） | 9.05 GiB | 100.02 GiB |

这些是5次预热、20次更新、完整dev、完整train读取和恢复验证的合并时间，不能直接视作纯训练速度。现实现每次接续都会重做完整预检。下一项合理优化是将资源类全量读取/验证证据按完整身份复用，同时对新断点保留最大眼数、恢复下一步和共存资源核验；**尚未实施或证明提速，不能直接跳过当前门槛。**

进程RSS约2–3 GiB不代表主机只消耗这些内存，step峰值包含页缓存等实际收费内存。当前三张卡各有14CPU/100GiB正式worker和owner，其中两卡还有LOOK工作或验证；不能仅凭显存剩余再塞一个100GiB的3D预检。

八个已有3D暂停训练的claim已是paused，可按原配置恢复；两个Swin3D因资源预检失败保持隔离。健康Dense3D继续运行，其他健康worker和allocation owner未被停止。父控制器、dispatcher恢复与新的allocation接管由共同管理入口负责，本次没有启动第二套申卡器。

## 科学身份与仍缺的验收

`eligible_for_formal_selection=False`属于原`expanded_cohort_native_screening`身份，原设计允许其进入有限父模型筛选，项目另做严格重放与接入验收；不能改写flag，也不能只凭该flag断言模型绝不可用。固定50轮配方的训练完成仍不等于证明收敛，须按其原配方报告，不能冒充平台验收。

当前Dense3D参考配方是FP32、SGD 0.025、有效batch16/微批1、BN训练、固定50轮；改变微批会改变BN统计，不能作为同一运行的无损加速。两轮dev宏F1均49.65%，AUROC48.79%→52.97%，全预测阴性；这是未完成父模型的诊断，不是R&B收益。近期进度epoch3继续推进。

正式研究仍缺：合格3D父模型完成、完整父模型对图/预测重放、实际GPU独立实验臂与恢复验收，以及绑定通过验收的六臂执行入口。修复清单注册不替代这些门槛。

## 后续代码补充：闭合资源证据复用（尚未生产激活）

`runtime/native_profile_reuse.py`与dispatcher现已实现：同一完整spec SHA（含数据清单、架构、输入、微批、优化器、框架及源码）且硬件/软件版本严格一致，完整预检回执和恢复文件SHA核验后，可复用全train读取和全dev覆盖。每次仍跑5次预热、20次新鲜最大眼数更新、验证、当前断点加载与恢复下一步逐位一致；新旧实测峰值取最大值，再核实际Slurm CPU/内存和整卡余量。存在未登记科学companion时拒绝复用，不能用一帧GPU读数替代其峰值。

完整原始回执不改写；新的组合回执明确标记哪些覆盖来自历史、哪些来自本次恢复。首次完整预检记录在同一run的`resource_qualification/full_reference.json`，已有封闭证据可通过明确SHA引用迁入；不扫描目录猜测可用结果。

该补充在Ibex独立源码通过30项针对测试（后续追加非有限数拒绝用例），包括更换spec/硬件/断点、证据损坏、资源不足和科学companion拒绝。**尚缺实际GPU新断点重验与耗时测量，故未换正在运行的管理器或科学worker，不能称已经减少了1–2小时。** 等安全资源准入完成该项后才能激活。
