# R&B grouped completion v1 — 右侧执行完成，左侧独立验收通过

任务ID：`radon-grouped-20260918-completion-v1`  
执行日期：2026-09-18  
当前执行方：右侧 WebCodex  
最终科学验收方：左侧规划/独立审查  
状态：**包内实现、部署、8/8 新执行与右侧 audit 已完成；左侧随后从原始资产独立复核并接受本 WS02 单种子开发集包。**

## 1. 精确版本与运行身份

- 本地分支：`webcodex/grouped-linear-20260918`
- grouped operator：`a01bba6f0579fb10df6f570f91bcd7d41f68fb4f`
- 初始本地 handoff：`c09ac5fe80930108e4a52ea0624d80cbbc80bdb7`
- WS02 supplemental adapter：`450a1227286efc2f87ec38be324ccbebcaedc001`
- WS02 部署源码：source snapshot `450a1227286efc2f87ec38be324ccbebcaedc001`
- MHD Framework：`3559caa8d596d4438533a69d39d8a2c32eb21e46`
- WS02 run逻辑身份：`Radon_Bridge/2026_09_18_22_01_54`
- sequence：`2026_09_18_22_01_54`
- GPU：既有 GPU1 finite manager / 原锁；未另建调度器或重复 claim。
- 最终 live probe（19:34:46 UTC）：WS02 GPU0/GPU1 均 15 MiB、0%；grouped manager 已结束。

## 2. 协议未变

固定 SVD `r=32, M=32, S=64, k=3`，Stage3；G=`1/2/4/8/16`，每个 G 比较 Radon 与同 G `linear_resample`。G=1 只严格复用已接受 core；G=2/4/8/16 共8个新执行臂。seed=3416、1264 train / 296 dev、原 batch/FP32/停止规则不变；test 始终封存。没有增加种子、骨干、比较或改变标签/划分/停止标准。

## 3. 完成条件核对

- 新执行：8/8 accepted；`failed={}`；manager `complete`；`active={}`。
- G=1：2个参考臂严格复用，无新训练。
- 每个 G>1 两臂均有独立 profile、resume、正式训练和 accepted receipt。
- 8个 profile 均：`formal_updates=0`、`checkpoint_update_exact=true`、`node_ids_preserved=true`、`autograd_equivalence=true`、full dev=296。
- 每个 profile 均验证组内跨来源直接梯度非零（13.0）、跨组直接梯度0、逆排列通过。
- 最终 audit：296 IDs/labels 顺序一致；独立 sklearn F1 与 receipt 一致；所有产物 SHA 通过；`test_used=false`。
- publication：`matched_results_complete=true`、`complete=true`。
- 左侧独立复核已通过：89个源码文件、54份科研资产、296人有序ID/标签与F1、8个profile/恢复以及10,000次五项普通/同时区间均重核一致。接受范围仅为本WS02单种子dev包。

## 4. 最终结果（单种子 dev，仅事实汇总）

| G | Radon 均值F1 | 普通通信均值F1 | Radon−普通(pp) | 普通95% | 五项同时95% |
|---:|---:|---:|---:|---|---|
| 1 | 71.7858% | 67.5262% | +4.2596 | [+0.1673, +8.2930] | [-0.9976, +9.5169] |
| 2 | 69.4856% | 66.0129% | +3.4727 | [-0.7042, +7.6069] | [-1.8795, +8.8249] |
| 4 | 68.6699% | 66.0129% | +2.6570 | [-1.1948, +6.4849] | [-2.3156, +7.6295] |
| 8 | 66.3025% | 66.0129% | +0.2896 | [-3.3734, +3.8347] | [-4.3574, +4.9367] |
| 16 | 66.2022% | 66.0129% | +0.1893 | [-3.0441, +3.2188] | [-3.8792, +4.2577] |

G=2/4/8/16 普通通信四臂均按原停止规则真实训练8轮，但 best_epoch=0，最终选择初始状态；相同预测 SHA 不能解释成“方法等效”。G>1 的普通95%与五项同时95%均跨零。区间只对固定已选模型的296名dev参与者做10,000次配对bootstrap，不包含种子波动或开发集选模偏差。

各 Radon 臂：
- G2：CFP 68.7132%，OCT 70.2580%，best/stop=6/12。
- G4：CFP 70.5997%，OCT 66.7400%，best/stop=8/14。
- G8：CFP 67.1213%，OCT 65.4838%，best/stop=6/12。
- G16：CFP 67.8874%，OCT 64.5169%，best/stop=8/14。

## 5. 参数量与直接连接规模

投影后每来源宽度为 `r×M=32×32=1024`，两来源 mixer 总宽度2048，kernel=3。固定 SVD basis 是 buffer；下表参数数是 mixer 的实际可训练 Conv1d 权重。直接 channel-pair 数不乘 kernel，仅用于解释连接稀疏度。

| G | 每个 Radon/普通臂 bridge 参数 | 直接 channel-pair | dense 比例 |
|---:|---:|---:|---:|
| 1 | 12,582,912（同结构公式；严格复用） | 4,194,304 | 1 |
| 2 | 6,291,456（真实 profile） | 2,097,152 | 1/2 |
| 4 | 3,145,728（真实 profile） | 1,048,576 | 1/4 |
| 8 | 1,572,864（真实 profile） | 524,288 | 1/8 |
| 16 | 786,432（真实 profile） | 262,144 | 1/16 |

分组只限制 bridge 内**直接**连接；不能外推为完整网络跨组独立。无压缩 grouped 与 SVD+dense 的等参数/近似等计算问题仍是单独未验收范围，本包没有暗改 r/M/S 或 padding。

## 6. 每臂 profile / resume 证据

| arm | profile receipt SHA256 | resume SHA256 | peak reserved / sampled GiB |
|---|---|---|---|
| G2 Radon | `2b92d348a563da3db40dda1dcbabd8fee0f09b62c9807e7e18cfcaac64151dad` | `ef4c21b86b6825deb12247277d24add54a712cc7e1a4d700a755be580d332118` | 7.383 / 7.805 |
| G2 ordinary | `bf28115a6132ab10bb0f0630e1e1a286f216ebc64da480b26e177ec4c22a1894` | `fd837a052966d346bb120b0e231d607ba3ccb6a0774c2941250654c01d3c9538` | 7.352 / 7.773 |
| G4 Radon | `f30d4dcab90b51edafdb9d6c064a1d46fe2523e268063947ad6a6d0b2ea24585` | `1ed17241c428e3490f15183ad42ba0679cb2f6fea380fa6097a308be57eeb85d` | 7.369 / 7.791 |
| G4 ordinary | `f2f1ff90580dbf7ab12c5497c6405815a08acd1785710ff222b546562f422591` | `1bca80ddd4bf738cf5aa94288919cb701a56fa93e29a25893f5989d0731f2950` | 7.377 / 7.799 |
| G8 Radon | `f28eb690881f655dc7599a20fd6ac32105e7778db20d7a6c13ef47f1dcb74eb5` | `d64e180ee839bc4ded45819097e411a41fd57a2b2773ce6654a0fb7aea6667c5` | 7.340 / 7.762 |
| G8 ordinary | `5ae03da0038d188dc665c7cfce8589bdedca4abccb64ed057fc88dd3dde261e3` | `febf34e4e77554ce1fed082848bdf144f6787d189e20301f85af2db1fe59cc81` | 7.359 / 7.781 |
| G16 Radon | `1c1906217a0cdbc119b8796a7b034050a779b6256535b7ed0349cfbbbedeaa3a` | `956d44e72c75a3fe91b61d671d39ca4dd65c03f94466b4e299b7830dd872eaec` | 7.102 / 7.523 |
| G16 ordinary | `bd4372d53eeed588b3446eff91284d357e7c4533dae555b36d7407aef2db23c4` | `574de32628270f6d22fd841f7c1d9d412ef880bb9a4bc51546498be4fc1dd027` | 7.211 / 7.633 |

所有 profile 的逻辑位置为：
`Radon_Bridge/2026_09_18_22_01_54/profiles/<arm>_<timestamp>/accepted.json`。
结构字段与校验值保留在受限原始audit；公开仓库只发布摘要与SHA。

## 7. 部署、队列与最终报告哈希

原始run逻辑身份：`Radon_Bridge/2026_09_18_22_01_54`

- `queue.json`: `206eb736b04a5a2da8f5def7eefbc4f4184e4a2eee06fb4dc2269be631686f0b`
- `status.json`: `5870e21b398a7f9032cf824c735e99a6d8e66f08e53499987558420a979b32a5`
- `dependencies_acceptance.json`: `d1efdeb02c577dc5d19337957b8e8b8bd3ff2748e1a27ec1e3871821e1b6c256`
- `code_acceptance.json`: `cddde9ebf723d2c1dfe0a0a5989cc79030fdd85519de19af2c700b478adfc498`
- target CPU acceptance：`f863d958d756cae6f5dee6b22fdabb416ee48fc49f7a4fe19c3338a9517f77dc`
- `independent_final_audit.json`: `3f50be5b6f58ebfddf780b08d90172be141bfb0b80d5995746ac726b99b12541`
- `publication/current.json`: `db16c21a70a01cb82d3c580a980f6886fcb99e2513a3a9b6526f6d536db9a285`
- `publication/README.md`: `b079cdca5275e320df3aa3eed90a1ed07a133fd93b9b5f80619273cb5cba5e4b`

仓库公开结果：
- `docs/reports/current/grouped_linear/current.json`
- `docs/reports/current/grouped_linear/README.md`
- `docs/reports/current/grouped_linear/audit_summary.json`（脱敏摘要，不含参与者、权重、凭据或服务器绝对路径）

## 8. 负面证据与边界

- G2/G4/G8/G16 的普通95%区间全部跨0；五项 simultaneous 95% 对全部 G 都跨0。
- 随 G 增加时本次单seed Radon点估计变化可直接从表中读取；右侧不把该点估计序列升级成稳定规律或机制结论。
- ordinary 四个 G>1 均选回 epoch0；这不是“没训练”，也不是不同 G ordinary 方法等效的证明。
- CFP/OCT 是 ws02 小队列输入，不可当作 Ibex 32×224×224 大队列结果。
- test 未读取；单seed/dev选模边界保持。
- 本包没有完成 `uncompressed_grouped_budget_matched`。
- Ibex/LOOK 的独立任务状态不由本包改变。

## 9. 左侧独立验收结论

左侧已从原始资产重新核：
1. run status 8/8、无 active/failed；
2. queue/dependencies/code/CPU receipt 哈希及部署源码 commit；
3. 8个 profile 的 grouped structure 与恢复校验；
4. 10个结果（含G1复用）的296人有序预测、独立F1与模型/预测SHA；
5. 10,000次参与者重采样的5个预登记普通/同时区间；
6. 89个源码文件与部署提交、54份科研资产身份。

上述均通过，因此 coverage 可将 `grouped_linear` 标为 accepted，并登记 `ws02_core/svd → grouped_g1_radon`、`ws02_core/linear → grouped_g1_linear_resample` 两条严格复用。此接受不扩展到Ibex、test、无压缩预算匹配或全项目。下一研究依据见 [grouped_next_evidence_20260918.md](grouped_next_evidence_20260918.md)。
