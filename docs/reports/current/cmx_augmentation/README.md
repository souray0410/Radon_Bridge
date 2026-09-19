# CMX-FRM-init 同A三臂（WS02，单种子）

本组回答的**有限问题**是：从同一个已经验收的 CMX-FRM-init checkpoint 出发，继续训练、加 Radon 桥、加匹配普通通信，开发集结果如何。这个 incoming A 在前一阶段选中了 **epoch 0 恒等 checkpoint**，预测与 no-communication reference 逐数组完全相同；因此本组**不能**被解释成“非退化 CMX 已经成功通信后，Radon 继续提升”，更不能声称复现作者完整 CMX 分割系统。

## 三臂同起点与 fresh basis

三臂的 `initial_predictions.npz` 与冻结 A 预测完全一致。由于二阶段协议要求固定来源，本包从该 CMX-init best0 的 1,264 名 train、Stage3 communication pre-write 特征重新拟合未中心化 r32 SVD；CFP/OCT basis SHA 分别为 `19fea81e…` / `5a264763…`，独立审计核验正交误差约 2.9e-15 / 2.7e-15。

|第二阶段|CFP Macro-F1|OCT Macro-F1|两分支均值|best / stop|
|---|---:|---:|---:|---:|
|CMX-FRM-init continue|66.890%|65.135%|66.013%|0 / 8|
|CMX-FRM-init + Radon|72.632%|72.627%|72.630%|20 / 26|
|CMX-FRM-init + 普通通信|70.265%|64.517%|67.391%|11 / 16|

10,000 次 participant-paired bootstrap 的主比较：

|对比|点差|普通95%|三项同时95%|
|---|---:|---:|---:|
|Radon − continue|+6.617 pp|[+2.512,+10.802]|[+1.725,+11.509]|
|ordinary − continue|+1.378 pp|[-1.687,+4.401]|[-2.215,+4.971]|
|Radon − ordinary|+5.239 pp|[+1.322,+9.234]|[+0.564,+9.913]|

这组中 Radon 的两个同时区间下界都高于 0；普通通信相对 continue 的区间跨 0。**但这是同一 296 人 development 既选 checkpoint 又估计效应的单种子结果**，participant bootstrap 不包含训练 seed 变化，因此仍不能升级成独立 test/多种子结论。

## 解释边界

- incoming CMX-FRM A 是项目组件适配，不是作者完整 CMX 系统；
- incoming A 选择的是 best0 identity，原 CMX residual gate 仍为 0，因而“CMX 自身的非退化通信增益”在这里没有建立；
- 二阶段 Radon/ordinary 不是等参数于 continue：continue 约 1.71M bridge 参数，Radon/ordinary 约 14.29M；Radon 与 ordinary 之间则是匹配预算级对照；
- 两分支均值不是概率融合后的单模型分数；
- test 始终封存。

独立资产/统计审计 SHA256：`38a2f5b723c3e04530d12f787b1ac14f9802e257577aa2928a8519d8a835647c`。当前仅右侧独立审计通过，科学接受仍等待左侧确认。
