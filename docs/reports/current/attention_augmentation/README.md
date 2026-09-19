# Cross-attention 宿主加桥匹配组（WS02，单种子）

这组实验直接回答一个有限问题：**已经训练完成的 cross-attention 交流宿主 A，在同一冻结 A 上继续训练、加入 Radon 桥或加入匹配普通通信，结果如何？** 三个第二阶段臂都从同一 accepted A 起点开始，初始 296 人预测逐数组精确重放原 A；A 本身在第一阶段是 best epoch 7 / stop 13，不是 identity/best0 状态。

## 为什么重新拟合 SVD 基

冻结 attention A 的原生 CFP/OCT 状态与旧父模型逐项比较时，两分支各 122/122 个状态张量均已变化，因此没有复用旧 SVD 基。新基只从冻结 A 的 1,264 名 train 输入、Stage3 **通信写回前**特征拟合；CFP/OCT 分别累计 495,488 / 728,064 个通道向量。独立审计核到两份 256×256 基的正交误差均约 3e-15，且 provenance 绑定当前 attention best checkpoint。

## 完整三臂

|第二阶段|CFP Macro-F1|OCT Macro-F1|两分支均值|best / stop|
|---|---:|---:|---:|---:|
|cross-attention 继续训练|70.946%|67.566%|69.256%|13 / 19|
|cross-attention + Radon|71.959%|70.269%|71.114%|7 / 13|
|cross-attention + 普通通信|67.514%|72.620%|70.067%|8 / 14|

Radon 相对继续训练的点差是 **+1.858 pp**；普通通信相对继续训练是 **+0.811 pp**；Radon 相对普通通信是 **+1.047 pp**。三项 10,000 次参与者配对 bootstrap 的普通和三项同时 95% 区间都跨 0，所以这里保留正向点估计，但**不宣称稳定优越或显著赢家**。

|对比|点差|普通95%|三项同时95%|
|---|---:|---:|---:|
|Radon − continue|+1.858 pp|[-1.660,+5.298]|[-2.314,+6.030]|
|ordinary − continue|+0.811 pp|[-2.535,+4.195]|[-3.207,+4.829]|
|Radon − ordinary|+1.047 pp|[-2.694,+4.793]|[-3.402,+5.496]|

## 分支与概率质量不能省略

相对 continue，Radon 的 CFP/OCT Macro-F1 分别 +1.013 / +2.703 pp；对应 log-loss 都下降（−0.1153 / −0.1115），AUROC 都上升（+0.0158 / +0.0103）。普通通信则出现明显分支不对称：CFP F1 −3.432 pp、OCT F1 +5.054 pp；两分支 log-loss 都下降。两分支均值不是概率融合后的单模型分数。

## 资源与匹配边界

Radon 与普通通信的 profile 都记录 13,110,272 个 bridge 参数，因此两者之间是同预算级的主要几何对照；continue 只有原 attention 的 527,360 个 bridge 参数，所以不能声称三臂等参数。20-update profile 的峰值显存约 7.43–7.54 GiB。

独立审计已经核验：3/3 完成、0 failed；所有保存资产 SHA；296 人顺序/标签；三臂初始 A 精确重放；profiles/resume；fresh basis 数值与来源；sklearn 分支指标；以及 10k bootstrap。审计 SHA256：`99ab7c52134d14d34f8501b503374d8e89686595a8d398725b0c278460fbeb47`。

**接受范围**：WS02、小队列、seed 3416、同一 296 人开发集既用于选模也用于效果估计。本 cross-attention 是项目适配模块，不等于外部论文完整系统；test 仍封存；参与者 bootstrap 不包含训练种子波动。
