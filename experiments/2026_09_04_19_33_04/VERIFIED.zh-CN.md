# 新算法验收记录

验收环境：ws02，PyTorch 2.8.0+cu128，RTX 5000 Ada；正式训练尚未由本文件声明完成。

- CPU：二维M=2/3/4/8/16均使用同一EEM生成器，最大等角间隔误差4.052e-7度。
- CPU：2D–4D原始Radon投影对独立SciPy插值/积分参考，相对误差<=3.59e-16。
- CPU：2D–4D直接BP对显式铺开+逆Householder参考，相对误差<=6.76e-16。
- GPU float32：投影相对误差<=8.95e-8，直接BP<=1.05e-7，满足1e-4门槛。
- 2/3/4个参与者、异构形状/维度/深度、多处不同MSH桥接：原Node和原边端点保持；返回同level；零桥输出逐值一致；MHD/native参数梯度误差0。
- 桥线性误差<=1.24e-16，跨网络输入梯度非零。
- 真实配对batch16、ResNet18、stage2+3双桥、M16/S64/H512：3步全参数训练通过；两主干各stage、各头、各桥都有参数更新。
- 第三步后CFP损失至OCT stem梯度范数8.352；OCT损失至CFP stem为0.39868。
- 真实预检峰值allocated6244.34MiB、reserved8186MiB，NVML进程采样峰值8598MiB（8.40GiB）；未超过每卡10GiB。
- 稳定双桥训练步约0.285–0.287秒，第一步0.646秒；仅为该批次的预检计时，不作为整轮耗时承诺。
- GPU几何+显存预检记账0.123718 GPU分钟，已写入远端ledger，计入新轮60分钟。
- MHD公开backward的样本加权梯度累积：非整除7样本/3+3+1微批对全批梯度误差2.78e-17。
- 调度模拟：两卡派发、实际GPU时间求和、完整比较组预算拦截、禁止覆盖均通过。

原始证据：cpu_acceptance.json、gpu_geometry.json、preflight.json、preflight_ledger.json。
以上是数学/工程验收，不是分类性能或临床有效性结论。
