# 2026-09-04 资格检查监控与下一阶段准备

本次两次SSH到ws的22端口连接超时，未取得资格检查的新状态。
不能判定训练失败或完成，不重启、不修改远端代码、不假定某个配方胜出。

已在本地完成以下待验收改动：

- 固定随机线性投影控制：逐行高斯随机向量，L2范数匹配对应已归一化Radon算子行。
  保留零行、相同P/S/H、相同线性mixer与可训练参数量，回传使用自身转置。
  这是控制组，不称为Radon。固定operator随机种子独立于网络训练种子；5种子复核不覆盖operator抽样不确定性。
- 提供小stage矩阵的谱范数、Frobenius范数、数值秩、非零奇异值有效条件数诊断。
  行范数匹配并不意味着同谱，不能据对照差异排除所有尺度/条件数因素。
- 五种子formal执行分支：支持协议中固定的全部控制组，按各自baseline配对计算差值与bootstrap。
- 启动前检查资格报告哈希、选中recipe一致性及累计240GPU分钟预算；缺少实际资格结果时不得启动。
- stdlib预算/协议守卫测试已通过；PyTorch数值、跨分支梯度及GPU验收尚未执行。

连接恢复后的必做步骤：

1. 先读取exp008_qualification真实状态及summary；若还在训练，保持原源码不变。
2. 完成后在ws运行verify_sweep_results.py，取回聚合资格结果，记录实际分钟数并冻结获选recipe。
3. 运行锁释放后才部署新源码；执行tests/check_random_projection.py和已有独立优化验收。
   新控制若未通过，不得启动正式对照。
4. 按实测epoch速度计算预算。先固定formal_protocol.json：5种子3411–3415、20epoch，
   baseline、两种既定R&B、自滤波、打乱几何、固定随机控制；预算不够时先按PLAN缩减候选并记录。
5. formal协议需包含qualification_summary_path、qualification_summary_sha256、prior_gpu_minutes、
   max_minutes、唯一recipe和formal_arms；正式入口仍为python -m radonbridge.sweep。
6. 监控继续每30分钟，不重复发同一网络故障通知；连接恢复或状态发生实质变化再报告。

没有启动新GPU训练，也没有推断资格结果。远端仍以最后确认的251c703为运行版本，需恢复连接后核实。
