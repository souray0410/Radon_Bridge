# Rhythm Bridge

当前实验：`2026_09_04_20_27_40`。当前协议见 [实验说明](experiments/2026_09_04_20_27_40/METHOD.zh-CN.md) 和 [protocol.json](experiments/2026_09_04_20_27_40/protocol.json)。

两条完整 ResNet18 分别学习 CFP 和 OCT 分类，达到开发集平台期后，加载各自最佳检查点，在原有中间节点加入线性 R&B 并全参数联合训练至平台期。各分支保留头、损失和预测，主指标为各自 macro-F1。

当前桥接口为 `nodes, M, S, rho, mode`，初测 `M=16, S=64, rho=1/8`。压缩保留数由实际输入宽度计算，不再固定512。固定对跖 EEM → Householder Radon → 展平 → 独立线性压缩 → 拼接线性卷积 → 独立恢复 → 普通直接反投影 → 原节点残差。固定几何没有训练参数。

正式代码在 GitHub main，远端唯一当前工作副本为 `ws02:/home/mengh/RadonBridge`。数据、检查点和参与者级预测只保存在 `/data/mengh/RadonBridge`。两张GPU各最多10GiB；本轮剩余预算继承旧账，换时间戳不重新计费。

当前入口为 `scripts/run_integer_experiment.py`（文件名保留，实际要求新v3协议），汇总入口为 `scripts/summarize_integer_experiment.py`。无效历史结果已从当前版本移除；远端剩余旧诊断与验收因自动审批拒绝永久删除，已可恢复地归档，不再用于结论。纯时间戳 Git 分支保留源码历史。`legacy_*` 仅为旧代码依赖与溯源，不可启动旧试验。

验收包括 `check_integer_bridge.py`、`check_two_stage.py`、`check_accumulation.py`、`check_convergence.py` 和 `check_scheduler.py`；结果以本实验目录的验收文件为准。实现正确与临床有效分别判断；目前没有本协议的效果结论。
