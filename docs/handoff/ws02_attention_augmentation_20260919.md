# WS02 cross-attention A → A+bridge 三臂

- run：`attention_teacher_20260919_v1`
- scientific source：`9cbb61648ba6ec8d0a699e36581f164415c9d26f`
- framework：`3559caa8d596d4438533a69d39d8a2c32eb21e46`
- 状态：3/3 complete，0 failed，manager 已退出，两张 WS02 GPU 回到 15 MiB / 0%。
- 独立审计：`99ab7c52134d14d34f8501b503374d8e89686595a8d398725b0c278460fbeb47`。

三个第二阶段臂都从同一 accepted cross-attention best7 起点开始，初始预测逐数组精确重放。由于该 A 的 CFP/OCT 原生状态与原父模型各 122/122 张量均已变化，本包没有复用旧 SVD 基，而是从冻结 A 的 train Stage3 pre-write 特征重新拟合共享未中心化 SVD。

结果的两分支平均 Macro-F1：continue 69.256%，Radon 71.114%，普通通信 70.067%。Radon−continue +1.858pp、ordinary−continue +0.811pp、Radon−ordinary +1.047pp；三项 ordinary / simultaneous 95% 都跨 0。Radon 两分支的 F1、AUROC均提高且 log-loss 都降低，但这仍只是单种子、同一开发集选择后的探索性证据。

本包不重启；后续只消费冻结产物做报告/跨方法汇总。test、Ibex owner 与其他健康任务未改变。
