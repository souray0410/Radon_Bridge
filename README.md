# R&B — Radon Bridge

两条完整ResNet18分别处理CFP与OCT，保留各自主干、任务头、CE损失和预测。先独立监督训练至开发集平台期，再恢复各自最佳检查点，通过MHD V4的中间节点接入线性R&B，进行全参数联合训练，BN正常更新。

## 最新结果：三组配对实验已完成

[完整结果与说明](reports/2026_09_05_three_group_final/README.zh-CN.md) · [23页导师汇报PDF](reports/2026_09_05_three_group_final/Radon_Bridge_Three_Group_Advisor_Report.pdf) · [129项结果CSV](reports/2026_09_05_three_group_final/results.csv) · [最终验收](reports/2026_09_05_three_group_final/FINAL_ACCEPTANCE.json)

非中心化SVD、中心化拟合SVD、可学习原生通道编解码三组各36项，另21项无桥/原可学习CM/固定随机QR参照，共129项，全部达到预设平台期。三种子3416–3418、ρ=1/16/1/8/1/4，主组包含标准Radon、自身处理、空间打乱和等宽线性重采样。报告保留负结果、逐种子结果、配对区间与实际成本，不把开发集结果当作独立测试证据。

![Main comparison](reports/2026_09_05_three_group_final/main_comparison.png)

## 当前实现与协议

- stage3、M32、S64；第二阶段backbone LR=6e−5、head/bridge LR=1e−4，batch16。AdamW WD0.01，各分支与桥分别裁剪5。
- 至少8轮；连续6轮无超过0.001的实质改善判平台；3轮停滞LR×0.3；最多60轮仅保护上限。按两任务macro-F1平均值选择共同检查点。
- 默认`learned_projected`：Radon后展平C×M通道，再进行可学习压缩/恢复。与新增`learned_channel`的Radon前逐点C→r编码、反投影后r→C恢复不同。
- 固定通道版本支持非中心化SVD、中心化拟合SVD和随机QR；基为持久化buffer。中心化仅改变基拟合统计量，运行时不减/加均值。固定版本桥内只有中间卷积可学习。
- 几何使用固定EEM、Householder Radon与普通直接反投影；中间kernel=3线性卷积，无偏置、零初始化。桥内无激活、门控或BN，残差写回原MHD Node ID。
- M/ρ支持逐来源配置；当前快速实验两来源使用相同值。固定通道与可学习原生通道版本ρ表示维数比例，r=max(1,floor(ρC))、h=rM；不是能量阈值。

生效要求及历史覆盖关系见[REQUIREMENTS.md](REQUIREMENTS.md)，配对与展示标准见[报告约定](experiments/2026_09_05_13_32_31/REPORT_CONTRACT.zh-CN.md)。训练源码归档于时间戳分支`2026_09_05_13_32_31`（35675f4）；之后的结果文档提交不改变训练实现。

## 数据与运行边界

正式代码在GitHub，训练机器为`ws02`。1264训练、296开发验证；开发集参与过任务/配置筛选，所有结果均为探索性证据，不读取测试集。数据、模型与参与者级预测保留于远端`/data/mengh/RadonBridge`，GitHub仅同步代码、协议和不含参与者标识的汇总。

GPU时长不限但持续记账；每卡本项目≤10 GiB，不影响其他LOOK任务。未平台、OOM或失败均不能算完整实验，不能自动改变batch或收敛标准。代码和诊断验收与医学有效性结论分别解释。
