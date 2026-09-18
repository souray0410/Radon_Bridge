# Radon_Bridge 小队列分组线性通信比较

本页固定SVD r32/M32/S64/k3，比较G=1/2/4/8/16；每个G均配同G普通线性重采样。另见[项目总览与覆盖](../README.md)、[已完成24项线性分解](../factorized/README.md)、[未完成的分组及机制](../coverage.md)。

ws02 GPU1；单种子3416。G=1两臂仅在配置、父模型、SVD基和接受文件SHA一致时严格复用；G=2/4/8/16为8个新执行臂。
CFP为224×224二维；OCT是旧数据32×96×96三维体积。不是Ibex的32×224×224大队列。
两条独立ResNet18专家；同一对父权重，Stage3通信；真实batch16，至少8轮、最多60轮、patience6；停止规则未为周报缩短。

完整分组匹配包：已齐全（manager、profiles 与独立 audit 均接受）；运行状态：complete。

|方法|CFP分支F1|OCT分支F1|分支均值F1|最佳/停止轮|来源|
|---|---:|---:|---:|---|---|
|G=1 Radon|72.30%|71.28%|71.79%|11/17|G=1 strict accepted-core reuse; no new training|
|G=1 普通通信|69.59%|65.46%|67.53%|9/15|G=1 strict accepted-core reuse; no new training|
|G=2 Radon|68.71%|70.26%|69.49%|6/12|new approved grouped-linear single-seed execution|
|G=2 普通通信|66.89%|65.14%|66.01%|0/8|new approved grouped-linear single-seed execution|
|G=4 Radon|70.60%|66.74%|68.67%|8/14|new approved grouped-linear single-seed execution|
|G=4 普通通信|66.89%|65.14%|66.01%|0/8|new approved grouped-linear single-seed execution|
|G=8 Radon|67.12%|65.48%|66.30%|6/12|new approved grouped-linear single-seed execution|
|G=8 普通通信|66.89%|65.14%|66.01%|0/8|new approved grouped-linear single-seed execution|
|G=16 Radon|67.89%|64.52%|66.20%|8/14|new approved grouped-linear single-seed execution|
|G=16 普通通信|66.89%|65.14%|66.01%|0/8|new approved grouped-linear single-seed execution|

分支均值不是概率融合后的单模型分数，不与LOOK的融合输出F1混排。完整后自动生成10,000次配对bootstrap普通与同时区间；单种子且dev参与选择，不能推出稳定泛化优势。
完整配置/状态和聚合指标：[current.json](current.json)。参与者预测及权重不上传GitHub。

## 完整匹配后的差异（百分点）

每项为同一G下Radon减普通线性重采样；完整组才给配对区间。

|对照|差值|普通95%区间|五项同时95%区间|
|---|---:|---|---|
|G=1 Radon − G=1 普通通信|+4.26|[+0.17, +8.29]|[-1.00, +9.52]|
|G=2 Radon − G=2 普通通信|+3.47|[-0.70, +7.61]|[-1.88, +8.82]|
|G=4 Radon − G=4 普通通信|+2.66|[-1.19, +6.48]|[-2.32, +7.63]|
|G=8 Radon − G=8 普通通信|+0.29|[-3.37, +3.83]|[-4.36, +4.94]|
|G=16 Radon − G=16 普通通信|+0.19|[-3.04, +3.22]|[-3.88, +4.26]|

选回初始父模型的设置：G=2 普通通信、G=4 普通通信、G=8 普通通信、G=16 普通通信。它们已按停止规则训练，最终选模回到第0轮；分数相同不能解释成方法等效。

区间是固定已选模型下的参与者重采样，未计入训练种子波动及开发集选择偏差。

分组只限制桥内的直接连接：每组同时含CFP/OCT来源、保留通道内全部M方向，并在返回前逆排列。跨组直接梯度为零不代表完整网络彼此独立。无压缩分组的等参数/近似等计算比较仍是单独待验收问题。

## 阅读图表前：缩写和参数

CFP（Color Fundus Photography）为彩色眼底照片；OCT（Optical Coherence Tomography）为光学相干断层扫描。Stage3是第3个残差阶段后的通信位置；r=32是每分支保留通道方向数，M=32是投影方向数，S=64是每方向采样格点数，k=3是一维卷积核宽。
SVD用训练特征确定固定通道方向；随机QR不按信息重要性排序；可学习通道映射额外更新编码/解码参数。全局分解中间通道数大写R（另一个研究包）不是这里的小写压缩秩r。
批准范围、未完成项和下一步见[覆盖清单](../coverage.md)，不能把局部包完成当项目所有情况完成。
