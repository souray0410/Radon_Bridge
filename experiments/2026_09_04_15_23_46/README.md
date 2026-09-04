# 2026_09_04_15_23_46

新的全网络联合训练实验版本，时区Asia/Riyadh。GitHub main保持最新，分支2026_09_04_15_23_46保存本次启动快照；ws仅一份/home/mengh/RadonBridge代码。新结果写/data/mengh/RadonBridge/runs/2026_09_04_15_23_46，不续写旧冻结实验目录或继承其检查点。

- protocol.json：5种子、7臂、16epoch、H1/8为主，全部参数联合训练。
- PLAN.zh-CN.md：预算、初始化与报告规则。
- METHOD_CLARIFICATION.zh-CN.md：几何回填、可学习滤波与低秩/1x1矩阵关系。
- acceptance_h8/h16/h32.json：全解冻梯度与资源验收。
- backprojection_audit.json：独立解析投影及实际形状伴随验证。
- superseded_run.json：旧部分解冻任务中止溯源；旧性能不作为当前主方案证据。
- run.sh：在ws运行的新启动入口。是否已启动以外置status.json为准。

目前可报告的历史性能在../008-expanded-validation/QUALIFICATION_REPORT.zh-CN.md与../007-combination-sweep/REPORT.zh-CN.md，均不属于本次全联合训练结果。测试集290人本阶段不读不评估。

最新拓扑修订见TOPOLOGY_REVISION.zh-CN.md；旧验收文件不能冒充新的原位level拓扑已通过。
