# 验收覆盖与发布门槛

验收绑定准确代码SHA、依赖和配置；“测试存在”“本地通过”“云端通过”“已部署”分别记录。准备分支通过CPU检查不代表完整研究已上线。当前实施缺口见[实现状态](ukb_complete_implementation_status.zh-CN.md)。

| 部分 | 自动检查或独立验收入口 | 能说明什么／仍需什么 |
|---|---|---|
| 安装、布局、MHD来源 | Research checks；manage.py check/verify-env/test | 干净安装、锁定源码与单元测试；不依赖开发者的其他仓库路径 |
| 投影、反投影、掩码、来源写回 | test_projection、test_operator、test_group_radon、test_native_group | 合成数值性质；真实各架构完整输入还需资源验收 |
| 训练与恢复 | test_group_lifecycle、test_group_placement、runtime_ddp | 梯度、更新、冻结、状态恢复与双进程；多GPU测试在CPU CI中明确跳过，不算通过 |
| 指标、配对扰动、统计 | test_binary_metrics、test_group_pairing、test_group_report、test_project_report | 固定指标定义、参与者关系及统计实现；不代表科学效应成立 |
| 矩阵、选择、去重、派发 | test_complete_matrix、test_complete_registry、test_native_priority、test_autoresearch | 已实现接口；调参/宿主case编译、完整接受链和多卡部署仍按实现状态验收 |
| 数据、父模型、历史重放 | cohort_audit、group_case及每个父模型的接受记录 | 必须核实清单SHA、标签、预处理、完整模型和预测重放；不由合成单元测试代替 |
| 完整输入GPU资源 | 独立前向/反向/更新/保存/恢复/诊断预检 | 逐资源类保留实测峰值、设备及准入记录；运行源码保持不可变 |
| 研究完成与test | 匹配组完整、平台证据、模型/比较/数据锁定 | 所有门槛满足才允许test；失败、不可行、未平台不能标完成 |

## 2026-09-14 CI失效修复

原提交e1708d1的云端检查出现两项ModuleNotFoundError：多来源评价与配对诊断从外部Model_Training的expanded.native导入指标。本地验收环境额外提供该仓库，掩盖了独立安装问题。

修复将相同二分类指标公式和字段放入本项目evaluation.metrics，并同步替换双来源评价和原诊断中的同类导入。训练、标签、选优和历史结果不改变。重新验收必须移除外部训练仓库PYTHONPATH，再检查准确修复SHA的GitHub结果。

真实数据适配器显式加载的固定训练快照是另一类依赖：继续校验来源和数据契约，不通过随意添加目录来绕过缺失模块。
