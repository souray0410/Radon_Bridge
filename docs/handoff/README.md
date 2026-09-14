# Radon_Bridge — 维护与交接

最近核查：2026-09-14 08:46（Asia/Riyadh）。本页是有日期的交接快照，不是实时监控。

## 职责与当前研究

完整任务网络通过Radon空间交换特征，保留各自标签、任务头和预测。固定基版本仅中间卷积可学习；原网络是否更新由明确的全参数/冻结实验臂决定。基础模型与项目训练独立，均使用各自准确的MHD_Framework V4锁。

眼科当前优先；其他器官数据准备与合格基础模型训练可提前，方法实验承接眼科结论。多器官UKB仍属于同一队列，不能称独立外部数据集。test仍封存。

## 版本和部署必须分开

| 对象 | 核查基线 |
|---|---|
| 准备分支 | `implementation/ukb_complete_20260913`；科学修复基线 `9d50f19` |
| 线上调度快照 | `radon_roles_a730653`；读取实际绑定，不能拿准备分支覆盖 |
| 框架依赖 | 准备分支 `framework.lock.json` 锁定 `c0a27ab`；旧worker保留自己的锁 |
| 云端验收 | [9d50f19检查成功](https://github.com/souray0410/Radon_Bridge/actions/runs/34810493480)：45项通过、1项因无双GPU跳过，CPU双进程四更新一致 |

读取main上的本页不代表main包含完整准备代码。完整多来源研究的92526个逻辑位置已准备登记，不能称为92526项已派发训练。实际新增数还需来源核验和去重。

## 运行状态、缺口和下一步

08:46读取的线上状态更新时间为08:43：旧筛选清单14个候选，6个接受；旧项目计划201个位置，实际项目队列0任务、0接受，等待配对三种子父模型。它不是完整多来源新协议的进度。

准备代码包含多来源网络、基/算子/连接掩码、独立标签、训练恢复、矩阵和部分统计。仍需父模型完整接受、真实输入六网络资源验收、调参与宿主case物化、接受记录衔接、多卡生产派发、全研究汇总。不得以旧状态中的空 `remaining_implementation` 字段宣称新协议无缺口。

下一步先在Ibex通过一个完整参考匹配组的组装、训练、恢复、诊断和报告验收，再滚动扩展至已批准矩阵。保持原batch、精度、BN、收敛与选优规则；OOM、失败、未平台分别隔离，不放宽规则。两/六网络的主要指标是各任务macro-F1等权平均，各原始分支单列；不同疾病概率不融合。

依据：[完整协议](https://github.com/souray0410/Radon_Bridge/blob/implementation/ukb_complete_20260913/docs/ukb_complete_research.zh-CN.md)、[实现缺口](https://github.com/souray0410/Radon_Bridge/blob/implementation/ukb_complete_20260913/docs/ukb_complete_implementation_status.zh-CN.md)、[逐部分验收表](https://github.com/souray0410/Radon_Bridge/blob/implementation/ukb_complete_20260913/docs/acceptance_gates.zh-CN.md)。

## 已知失败与防回归

- `e1708d1`曾因评价/诊断隐式导入其他仓库的 `expanded.native.metrics` 而CI失败。`9d50f19`已移为本项目指标，20组输入与原定义完全一致；不添加外部PYTHONPATH掩盖缺失依赖。
- 历史ws02单元46项通过是对应环境证据；额外CUDA DDP探针未完成，不算CUDA DDP通过。后续完整GPU验收直接在Ibex执行，CPU Gloo通过不能替代GPU通信/多卡恢复验收。
- 新代码通过单元测试不等于完整研究已上线，也不改变历史模型或结论。

## 如何刷新状态

授权Ibex操作根：`/ibex/project/c2377/souray/home/mengh/operations/2026_09_10_11_11_31`。
先读 `radon_bridge_active_workflow.json`，再读其控制器/派发配置、`output`的状态、队列、实际Slurm step及日志。完整研究准备目录与旧线上队列分别核对，不按目录名称判断已接受。绑定和状态读取均只读，不因新会话重复启动控制器。

## 跨项目调用

- [Model_Training交接](https://github.com/souray0410/Model_Training/blob/main/docs/handoff/README.md)：按完整配置身份接受父模型；2D OCT不能作为3D父模型。来源键区分疾病、模态和架构，记录Node映射及检查点SHA。
- [LOOK交接](https://github.com/souray0410/LOOK/blob/main/docs/handoff/README.md)：可复用准确匹配父模型，项目权重/状态各自保存；LOOK最终宿主指标与本项目分支平均不混用。
- [MHD版本说明](https://github.com/souray0410/MHD_Framework/blob/release/v4/docs/installation.md)：锁准确提交，不升级浮动main，不让模型库依赖本项目。

## 验收、资源与维护责任

GitHub干净环境代码检查后，在Ibex的实际allocation做完整输入/算子/更新/恢复/诊断/报告验收；ws02不是必经步骤。新验证独立attempt且先通过资源准入，保持健康旧任务和不可变源码。所有新GPU申请至少48小时，资源和并发遵循[共享规范](../../workspace/GPU_EXECUTION_STANDARD.md)。

每次部署、验收失败/修复、协议调整、阶段完成或阻塞变化后更新本页和证据链接。明确日期及计划/代码验收/环境验收/部署/接受结果，不假装实时。跨项目读取先刷新，不能凭README修改他方队列。原始数据、参与者标识/特征/预测及受限权重不上传GitHub；历史研究记录不改写。
