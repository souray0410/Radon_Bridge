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

- [MHD_Models交接](https://github.com/souray0410/MHD_Models/blob/main/docs/handoff/README.md)：按完整配置身份接受父模型；2D OCT不能作为3D父模型。来源键区分疾病、模态和架构，记录Node映射及检查点SHA。
- [LOOK交接](https://github.com/souray0410/LOOK/blob/main/docs/handoff/README.md)：可复用准确匹配父模型，项目权重/状态各自保存；LOOK最终宿主指标与本项目分支平均不混用。
- [MHD版本说明](https://github.com/souray0410/MHD_Framework/blob/release/v4/docs/installation.md)：锁准确提交，不升级浮动main，不让模型库依赖本项目。

## 验收、资源与维护责任

GitHub干净环境代码检查后，在Ibex的实际allocation做完整输入/算子/更新/恢复/诊断/报告验收；ws02不是必经步骤。新验证独立attempt且先通过资源准入，保持健康旧任务和不可变源码。所有新GPU申请至少48小时，资源和并发遵循[共享规范](../../workspace/GPU_EXECUTION_STANDARD.md)。

每次部署、验收失败/修复、协议调整、阶段完成或阻塞变化后更新本页和证据链接。明确日期及计划/代码验收/环境验收/部署/接受结果，不假装实时。跨项目读取先刷新，不能凭README修改他方队列。原始数据、参与者标识/特征/预测及受限权重不上传GitHub；历史研究记录不改写。

## 2026-09-14 名称与框架对应更新

训练仓库已从 `Model_Training` 更名为 `MHD_Models`，仍为私有仓库并保留完整历史。新引用使用 `souray0410/MHD_Models`；旧源码目录、运行ID、检查点和框架锁不因改名重写。详细规则见 [MHD_Models版本与发布契约](https://github.com/souray0410/MHD_Models/blob/main/docs/framework_compatibility.md)。本条只更新名称与依赖说明，不刷新上文训练进度，也不表示新科学代码已部署。


## 2026-09-14 执行衔接修复（上线前核验）

临时共享提交锁竞争改为明确的`waiting_submission_lock`，只在获取flock的位置捕获；真正的fork/Slurm查询失败仍报错，不将所有BlockingIOError隐藏。原工作队列可追加独立、按SHA核验的native feed；同run同spec的多研究引用去重，保持原领取和科学身份。控制器与GPU worker分开验收和部署，旧健康训练不热改。
完整协议的73项父候选与旧14项筛选范围分开登记；新增独立父模型准备控制器，不把92526逻辑位置当成执行任务。复制重复种子之前可按绑定队列复用已有run，仅允许recipe_selection（选入该研究的出处）不同；任何训练、数据、预处理、框架、初始化或种子差异均不复用，原spec不改写。复用仍须完整接受与项目重放。完整方法case编译、真实完整输入验收、多卡及统计接通仍是后续门槛，不因父准备控制器上线而宣称完成。

## 2026-09-14 11:12 控制器交接与长期自审

控制补丁8a168f7；CI34820268581，54项通过、1项GPU测试跳过。新控制入口已在Ibex通过清单/源码核验及启动验证，旧CPU dispatcher无子进程后通过stop文件正常退出，再移交原journal和共享资源锁；未发送GPU终止信号、未取消任何allocation。当前绑定文件指向新控制快照，现有GPU owner继续使用原配置和科学源码，新allocation才采用新feed。现场23个单卡allocation仍运行，四个项目父模型持续更新；这不等于LOOK或Radon_Bridge方法实验已产生结果。

通用自审原则见[RESEARCH_AUDIT_STANDARD](../../workspace/RESEARCH_AUDIT_STANDARD.md)：主动检查假设、实现、证据与检查本身的盲点，不依赖用户发现问题。监控须同时核对完整研究范围和运行事实，不以部分CI通过或进程存活概括整体就绪。OPS/execution_audit_20260914/{config,latest,repair_ledger}.json记录独立门槛、覆盖与修复状态；脚本只读，不替代完整模型/Slurm/真实GPU验收。

完整父准备控制器已独立上线，73个候选中24个接受；合并旧/新feed后81个唯一原生任务均可匹配原训练源码。重复种子严格复用原run，不改科学spec。该分母与旧14个候选分开；完整多来源case编译、真实输入资源/恢复、多卡派发与全研究统计仍未全部验收，不能标成全流程完成。

## 2026-09-14：线性分组补充与留存核查

用户要求将分组卷积纳入机制研究。见[补充设计](../linear_grouping_supplement_20260914.zh-CN.md)。本提交只登记设计、挂载盘点及清理保护条件；未实现分组训练或新增科学队列，未删除检查点，未将来源未确认的MRI作为UKB迁移。G1兼容、跨来源分组、匹配普通通信、无压缩匹配可行域及完整资源验收通过后才派发。原研究范围和健康worker保持原版本。

## 2026-09-14：9月20日阶段交付优先级

见[本周执行安排](../weekly_delivery_20260920.zh-CN.md)。眼科两项目完整参考匹配组优先于继续增加独立模型配置。现场24张GPU运行，但两项目方法接受数仍为0；LOOK旧父筛选50/116，RB旧6/14、完整父准备25/73，分母分开。未来准入策略已在共享锁内改为LOOK6/RB6/model最多12；现有2/2/20 allocation不强停，预留不等于实际已分配。OPS/weekly_delivery_20260920保存原策略和变更验收。三维父模型及完整case/资源/执行接受仍为阻塞；周中检查不以缩短训练或2D替代3D赶交付。心脏5组数据迁移，其他器官及颈动脉暂缓；test继续封存。本文档未上线新的科学worker。

## 2026-09-14：学期投稿硬截止与分阶段授权

用户明确2026-12-31前正式投稿，应尽早；允许首篇完整研究包与729六网络等长期矩阵分阶段，后续实验保留。见[学期交付安排](../semester_delivery_2026.zh-CN.md)。LOOK首篇优先、RB并行阶段推进；11月30日为LOOK内部争取目标，不是未经测量的完成保证。分阶段test入口尚未实现/验收，当前test保持封存。不能把已查看test在后续调参后再次当作未见证据。

## 2026-09-14：阶段扩展复用管理层

用户明确阶段必须可接续、扩展、复用，不能换阶段就重复训练。新增共用workspace/stage_registry.py和STAGED_RESEARCH_STANDARD.md；科学身份与stage分离、文件SHA、原run复用/恢复、不可变阶段与前驱摘要、跨进程锁、失败不自动重训。它仅做管理规划，必须由项目真实接受器/存活检查完成verify_live，未部署全部生产适配器，也不解封test；具体集成和真实模型恢复继续验收。

## 2026-09-14：旧训练版本资源预检与短step退出修复

现场发现两个新Radon allocation在预检导入阶段失败：资源探针假定所有不可变历史trainer均提供training_components/clipped_step和Inputs(recipe)，旧197a8cd不满足。修复在独立探针适配层完成，按精确native源码SHA限制历史适配，保留原优化器/组内裁剪、初始化、数据参数和原run；未知API在申卡前拒绝。现代路径仍调用其自己的函数，不改正在执行的科学源码。

Radon调度补充短worker退出后读取step记录、有限等待Slurm确认退出、清除子step继承的一核请求，并对非0/75退出保留失败和禁止继续盲申的incident hold。陈旧paused状态不能掩盖本次预检失败。13项针对性测试已在Ibex通过；独立GPU恢复预检正在现有allocation内执行，旧健康训练保留。完整GPU接受、准确提交CI、claim恢复和新调度部署分别记录，不能由本段文字认定全部完成。

现场证据：operations/2026_09_10_11_11_31/profile_recovery_20260914；此路径包含受限运行引用，GitHub仅保存代码、规则及非参与者级摘要。
