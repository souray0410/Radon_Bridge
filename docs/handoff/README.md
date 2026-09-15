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

## 2026-09-14：长期自动纠错责任

用户明确自动纠错必须成为通用长期规则，不能只适用于本次故障。AGENTS及共享RESEARCH_AUDIT_STANDARD已规定：已授权流程必须从发现推进到诊断、版本化修复、验收、安全恢复和真实进度复查；不能停在报警、CI通过或调度器重启。未关闭问题保留证据、责任主体、下一步及可执行自动续接。保护健康任务和原科学标准，限制瞬时重试，不盲重启确定性错误。规范适用于未来模型、数据、框架、项目与CI/资源/评价/报告链路，但不是保证所有故障都能自动解决。规则更新本身不表示任何当前训练或故障已完成验收。

## 2026-09-14 current-format migration boundary

Shared model standard v4: new releases use one canonical artifact format and reader.
MHD_Models main 0859bfe implements the new package and one-time converters. Running
models and research projects retain original immutable snapshots until accepted
transition. See [migration](../model_migration.md); package-layout deployment is not
permission to change scientific protocols or follow main at runtime.


## Permanent contract review gate (2026-09-14)

The shared [research audit standard](../../workspace/RESEARCH_AUDIT_STANDARD.md)
and AGENTS.md now require one current contract and explicit version migration.
This gate covers the whole workflow, not only the model catalog. New consumers
must reject unconverted legacy inputs; old pinned workers finish unchanged.
Migration, downstream replay and release evidence are required before switching.
This documentation update does not certify remaining production migration or
upgrade MHD V4. See the standard for the mandatory review checklist.


## 2026-09-15 project-first resource policy

The user superseded the generic-model reservation: project work and required
parent models have priority; generic models get only unused capacity. The shared
GPU standard now also requires independent configurations, controls and seeds to
be separately claimable for parallel execution. Real scientific dependencies remain.
MHD_Models prepares a CPU demand publisher for the existing budget contract; it
does not replace this project's running source or automatically split old cases.
Live policy deployment and per-arm parallel execution are separate acceptance gates.


## 2026-09-15 revised allocation targets: 10 / 10 / 4

This supersedes the no-generic-floor policy earlier today: preserve four generic
model slots, cap LOOK and Radon_Bridge targets at ten each, and lend every currently
unused project slot to generic models. Targets govern new admission, not forced
termination of healthy existing work. Explicit publisher configuration and live
consumption must be verified. Independent project-arm parallelism remains a
separate task-DAG integration gate, not automatically solved by quota changes.
