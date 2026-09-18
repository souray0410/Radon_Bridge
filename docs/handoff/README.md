## 2026-09-18：同一MMTM宿主加桥三臂已正式接入

新ws02 run2026_09_18_17_04_31_666766：MMTM继续训练／MMTM+Radon／MMTM+普通通信三臂，同种子3416同验收宿主和原停止规则。c0211ac独立源码，11项迁移/报告/覆盖检查通过；第一臂完整GPU恢复预检通过，正式训练已至epoch3，其他按原manager有限顺序接续。原MMTM best_epoch0与原生父状态逐项一致才允许本包复用SVD基。

[配置和边界](ws02_augmentation_20260918.md)；[累计结果](../reports/current/augmentation/README.md)。

## 2026-09-18 09:12UTC：ws02六臂已完整接受

[同配置六臂、普通/同时区间及初始模型选回解释](../reports/current/small_cohort/README.md)。原子队列四新臂自动全部完成，独立SHA与296人F1核验通过；Ibex父模型暂停/新32待资源，未混入本结论。

## 2026-09-18：ws02快速核心包

[单种子六臂协议与实际运行](ws02_single_seed_core_20260918.md)；[累计结果](../reports/current/small_cohort/README.md)。GPU1普通通信已越过预检进入正式训练；Ibex大队列依赖与原队列保持，旧记录按时间追溯。

# Radon_Bridge — 维护与交接

最新接续：[32层独立阶段的获批自动接续](fast32_activation_20260917.md)：9个待批及未来申请绑定已通过实际CPU周期核验；新32正式训练仍等待获批，不能称六臂完成。

最新实施：[2026-09-17独立交付与依赖并行](independent_delivery_20260917.md)。本地检查及Ibex独立CPU检查已通过，正式逐臂派发和真实下游验收仍未完成；下文较早运行快照保留原日期。


最新专项核查：[2026-09-17父模型模态路由修复](parent_routes_20260917.md)。下方9月14日内容保留历史日期；管理修复现已[正式启用](production_activation_20260917.md)，GPU新资源预检在获批时自动执行。

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


## 2026-09-15 resource-lease continuity standard

The shared model/run and GPU execution standards are version 5 and AGENTS.md
requires their continuity gates for training, PCA/SVD, correction and evaluation.
See [the audit](../allocation_continuity.md) for checked production boundaries and
remaining acceptance. Actual new allocation policy stays 48h. Existing healthy
scientific snapshots remain unchanged. MHD_Models centralizes new submissions in
the guarded planned pool; short unguarded legacy writers are retired. Prepared
segment-admission changes do not certify all production recovery paths.

Isolated Ibex management acceptance: 272 tests passed in
`lease_continuity_20260915/validation2/`; the first failed fixture run is preserved.
Shared document hashes match across all three repositories. This source is for
new validated owners; original worker bindings and runtime states were not changed.

## 2026-09-15 afternoon resource handoff

The latest user instruction supersedes the four-card generic minimum: retain two
generic GPUs and distribute other capacity dynamically to ready project work. The
sole deployed publisher is `project_demand_floor2_20260915`; the old publisher is
retired, with GPU workers and allocation owners preserved. Its 15:46 Saudi snapshot
reported LOOK 7, Radon_Bridge 10 and other work 7 (running plus pending), with demand
targets 11/11/2. These counts require a fresh Slurm/role check before action.

The policy implementation passed 13 targeted tests on Ibex, including the two-card
floor, borrowing, reduced limits and uncertain ownership. Healthy project work and
registered project prerequisite models are distinct from generic exploration.
Extra generic workers may yield at a verified checkpoint after a project preflight;
quota publication alone does not transfer an existing allocation. The staged
legacy-to-project handover must verify process identity, source-specific recovery,
old step death, shared claims and actual downstream execution. Never cancel the
allocation owner to rebalance. All new requests remain 48 hours.

## 2026-09-16: Scientific review obligations

Added the identical SCIENTIFIC_REVIEW_STANDARD.md to LOOK, Radon_Bridge and
MHD_Models and linked it from AGENTS and RESEARCH_AUDIT_STANDARD. This documents
mandatory comparison/algorithm/evidence checks; no runtime checker, training
snapshot, scientific result, GPU owner or test access changes in this update.

## 2026-09-16 11:30：年末阶段与每周证据入口

用户批准docs/semester_delivery_2026.zh-CN.md：首阶段309逻辑位置（162核心+147参考机制），既有线性补充另计，729六网络保留后续；眼科阶段接受后进入心脏内部，再进入有临床与时间依据的眼心研究。年内分类，三层迁移分别评价。LOOK年底前投稿目标不改。阶段test必须另验收，本次未解封。

管理源码3036c22经本地/Ibex36针对测试及GitHub35073121989完整61项与两rank检查通过，PR4已合并main612c9ce。CPU controller在固定vsc509-03-l接替旧1071888，科学配置/source pins/GPU workers不改。真实新周期08:23 UTC状态active、native8接受；旧ResNet队列201计划位置，实际项目0登记，等待父模型锁定。它与新309位置登记是不同分母。seed3416不再等待其他父种子；后续种子要求首种子完整case技术/诊断/报告接受，不看分数符号。

证据监控源码a89f392（37针对测试；GitHub35073920455成功）每900秒执行，输出semester_20260916的weekly_latest、positions、incidents、progress_clock和周报事件。按上一周交付快照比较，不按上一监控周期比较。只认可真实进度字段，不以心跳掩盖48小时停滞。它是观察器，不是另一个GPU调度器或万能修复器；既有ukb维护自动化继续诊断、版本修复、验收、恢复和复核。

OPS/semester_20260916保存validation、deployment、monitor_deployment_v2及启动脚本；真实入口从radon_bridge_active_workflow.json读取。controller_source=source_3036c22，semester observer=source_a89f392；仅管理源差异，原GPU科学源保持。原生3D父模型仍在训练，未接受的不能进入正式桥研究。

开放且未宣称完成：独立arm GPU执行DAG（当前旧worker仍整case串行）、三架构真实父模型/运行验收、分解等补充case编译接队、完整阶段统计/test门槛、心脏标签/访视/全局划分语义验收。下一维护应推进独立arm实现和首个合格父组的实际闭环，不仅看队列。源码矩阵不等于全部生产能力。

每周六快照，周日06:45英文PPTX/中文讲稿、09:45刷新、09:55交付，10:00 Asia/Riyadh汇报。KAUST已有2026-09-20准备目录；未生成/渲染的PPTX不标ready。三小时ukb自动化已更新并保留原任务/保护规则。所有新申请仍至少48h、通用model底线2张，按实际配额与资源准入。

### 心脏元数据首次可执行核查

9add346增加data.cardiac_inventory，Ibex三个测试及真实文件核查通过。按已有独立传输receipt核对361888文件/8360335258048 bytes，总量一致；核对6个全量表型/字典文件当前大小与原验收引用。仅解析两份CSV表头：18255列/3426个字段、27750列/7145个字段；不读取参与者行、不读test性能。结果在semester_20260916/cardiac_metadata_audit.json，semantic_data_acceptance=false。标签含义、配对时间、全局split/父暴露、病例数及用户任务锁定仍待审计；不能据字段数量宣称心脏训练已就绪。

## Current status publication

Machine-readable current evidence is [status.json](status.json). Source review and live runtime verification are distinct; this publication does not change scientific jobs or certify unfinished experiments. Update this record after material evidence review.

Private research overview and weekly archive: [PHD](https://github.com/souray0410/PHD). Adopted hub standards: research-standards-v1, 2026-09-16. Existing pinned study/runtime rules remain authoritative for current executions.

## 连贯性规范采用（2026-09-17）

采用PHD `62fd7e08770f3628a048fd59b82da2b9e545afab`，进入[固定累计结果入口](../reports/current/README.md)和[尝试档案](../archive/README.md)。完整规则见[共同契约](../../workspace/RESULT_CONTINUITY_STANDARD.md)。此次仅更新结果管理入口；未刷新运行事实、未迁移全量旧产物、未删除服务器文件，自动累计发布仍须独立验收。

## 2026-09-17 父模型接续与性能审计

见[parent_recovery_20260917.md](parent_recovery_20260917.md)：已修复多架构父清单审计回归及一个旧profiler失败后未回池任务；54/73父模型验收保持原证据，3D尚无accepted。列出1.37–2.04小时完整预检成本和资源限制，未把登记修复称为正式研究完成。

Independent review found and fixed a per-step RAM admission gap before deployment; see [step-memory acceptance](native_profile_step_memory_20260917.md). Code65d98f6 passed44 isolated CPU tests. Management is now activated; fresh GPU qualification is automatically grant-gated, see [deployment](production_activation_20260917.md).

## 2026-09-18 05:34UTC维护

原DenseNet121-3D同run推进到epoch5/updates15064，检查点137秒新，仍未通过父模型接受。CPU控制面短时Slurm查询超时已在后续周期恢复，未重启GPU。共享唯一待批watcher已随LOOK方法优先调整切到`OPS/look_fitting_priority_20260918/owner_v7/plan.json`，其中9个R&B与1个native条目原样保留；R&B科学源和finite32 owner不变。新32仍无正式更新，不能称六臂开始。旧科研结果保留原截止，不用新维护时间刷新其性能证据。

## 2026-09-18：累计覆盖漏报修复

当前统一入口已补[24项分解结果](../reports/current/factorized/README.md)、[六臂](../reports/current/small_cohort/README.md)及[覆盖清单](../reports/current/coverage.md)。旧总入口“暂无合格结果”和源状态“四对照未齐”已修正，历史时间线不改写。显式迁移器只读旧factorized_ws_queue_v1，重核24组receipt/config/artifact/parent/basis SHA及有序dev F1后输出脱敏publication_v1；当前训练不增加旧格式fallback。原权重/断点/预测完整保留，未新训或解封test。发布覆盖门槛与故障注入测试在CI执行，ukb同步前使用同一检查；不是所有科学矛盾都能自动判断。下一阶段和开放范围见coverage，不能把本次补刊算作本周新训练。

## 2026-09-18 核心通道压缩补充实际启动

见[三种通道处理×两几何](../reports/current/channel_compression/README.md)。六臂不覆盖全部机制；QR-Radon已实际profile接受并正式训练，四新执行复用同父模型，原SVD两项只读引用。新active指向channel包，旧core接受结果/权重不改。标签统一CFP/OCT；原科研快照不热改。

2026-09-18报告发布修复：52b469f/385a83a的嵌套f-string在ws02较新Python可运行，但GitHub Python3.11拒绝解析。已提取局部标签变量、全src按3.11语法解析通过，目标环境报告2测试通过；独立CPU报告修复，不热改运行中的52b469f科学快照，不影响训练。新提交CI须按准确SHA核验。

## 2026-09-18 13:40UTC完整匹配核验

通道压缩六项完整匹配已独立验收：SVD、QR、可学习通道的Radon分支均值F1为71.7858/70.2317/69.5840%，普通通信为67.5262/66.0129/66.0129%；SVD−QR和SVD−可学习通道普通区间跨零，7项同时区间全部跨零。四新执行、两项复用；当前三包去重32次执行，不是全项目完成。

ws02本批有限流程已正常结束，GPU空闲；下一个研究包尚未运行。Ibex当前1运行allocation/23待批，两个派发器活跃且周期无错误；LOOK原内存保护故障等待资源，Dense3D暂停，新32无正式更新。不能把管理器健康当GPU科研任务正在运行。
