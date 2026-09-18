# WebCodex 下一有限研究包：分组线性通信结构/匹配验收

日期：2026-09-18  
负责人：WebCodex 主 agent（本独立 Mac clone）  
分支：`webcodex/grouped-linear-20260918`  
基线提交：`00a1f317e69970595162168e655ff9e8e0e9c681`

## 1. 本轮问题与边界

目标不是重跑已经接受的 ws02 结果，而是关闭当前累计 coverage 明确登记但尚未接受的 `grouped_linear` 实现/CPU 结构验收缺口，为后续由原远端调度负责人执行一个代表配置及其匹配普通通信对照准备可审查代码。

导师问题对应：
- 分组卷积限制直接连接后，桥的收益是否来自“任意稠密跨来源混合”，还是在受限的跨来源组内通信下仍存在？
- 每个 G 的 Radon 几何臂必须与普通线性重采样臂保持相同分组结构，避免把几何差异和连接稀疏度混在一起。
- 不压缩+分组 与 SVD+稠密 的容量/计算匹配属于后续单独可行性核查；本轮不通过修改 r/M/S 或填充通道制造伪匹配。

## 2. 已确认的证据与日期

- `docs/handoff/status.json`：
  - `evidence_cutoff=2026-09-18T08:49:11.006936+00:00`
  - `verified_at=2026-09-18T14:53:07.294070+00:00`
  - `next_acceptance` 明确为：MMTM 加桥三臂已完整接受；下一有限包优先分组卷积结构/匹配验收。
- `docs/reports/current/coverage.json`：
  - `reviewed_at=2026-09-18T10:40:22.703894+00:00`
  - `grouped_linear.state=not_accepted`
  - G 固定为 `1/2/4/8/16`
  - 变体登记为 `fixed_svd_grouped` 与 `uncompressed_grouped_budget_matched`。
- `docs/reports/current/coverage.md` 明确要求：
  - 固定 SVD+分组卷积：每个 G 匹配普通通信；
  - 先实现跨来源分组和逆排列，验证组内/跨组梯度；
  - 再做一个代表配置及匹配对照；
  - 无通道压缩匹配不得暗改 `r/M/S`。
- 历史批准协议 `91beacfdbc3173993dbe84e11a824633b98b28c4:docs/linear_grouping_supplement_20260914.zh-CN.md` 已从本仓库 Git 历史读取。该文件当前 checkout 不存在，因此它是历史批准证据，不是假装为当前工作树文件。
- 该协议规定固定 `SVD r32/M32/S64/k3`，比较 `G=1/2/4/8/16`；每组必须包含各来源，按每来源通道分块并保留组内全部 M 方向，排列为 `(group,source,channel,direction)`，输出后逆排列。不得把按来源拼接后直接设置 `groups=2` 当作跨来源桥。
- 当前 `src/radon_bridge/methods/operator.py` 只有稠密卷积、self 掩码和 `cross_edges` 掩码，没有上述跨来源 grouped Conv1d + 逆排列实现。
- 当前 `src/radon_bridge/studies/research_matrix.py` 的 49 个参考机制臂不包含该独立 grouped 补充；这与 `docs/semester_delivery_2026.zh-CN.md` “独立补充另计”一致，不应把 grouped 包塞入 309/49 臂主矩阵。

## 3. 尚待核实/不得冒充完成的证据

- 原协议称覆盖九个同疾病同架构双来源组、三个种子；本轮不读取或修改远端队列，不宣称这些来源/父模型现在均已具备正式 GPU 条件。
- Ibex 3D 父模型、完整 pair/runtime/恢复、多卡与正式诊断仍按既有 owner 和 handoff 日期推进；本地 CPU 结构通过不能替代正式运行验收。
- `uncompressed_grouped_budget_matched` 的“等参数/近似等计算”可行性尚未完成。本轮只保存实际参数/有效连接的计算接口基础，不自行选择新的匹配规则。
- test 继续封存；不读取 test，不改变标签、划分、停止规则、主要比较或 test 用途。
- 不处理 `Uncertainty_Lab`。

## 4. 本轮固定条件

以下条件保持不变：
- 主科学配置：固定 SVD `r32/M32/S64/k3`；G 只取 `1/2/4/8/16`。
- grouped 结构的每个组都包含所有来源；每来源按 channel 分块，channel 内全部 M direction 不拆散。
- G=1 必须与既有稠密 mixer 保持输出、梯度及旧 checkpoint/state_dict 结构兼容；不能因新增 grouped 支持破坏既有接受结果的恢复。
- grouped Radon 与 grouped `linear_resample` 使用同一 G、r、M、S、k 与父模型/数据身份；唯一几何差异仍是既有 Radon vs 普通线性重采样。
- 首轮不增加非线性、门控、额外卷积层或自动通道洗牌。
- 不把直接梯度隔离外推为“全网络完全独立”。

## 5. 本轮实现

计划在本独立 clone 内完成：
1. 在 `LinearMixer` 增加显式 `group_count`：
   - G=1 走原路径，不新增 permutation state，保持旧 state_dict 键与数值行为；
   - G>1 时构造 `(group,source,channel,direction)` 的固定通道排列；
   - 使用真实 `nn.Conv1d(..., groups=G)`；
   - 输出后应用固定逆排列，再按来源 split。
2. 在 `BridgeExchange` / `attach_group` / `attach_to_nodes` / 配置校验中传递并严格限制 grouped 参数，拒绝与未批准的 self、cross_edges、nested widths、S-axis control 混用。
3. 在 `project_build.bridges_for` 传递 grouped 配置，使 Radon 与 `linear_resample` 能生成严格匹配的 grouped 结构。
4. 新增独立 grouped supplement study helper，只登记批准的 G 与两种几何臂，不修改 49 臂主矩阵。
5. 新增 CPU 单元测试验证：
   - G=1 state_dict / 输出 / 梯度兼容；
   - 每组包含两来源；
   - 组内跨来源直接梯度非零；
   - 跨组直接梯度为零；
   - 排列后逆排列恢复原来源通道顺序；
   - 非法 G / 非整除 rank / 未批准组合拒绝；
   - grouped Radon 与 grouped linear_resample 配置除 `mode` 外匹配。

## 6. GPU 前交给原 owner 的精确内容

本轮不新建远端调度器、不重复派发现有任务。实现已冻结在 `a01bba6f0579fb10df6f570f91bcd7d41f68fb4f`。G=1 经配置级测试证明与既有 `svd_radon` / `svd_resample` 完全同构，因此只允许在完整身份一致时复用；真正新增执行是 G=2/4/8/16 × {Radon, linear_resample} 共8臂。

原 production `project_case/project_units` 尚未提供 grouped supplement 的正式 case/receipt adapter；这是独立终审核实的部署 gate，不是 operator 缺陷。给原 owner 的精确接续要求已写入 [grouped_linear_gpu_handoff_20260918.md](grouped_linear_gpu_handoff_20260918.md)：先接入独立 supplemental receipt contract，再以 G=2 Radon+linear_resample 作为第一完整匹配组做原队列 GPU profile/恢复/正式 dev 验收，之后按预登记 G=4/8/16 接续。只有完整 matched receipt 通过后才能更新 `grouped_linear` 科研接受状态。

## 7. 当前阶段状态

- 计划：已确认，未扩大批准范围。
- 实现：本地完成并冻结为 `a01bba6f0579fb10df6f570f91bcd7d41f68fb4f`。
- 本地 CPU 验收：已通过。精确 Python3.11/Torch2.8.0/MHD V4 环境下 grouped 新测试 6 passed；相关 methods/automatic-matrix/grouped 回归 12 passed。`scripts/manage.py check`、publication coverage、terminology、`git diff --check` 均通过。
- 独立审查：只读 `codex-sol-standard` 终审确认 general/unequal-M 排列、组内跨来源/跨组隔离、G=1 compatibility、参数/连接 metadata、matched control 均通过；无 blocking local code defect。另补了 unequal source-rank 专门测试。
- 远端部署/运行：WebCodex 未启动；正式 grouped case/receipt adapter 与 GPU 运行继续由原 Codex owner 接入，精确请求见 `grouped_linear_gpu_handoff_20260918.md`。
- 科研接受：仍未接受；当前只完成本地实现/CPU acceptance，不能冒充 G>1 GPU 或真实 dev 结果。
- `uncompressed_grouped_budget_matched`：仍待可行匹配定义，未擅自改变 r/M/S 或填充通道。
