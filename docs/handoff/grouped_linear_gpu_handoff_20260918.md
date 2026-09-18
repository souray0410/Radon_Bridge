# 分组线性通信：原 GPU owner 精确接续请求

日期：2026-09-18  
本地实现目录：`/Users/mengh/Documents/WebCodexProjects/Radon_Bridge`  
本地分支：`webcodex/grouped-linear-20260918`  
实现提交：`a01bba6f0579fb10df6f570f91bcd7d41f68fb4f`  
批准协议来源：`91beacfdbc3173993dbe84e11a824633b98b28c4:docs/linear_grouping_supplement_20260914.zh-CN.md`

## 责任边界

WebCodex 已完成本独立 Mac clone 的 grouped operator、匹配配置、CPU 结构验收和独立只读审查；没有新建远端调度器、没有申请/领取 GPU、没有修改现有远端 worker，也没有读取 test。

现有 Ibex/ws02 调度、claims、allocation owner、父模型与 UKB 仍由原 Codex owner 管理。下面是给同一 owner 的接续请求，不授权第二个 owner 或重复派发。

## 科学身份

固定条件保持批准协议原样：

- compression：`fixed_svd_channel`
- `r=32, M=32, S=64, k=3`
- bridge stage：3
- G：`1,2,4,8,16`
- 每个 G 均比较 `radon` 与 `linear_resample`
- 每组都同时含 CFP/OCT 来源；排列为 `(group, source, channel, direction)`，返回前逆排列
- 不添加非线性、门控、额外卷积或自动通道洗牌
- 标签、划分、父模型选择、训练停止、BN/精度、主要比较和 test 用途全部不变

G=1 不是新训练身份：当且仅当父模型、数据、节点、SVD 基、r/M/S/k、种子和训练协议完整身份一致时，复用既有 `svd_radon` / `svd_resample` 接受产物。否则不得声称复用。

新执行只包含 G=2/4/8/16 × {Radon, linear_resample} 共 8 个 arm；先用 G=2 完成正式技术/资源接通，再按同一预登记顺序 G=4/8/16。此顺序只控制技术接续，不按分数改变科学范围。

## 原 owner 接续前必须完成的部署门槛

当前 production `project_case.validate_spec` / `project_units` 只接受主研究矩阵，尚无 grouped supplement 的正式 case/receipt adapter。独立终审将其判定为部署/交接 gate，而不是本地 operator 缺陷。

原 owner 应在独立版本化源码中接入 supplemental case/receipt contract，并满足：

1. grouped supplement 与 49-arm/309 主矩阵分开登记，不修改主矩阵含义或计数。
2. 任务 identity 包含完整 case、arm、G、父模型 manifest SHA、SVD basis SHA、source pin 和种子；重复身份拒绝二次训练。
3. G=1 先走严格复用验证；只有身份完全一致才引用已有接受产物。
4. G>1 每个 arm 有独立 claim/run/recovery/accepted receipt，可使用现有唯一 dispatcher/owner；不得另建竞争调度器。
5. feed、case、receipt 都保持 `test_access=false`。
6. adapter CPU/真实输入预检通过后才允许 GPU admission。

## 每个代表 GPU arm 的验收

至少记录并核验：

- 严格加载父模型、固定 SVD basis 和原 Node ID
- 零初始化时预测与对应父/基线严格重放一致
- grouped weight 的真实 `Conv1d(groups=G)` 结构
- 组内跨来源直接梯度非零
- 跨组直接梯度为零；只表述直接连接限制，不外推全网络独立
- optimizer 参数覆盖完整，固定 basis 不更新
- 一次实际优化更新及下一更新 checkpoint 恢复一致
- full-development 评价仍使用原 dev 角色
- GPU/host 峰值、实际 grouped 参数数、dense-equivalent 参数数、有效连接比例、吞吐
- 正式训练仍按原 stopping/plateau 规则，负结果不阻止后续 G 或锁定重复
- accepted receipt 带文件 SHA；原始参与者预测留在授权存储

G=2 的 Radon 与 linear_resample 两臂形成第一完整匹配组。只有这两臂及上述诊断/报告均接受后，才能称“代表 grouped 匹配组已接受”；不能用单臂或 CPU 测试替代。

## 当前已通过的本地证据

精确 Python 3.11 / Torch 2.8.0 / MHD Framework V4 `3559caa8d596d4438533a69d39d8a2c32eb21e46`：

- grouped 新增测试：6 passed
- methods + automatic-matrix + grouped 相关回归：12 passed；现有 projection reference 测试产生 9 个既有 NumPy RuntimeWarning，但无失败
- `scripts/manage.py check`：structure/framework 通过
- publication coverage：4 accepted packages / 10 indexed packages / 35 unique accepted executions，`full_project_accepted=false`
- terminology：95 files checked，known aliases absent
- `git diff --check`：通过
- read-only `codex-sol-standard` 终审：排列、group connectivity、G=1 compatibility、metadata、matched control 均 Pass；无 blocking local code defect

## 仍未关闭

- 本文件本身不是 GPU 部署或 scientific acceptance receipt。
- grouped formal case/receipt adapter 尚未由原 owner 部署/验收。
- G>1 尚无正式 GPU 运行结果。
- `uncompressed_grouped_budget_matched` 的等参数/近似等计算可行匹配尚未定义并验收，不得自动改变 r/M/S 或填充通道。
- Ibex 3D 父模型与既有正式门槛沿原 owner 当前状态继续；本地实现不刷新其远端证据时间。
