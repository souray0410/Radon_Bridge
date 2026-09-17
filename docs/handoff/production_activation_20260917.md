# 2026-09-17：管理修复正式启用与合盖独立运行

## 已部署

Ibex 管理快照 `7d1a723` 已接替旧CPU dispatcher；共享账户锁、领取表、run/spec、科学源码和现有GPU owners均保留。部署接替前后 `51909172.0/.1/.17`、`51919716.0/.3`、`51907003.0/.1` 完全一致。新dispatcher PID604476状态 `workflow_allocations_active`、error为空。

九个尚未获批且未启动的R&B申请，已通过版本化恢复清单绑定新管理快照。未来新申请也由新dispatcher生成。获批后自动先做适用的完整或严格复用资源预检，再训练，无需Mac保持SSH在线。其余历史待批申请使用同一恢复锁与原日志；本次不追加申请、不取消现有申请。

## 验收边界

- 54项针对CPU测试通过；实际清单87项、ready15项，全部源绑定已核验。
- 新worker主机内存申请104GiB；allocation128GiB、owner2GiB。原100.022GiB峰值不再被误认为可放进100GiB step。总申请106GiB仍保留超过15%的主机内存。
- 复用必须核验配置、科学源码、完整既有收据及硬件身份；硬件变化重新完整测量，损坏收据拒绝。即使复用，也需当前断点、最大眼数、25次更新及保存/恢复验收。
- **尚未在新获批GPU上完成该预检**，已自动接入获批入口。管理上线不等于父模型接受或六臂研究完成。3D父模型仍按原验收标准接受。
- 不改变batch、BN、精度、训练/停止规则，不访问test，不热改运行中的科学源。

## 远端证据

操作根 `OPS=/ibex/project/c2377/souray/home/mengh/operations/2026_09_10_11_11_31`。

- `radon_live_20260917_v1/{preflight,activation}.json`
- `radon_live_20260917_v1/{dispatcher.json,source,gate}`
- `lease_recovery_after_reboot_20260917/plan_radon_7d1a723.json`
- `radon_bridge_active_workflow.json`

远端tmux/独立管理进程与Slurm任务不依赖Mac连接。此结论不包含登录节点重启后的系统级自启动保证。旧owner、健康训练和已接受科学身份继续保持。控制器首次启动状态发布时间超过短轮询窗口；已复核其真实成功状态后完成切换，没有重复启动。
