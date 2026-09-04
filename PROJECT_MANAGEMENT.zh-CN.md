# 代码、实验与服务器管理

GitHub 是代码和研究记录的正式来源；ws 只保留 `/home/mengh/RadonBridge`
这一份当前工作区。Git 历史对象无需删除。数据、缓存和结果在
`/data/mengh/RadonBridge`，与代码分开，不随分支切换删除。

- `main`：始终保存最新代码、实验协议和已完成的结果记录。每次有意义的代码更新前，
  先把旧 main 固定为 `YYYY_MM_DD_HH_MM_SS` 归档分支，再快进更新 main。
  时间戳表示归档时刻，使用 Asia/Riyadh 时区；源代码提交时间另由 Git 保留。
  不覆盖、不强推历史归档，失败与负结果也保留。
- `exp/NNN-假设`：额外保留便于检索的实验分支，记录修复、协议、代码、
  验证和结论；它不替代始终最新的 main 与时间戳历史分支。
  常规小修复逐次提交，按一次可执行更新归档，不按每一行修改建分支。
- `pilot/NNN-source` 或 `run/NNN-source`：固定实际执行的代码提交；结果记录
  完整 commit、MHD commit、协议、随机种子、权重和数据摘要。报告提交可以
  晚于执行提交，不能把报告提交误写为训练代码版本。
- `experiments/NNN-*/`：执行前协议、执行后汇总和决策依据。
  原始图像、参与者清单、个体预测、模型权重、日志不进入 Git。

已有实验：`exp/001-initial-handoff` 与 `exp/002-frozen-small-handoff` 固定
历史训练代码，`PILOT_REPORT.zh-CN.md` 保存其结果。003 起使用独立实验目录。

部署步骤：归档旧 main → 提交最新改动并快进 main、推送 GitHub → 确认 ws 工作区干净 → 获取运行锁
`/data/mengh/RadonBridge/.active.lock` → fetch main（可用 Git bundle，
无需在 ws 配置 GitHub 凭据）→ 单工作区切换到 main 并快进 → 校验 commit
与 MHD submodule → 放开部署锁 → 在同一个锁下执行实验。锁被占用则不切换。
本轮运行结束后，ws 从实验分支切回最新 main。所有后续部署和批次运行必须遵守该锁；它不是对绕过流程的系统级隔离。

`scripts/run_experiment.py` 顺序执行协议，持锁直到结束；拒绝覆盖非空结果
目录，拒绝脏代码训练，限定总时长。失败时保留现场，修复后用新提交和新
结果目录重跑，不混写旧日志。当前未实现自动断点续训。

批次命令（从 ws 代码根目录，已有环境变量 PYTHONPATH/TORCH_HOME）：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_experiment.py \
  --protocol experiments/003-cfp-resolution/protocol.json \
  --data /data/mengh/RadonBridge/cache/pilot256_128 \
  --output /data/mengh/RadonBridge/runs/exp003 \
  --lock /data/mengh/RadonBridge/.active.lock
```

每组实验只在改变下一步决策时扩展；先确认基线和可重复性，再扩大样本与
任务。增加训练次数本身不能弥补标签、验证选择偏倚或机制对照的缺陷。
