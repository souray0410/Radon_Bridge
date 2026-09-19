# WS02 CMX-FRM-init → same-A 三臂

- run：`cmx_frm_augmentation_20260919_v1`
- source commit：`28ebefcb9136e5fe1e56fc1d102130c288203670`
- author component：CMX FeatureRectifyModule，作者 repo commit `e251d860…`，MIT。
- 状态：3/3 complete，0 failed，manager 已退出。
- independent audit：`38a2f5b723c3e04530d12f787b1ac14f9802e257577aa2928a8519d8a835647c`。

incoming A 必须保留边界：前一 CMX-FRM host 包选择 best0，native parent state exact，预测与 no-communication reference 完全一致；所以这不是“非退化 CMX A 已成功”的证据，也不是作者完整 CMX 的复现。

二阶段从同一冻结 checkpoint 开始：continue 66.013%，Radon 72.630%，普通通信 67.391%（两分支平均 Macro-F1）。Radon−continue +6.617pp、Radon−ordinary +5.239pp，三项同时95%分别 [+1.725,+11.509]pp、[+0.564,+9.913]pp；ordinary−continue +1.378pp，同时95%跨0。

所有初始预测、accepted资产SHA、profiles/resume、fresh basis、296人指标和10k bootstrap已独立重算一致；test未使用。后续只消费冻结产物做汇总，不重启三臂。
