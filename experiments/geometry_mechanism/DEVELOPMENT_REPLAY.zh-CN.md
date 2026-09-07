# 统一开发集重放执行（2026-09-07）

此前三小时自动跟进已启用，但未接上远端统一模型重放；不能把监控提示词等同已经运行的流水线。本次补齐独立、可恢复的重放队列。

输入为已完成文件核验的777个候选模型视图（含3个独立父模型对），固定候选清单SHA、当前代码提交和原development缓存清单SHA。完整MHD严格加载、batch16、eval/no_grad推理；逐参与者顺序/标签一致，概率沿用既有恢复验收rtol=1e-5、atol=1e-6，argmax决策与分支macro-F1必须完全一致。模型参数、持久buffer/BN、梯度及随机状态不能改变。多宽度联合训练使用已保存的三个导出模型视图；父模型分别严格加载两路最佳检查点并核对预训练保存预测。

仅从PairedDataset的validation分割读取296人，入口没有test选项，也不写test解封标记。此重放完成仍不替代当前模型来源、主要比较清单及统计规则锁定；原test评价门槛继续保留。

控制器使用原项目.active.lock及本次queue.lock。读取现有gpu_allocation.json，空集合/无效配置暂停派发、减卡排空、加卡派发；每物理卡最多一个本队列worker，遵守本项目总显存10GiB及实际剩余空间，不计LOOK显存为本项目用量。所有已验收结果有内容SHA，恢复时验证后跳过。失败停止新派发，保留独立attempt，只有明确--retry-failed才重试失败项；不降低精度容差或改变科学配置。磁盘保持100GiB安全余量。

负向测试覆盖参与者顺序、标签、概率偏差，以及容差内argmax翻转不得通过。真实开发重放输出本身构成模型兼容验收；确定性错误修复必须新版本新目录保留旧attempt。

入口：scripts/run_development_replay.py。必要参数：--inventory 候选目录 --output 唯一运行目录 --data 原缓存目录。输出status.json、registration.json、ledger.json、逐模型acceptance.json和accepted_models.json。只有development_replay_complete_test_still_sealed表示全部重放通过。
