# 32层独立阶段：获批接续已绑定，正式训练等待资源

核验截止：2026-09-17 17:53 UTC。既有眼科研究、128层父模型和LOOK健康任务继续；本页不是六臂科研结果。

## 配方与当前边界

用户批准独立的青光眼3416阶段：ResNet18-3D随机初始化，沿完整OCT体积均匀抽32层，224×224，每次16名参与者、双眼最多32个体积，FP32。大队列划分不变。原128层路线保留，两者不得混成同一实验身份。

5次预热、20次更新、16人dev推理与断点下一步一致的短GPU预检已通过：GPU峰值37.396GiB，step主机峰值6.016GiB，约5秒/update。短预检为单线程；正式executor为两线程，必须重新完成全train读取、完整dev、更新及恢复资源验收。不能将短预检计入正式训练或接受父模型。

Canonical run为2026_09_17_20_04_14_937028，spec SHA为c2b3637f8f0fa195b63c74d03f3bbca50947d8ebebfc979712c77175cce1b59c。
当前仍只有登记文件，formal_updates=0；完整资源预检和正式领取均等待合适allocation获批。

## 已完成的操作接续

- 9个旧待批R&B申请已在原统一lease watcher中绑定有限canonical队列，再接原R&B owner；其他11个LOOK/native条目逐对象不变。
- 未来申请也使用相同finite owner；复用原R&B CPU daemon、原申请参数、requests账本、dispatcher锁、统一Claims和额度算法。新增申请仍单卡48小时、16CPU、128GiB，不创建新调度队列。
- 旧CPU与旧watcher退出前，核实PID/start、子进程和同session全部成员；新CPU standby通过后才安全交接。没有向GPU进程或allocation发送停止信号。
- 交接后真实CPU周期为workflow_allocations_active/error=null；唯一watcher完成20条waiting_grant周期。请求账本SHA与交接前相同，无重复job ID；实际仍3运行、21待批，而非24张正在训练。
- 每个allocation独占锁、共享有限预检锁以及原Claims避免多个新卡重复预检/领取同一32层任务。实际EndTime保留900秒；恢复预检读取实际last.pt，前后及receipt SHA必须一致。未知存活、缺失断点或未验收完成均不视为可抢占。

MHD_Models独立管理快照090c906通过本地与Ibex52项针对测试；future绑定715866a通过两端43项测试，两个准确SHA的GitHub检查均成功。正在执行的科学源码未改。

证据入口：OPS/radon_fast32_20260917下pending_activation_v1.json、future_activation_v1.json、control_followthrough_v1.json、finite_owner_config_v1.json及future_binding_v1.json。OPS指operations/2026_09_10_11_11_31。实际当前入口仍从radon_bridge_active_workflow.json读取。

## 尚未通过的门槛与周日交付

实际获批后须看到完整profile、唯一claim、正式更新及断点增长，才称32层正式运行。父模型训练、预测重放和选型验收后，才能进入匹配六臂；项目完整pair、资源、恢复与报告绑定仍待验收。约5小时/轮的短probe外推意味着不能保证周日前完成50轮及六臂，不为赶进度改停止规则或宣布未收敛模型合格。

旧DenseNet121-3D AMD3416当前epoch3验证推进到10000/12505；这是进度，不是接受模型。周日报告保留已核验结果、当前阻塞与实际修复，并明确这条独立阶段能回答的范围。
