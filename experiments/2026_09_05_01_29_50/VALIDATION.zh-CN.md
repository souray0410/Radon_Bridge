# SVD代码验收记录

验收在ws02的隔离副本完成，未修改当时运行中的正式源码，GPU验收等待前32项完成。

- check_fixed_channel_basis.py：正交、与直接SVD能量排序/投影子空间一致、rho嵌套、RNG不变、不同输入基差异；2D/3D投影与Radon/普通反投影交换关系；只含中间卷积可学习；检查点拒绝不匹配基；零初始化预测完全一致、MHD原Node ID及前后向、原生参数更新、BN更新、跨任务梯度通过。
- check_integer_bridge.py：原2D/3D/4D数值与异构拓扑通过，MHD和PyTorch梯度误差0；几何误差约1e-16。
- check_two_stage.py：六实验臂原生检查点初始预测完全一致，优化器刷新，全参数解冻；机制控制初始化通过。
- check_scheduler.py、check_participant_sampling.py、check_learning_rates.py：无限时长/10GiB排程、标量与逐来源配置、分组学习率兼容通过。
- 与e6a4f4f原bridge.py做独立对比：radon/self/pooled/random/scrambled五模式的state_dict键和值初始化逐位一致；载入非零卷积后输出及输入梯度逐位一致。
- 核对真实12项可学习参考及4项无桥的配置、完整平台、父checkpoint SHA及native初始化一致。
- 报告临时QA夹具（不是研究结果）：0项、12项以及其中1项达到epoch_cap场景通过，24个10000次配对bootstrap比较与指标核验通过；渲染检查布局。QA图不作为实验交付。
- CPU静态编译及git diff空白检查通过。

正式GPU验收保存在运行目录gpu_acceptance.json，包含仅训练集拟合及rho1/4 batch16的3次完整前向、MHD反向、AdamW更新、原生模块更新、双向跨来源梯度和峰值显存。未生成该文件之前不能声称GPU验收通过。
