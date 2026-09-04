# R&B（Radon Bridge）：backbone学习率与ρ交叉比较

backbone比较原值3e-5和提高一倍的6e-5；head/bridge均1e-4。固定stage3、M=32、S=64。两来源统一ρ=1/16、1/8、1/4，对应每路h=512、1024、2048。

两个backbone学习率×两个种子3416/3417×（无桥+三档ρ），共16项新实验。各臂从该种子独立模态最佳检查点初始化，共同检查点按两分支平均macro-F1选取，报告各分支结果。低原生学习率3e-6保留为历史负结果，不加入本轮。

用户明确取消GPU时长预算，实验不再受历史240 GPU分钟上限约束，继续记录实际耗时。先以原batch16验证大宽度显存。保留每卡10 GiB项目显存约束；资源受限必须明确标注，不得报告为收敛或无收益。其余AdamW、梯度裁剪、数据、BN更新、平台期规则不变。60轮为保护上限，达到上限不代表收敛。

全部完成后自动生成统一ρ扫描图、两个backbone学习率的匹配对照、完整指标、配对bootstrap及原始来源。历史实验不覆盖。

ρ=1/2 was deferred by the user after exceeding the 10 GiB profile limit; it is excluded from this experiment matrix.
