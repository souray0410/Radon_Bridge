# 修复后组合筛查：38次有界短跑

目的：探索可用的训练配置与R&B通信设置；不是确认性效果评估。
协议在启动前固定于protocol.json。使用修复后的任务CE之和与逐任务梯度裁剪。

1. 8组无桥接：head-only / stage4 / stage3+4，主干学习率1e-6至1e-5，
   head学习率1e-4或3e-4，weight decay 0.01或0.1。
2. 20组桥接：stage2、stage3、stage2+3；H比例1/32、1/16、1/8；
   M半密度或双参考密度；S半分辨率但完整支持；线性Conv核1、3、5；
   self-only与打乱几何控制；另有M/H组合。具体仅执行协议列出的20组，不作全笛卡尔积。
3. 两个新种子3408/3409：各自重跑选出的无桥接配置、两个R&B候选，
   以及最优候选对应的同尺寸self/scrambled控制，共10次。

每次8 epoch，从ImageNet初始化及种子匹配的独立头开始；没有暖启动选优、
没有中途切换解冻范围、没有中途重置优化器。BN统计固定。
所有候选以相同数据顺序和训练步数比较；固定最后epoch为主要结果。

基线按两任务macro-F1算术均值排序，同分选参数更少的；R&B按两分支相对匹配
基线的较小F1增量优先、再按均值排序，取前2名。即使没有双分支共同改善，
仍复核前2名并标明负向或不稳定结果。排序分数不是另一个分类macro-F1。

每个epoch记录各分支验证macro-F1/precision/recall、训练CE、裁剪前梯度范数与
裁剪比例；最终记录训练/验证差距、通信增量/原生特征范数、显存和时间。
保存每次last权重与验证预测在ws；selected epoch仅为补充指标，没有保存selected权重。
不支持中断中途续训；已有status.json的目录不得重启覆盖，失败须保留记录后处理。

数据仍为256训练/128验证参与者，均与既有分割一致；测试集不使用。
这128人已经被反复查看；新种子复核不能替代独立验证数据。筛出的最高分有
选择偏差，所有bootstrap仅为探索性、未做多重比较修正。全量数据与任务选择
是否足够支持医学结论仍未验证；本轮不据最高分作临床或论文层级结论。

单GPU顺序执行；8GiB分配上限、自己进程9.5GiB停止阈值、整批最多45分钟。
运行锁防止部署/重复训练冲突；五分钟heartbeat检查进程与status。
验收：check_optimization.py及check_sweep.py，后者检查实际最大参数/几何配置的batch4前后向及一步更新。

启动：scripts/run_combination_sweep.sh
运行目录：/data/mengh/RadonBridge/runs/exp007_sweep
阶段产物：status.json、history.jsonl、partial_summary.json；完整结束才生成summary.json。

资源与优化验收已通过（源码9e3a2c1）：真实batch4较大配置峰值allocated约1559/1415MiB；两步独立CFP参数更新误差均为0。验收JSON随本协议保存。

已正常完成全部38次：详见REPORT.zh-CN.md和summary.json；未见稳定桥接收益。
