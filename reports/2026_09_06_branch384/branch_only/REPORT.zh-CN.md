# R&B（Radon Bridge）：机制基准结果

共378个结果位置，已验收378项；其余技术不可行单列。
仅分支协议；用户在部分融合结果已揭示后撤销融合协议，其结果保留于历史记录。28项直接与4项增强比较沿用原分支对比定义，比较族缩小属于事后范围修订，不是原56/8项校正或确认性证据。图中误差条是种子和学习率共同波动，不能当作泛化误差。
全部结果来自296人开发集，参与过历史任务筛选。尚未读取独立test，不能据此宣称临床有效或普遍优于其他方法。

## 主张与反例

|预定比较|差值 pp|同时95%区间|判定|
|---|---:|---|---|
|branch_equal_parameters_h512_k1|1.0133414932415992|[-0.427383171945948, 2.4540661584291463]|unresolved|
|branch_equal_parameters_h512_k3|1.9708640210427808|[0.08132876804119205, 3.8603992740443696]|unresolved|
|branch_equal_parameters_h512_k5|2.7035074874084444|[-0.4645530416104302, 5.871568016427319]|unresolved|
|branch_equal_parameters_h1024_k1|2.037938188038792|[-0.4957186693739404, 4.571595045451525]|unresolved|
|branch_equal_parameters_h1024_k3|2.9963610597667483|[-0.007142136395792864, 5.999864255929289]|unresolved|
|branch_equal_parameters_h1024_k5|3.140411510763485|[-0.18067348464472976, 6.4614965061717005]|unresolved|
|branch_equal_compute_budget512|1.8188818503911328|[-0.8411810945581997, 4.478944795340466]|unresolved|
|branch_equal_compute_budget512_k3_minus_k1|1.0294789457802231|[-0.6381758259184793, 2.6971337174789256]|unresolved|
|branch_equal_compute_budget1024|3.0723361144829595|[0.14708882366057674, 5.997583405305342]|unresolved|
|branch_equal_compute_budget1024_k3_minus_k1|-0.17095993227769846|[-2.174543518855874, 1.8326236543004772]|unresolved|
|branch_equal_compute_budget1024_k5_minus_k3|0.9789969155927203|[-1.242282359257437, 3.2002761904428776]|unresolved|
|branch_reference_h512_minus_mmtm_hidden128|2.6459505472037357|[0.13262173384609977, 5.159279360561372]|unresolved|
|branch_reference_h512_minus_mmtm_hidden256|2.645950547203734|[0.132621733846098, 5.15927936056137]|unresolved|
|branch_reference_h512_minus_mmtm_hidden512|2.6459505472037357|[0.13262173384609977, 5.159279360561372]|unresolved|
|branch_reference_h512_minus_mmtm_hidden1024|2.645950547203734|[0.132621733846098, 5.15927936056137]|unresolved|
|branch_reference_h512_minus_attention_d128|2.2551520945523778|[-0.14199483984653627, 4.652299028951292]|unresolved|
|branch_reference_h512_minus_attention_d256|2.198003193062064|[0.15373904316703468, 4.242267342957094]|unresolved|
|branch_reference_h512_minus_attention_d512|1.507147297755733|[-0.5025206557123589, 3.516815251223825]|unresolved|
|branch_reference_h512_minus_attention_d1024|1.4830628375283634|[-0.6111600599128755, 3.5772857349696023]|unresolved|
|branch_reference_h1024_minus_mmtm_hidden128|3.220622709196066|[-0.4805795767827883, 6.92182499517492]|unresolved|
|branch_reference_h1024_minus_mmtm_hidden256|3.2206227091960677|[-0.4805795767827865, 6.921824995174922]|unresolved|
|branch_reference_h1024_minus_mmtm_hidden512|3.220622709196066|[-0.4805795767827883, 6.92182499517492]|unresolved|
|branch_reference_h1024_minus_mmtm_hidden1024|3.2206227091960677|[-0.4805795767827865, 6.921824995174922]|unresolved|
|branch_reference_h1024_minus_attention_d128|2.8298242565447094|[-0.8030169292115592, 6.4626654423009775]|unresolved|
|branch_reference_h1024_minus_attention_d256|2.7726753550543943|[-0.5834144822140575, 6.128765192322846]|unresolved|
|branch_reference_h1024_minus_attention_d512|2.0818194597480613|[-0.9222555855241716, 5.085894505020294]|unresolved|
|branch_reference_h1024_minus_attention_d1024|2.0577349995206955|[-0.8207174125936656, 4.936187411635057]|unresolved|
|branch_equal_compute_budget512_k5_minus_k3|—|—|not_estimable|
|branch_mmtm_hidden256_augmentation_radon_minus_continue|2.1176981289826853|[-0.1809772584865179, 4.4163735164518885]|unresolved|
|branch_mmtm_hidden256_augmentation_radon_minus_linear_resample|2.08257452724432|[-0.12405367732670314, 4.289202731815343]|unresolved|
|branch_attention_d256_augmentation_radon_minus_continue|1.6440014502937643|[-0.2984137810932199, 3.5864166816807486]|unresolved|
|branch_attention_d256_augmentation_radon_minus_linear_resample|0.8844994293663853|[-0.7817594859825949, 2.5507583447153657]|unresolved|

完整逐种子、学习率、两分支及晚期融合见 all_results.csv；模型资源见 model_costs.json 和结构预检。
未来独立验证需扩大队列、审计标签与抽样、跨任务和模型；本轮不确定的结果如实保留，不根据排名继续搜索。
临床表型不等于新增影像专家金标准；病例对照平衡队列的预测值与校准不得直接推广到真实患病率。
