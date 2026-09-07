# R&B：功能依赖与冻结原网络补充

6项冻结训练、24个检查点×4种开关状态完成；原378项不改写，不把96个诊断视图当作96次训练。
冻结主干、原分类头及BN统计，仅中间卷积训练。所有原生参数/缓冲区逐轮SHA保持一致。
关闭诊断保留原生特征路径，两者关闭对应微调后的原生网络；原独立预训练预测另列。推理干预可能造成分布变化，不能当严格因果证明。
冻结实验2项比较与功能依赖8项比较分别校正；不以增加实验消除不利结果。所有统计为296人开发集探索性证据。

- frozen_radon_minus_linear_resample: +1.483 pp, simultaneous CI [-0.7206801718148301, 3.686139632577255]; unresolved
- frozen_radon_minus_parent: +1.483 pp, simultaneous CI [-0.7206801718148301, 3.686139632577255]; unresolved
- mmtm_hidden256_radon_both_on_minus_host_only: +19.462 pp, simultaneous CI [15.642695903885837, 23.281340843359686]; supports_substantive_improvement
- mmtm_hidden256_radon_both_on_minus_new_only: -0.102 pp, simultaneous CI [-0.9108890366210554, 0.7069405805836178]; supports_practical_similarity
- mmtm_hidden256_linear_resample_both_on_minus_host_only: +0.549 pp, simultaneous CI [0.0662870959472524, 1.0312575639493606]; unresolved
- mmtm_hidden256_linear_resample_both_on_minus_new_only: +0.000 pp, simultaneous CI None; undefined_zero_variance
- attention_d256_radon_both_on_minus_host_only: +20.467 pp, simultaneous CI [16.772233180391705, 24.16176041223402]; supports_substantive_improvement
- attention_d256_radon_both_on_minus_new_only: +5.699 pp, simultaneous CI [3.881403108728714, 7.515999624869812]; supports_substantive_improvement
- attention_d256_linear_resample_both_on_minus_host_only: +0.633 pp, simultaneous CI [-0.4288854987861168, 1.6942046489493248]; unresolved
- attention_d256_linear_resample_both_on_minus_new_only: +4.106 pp, simultaneous CI [2.2475827403555386, 5.96414704352752]; supports_substantive_improvement

all_results.csv和所有聚合图表可以同步；parent_seed*.npz及诊断参与者预测必须保留授权环境，不得上传GitHub。
