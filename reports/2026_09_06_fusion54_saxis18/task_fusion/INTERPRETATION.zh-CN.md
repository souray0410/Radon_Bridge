# R&B：任务融合基准结果

54项均按融合预测macro-F1选优并达到平台。全部结果来自296人开发验证集，未读test。不能与原双头损失/选优研究混为同一排名。

| 主干LR | 方法 | 融合F1，均值 ± 种子SD (%) |
|---|---|---:|
| 3e-05 | Concat + MLP | 68.19 ± 2.59 |
| 3e-05 | Gated attention MIL | 67.73 ± 0.60 |
| 3e-05 | R&B SVD | rho=1/16, r16, h512 | 70.31 ± 0.55 |
| 3e-05 | R&B SVD | rho=1/8, r32, h1024 | 69.97 ± 0.66 |
| 3e-05 | R&B SVD | rho=1/4, r64, h2048 | 68.61 ± 1.74 |
| 3e-05 | MMTM | reduction4 | 67.75 ± 0.85 |
| 3e-05 | MMTM | reduction8 | 70.91 ± 1.15 |
| 3e-05 | Cross-attention | d128 | 69.79 ± 1.38 |
| 3e-05 | Cross-attention | d256 | 69.78 ± 1.90 |
| 6e-05 | Concat + MLP | 69.52 ± 1.00 |
| 6e-05 | Gated attention MIL | 68.46 ± 2.19 |
| 6e-05 | R&B SVD | rho=1/16, r16, h512 | 70.25 ± 1.69 |
| 6e-05 | R&B SVD | rho=1/8, r32, h1024 | 69.06 ± 3.16 |
| 6e-05 | R&B SVD | rho=1/4, r64, h2048 | 69.24 ± 1.46 |
| 6e-05 | MMTM | reduction4 | 69.07 ± 2.72 |
| 6e-05 | MMTM | reduction8 | 68.70 ± 0.19 |
| 6e-05 | Cross-attention | d128 | 69.81 ± 0.85 |
| 6e-05 | Cross-attention | d256 | 68.45 ± 1.27 |

## 36项预定比较
普通与36比较max-|t|同时区间见statistics.json和paired_comparisons.csv。±1pp仅为方法研究参考。

- bb3e-05_svd_rho1_16_minus_concat_mlp: +2.12 pp；同时区间 [-2.2940162025749657, 6.536654045859407]；unresolved。
- bb3e-05_svd_rho1_16_minus_gated_mil: +2.58 pp；同时区间 [-1.9724663634744672, 7.13271195421288]；unresolved。
- bb3e-05_svd_rho1_16_minus_mmtm_r4: +2.57 pp；同时区间 [-1.7268009368291324, 6.858421149548743]；unresolved。
- bb3e-05_svd_rho1_16_minus_mmtm_r8: -0.60 pp；同时区间 [-4.6171659045411, 3.422053009592424]；unresolved。
- bb3e-05_svd_rho1_16_minus_attention_d128: +0.52 pp；同时区间 [-3.2673162533418494, 4.3094748467460455]；unresolved。
- bb3e-05_svd_rho1_16_minus_attention_d256: +0.53 pp；同时区间 [-3.0138418289069397, 4.072695446386496]；unresolved。
- bb3e-05_svd_rho1_8_minus_concat_mlp: +1.78 pp；同时区间 [-3.2148860253189335, 6.774347228082221]；unresolved。
- bb3e-05_svd_rho1_8_minus_gated_mil: +2.24 pp；同时区间 [-2.8216465203287804, 7.29871547054606]；unresolved。
- bb3e-05_svd_rho1_8_minus_mmtm_r4: +2.22 pp；同时区间 [-3.0170086778203675, 7.465452250018824]；unresolved。
- bb3e-05_svd_rho1_8_minus_mmtm_r8: -0.94 pp；同时区间 [-5.8792232737913945, 4.00093373832159]；unresolved。
- bb3e-05_svd_rho1_8_minus_attention_d128: +0.18 pp；同时区间 [-4.397168324199253, 4.756150277082316]；unresolved。
- bb3e-05_svd_rho1_8_minus_attention_d256: +0.19 pp；同时区间 [-4.514087963323062, 4.889764940281465]；unresolved。
- bb3e-05_svd_rho1_4_minus_concat_mlp: +0.42 pp；同时区间 [-4.898629058741883, 5.7456305443962]；unresolved。
- bb3e-05_svd_rho1_4_minus_gated_mil: +0.88 pp；同时区间 [-4.811507381188332, 6.576116614296635]；unresolved。
- bb3e-05_svd_rho1_4_minus_mmtm_r4: +0.87 pp；同时区间 [-4.6719194835516245, 6.407903338641125]；unresolved。
- bb3e-05_svd_rho1_4_minus_mmtm_r8: -2.30 pp；同时区间 [-7.55418378203387, 2.963434529455081]；unresolved。
- bb3e-05_svd_rho1_4_minus_attention_d128: -1.18 pp；同时区间 [-6.412542977808629, 4.0590652135827145]；unresolved。
- bb3e-05_svd_rho1_4_minus_attention_d256: -1.17 pp；同时区间 [-6.408334166180607, 4.07155142603005]；unresolved。
- bb6e-05_svd_rho1_16_minus_concat_mlp: +0.73 pp；同时区间 [-4.348138682186637, 5.8073573352872785]；unresolved。
- bb6e-05_svd_rho1_16_minus_gated_mil: +1.79 pp；同时区间 [-3.0148621494984376, 6.588418584030304]；unresolved。
- bb6e-05_svd_rho1_16_minus_mmtm_r4: +1.18 pp；同时区间 [-4.012987107915375, 6.365859662625649]；unresolved。
- bb6e-05_svd_rho1_16_minus_mmtm_r8: +1.55 pp；同时区间 [-3.4654419457365893, 6.569277065482288]；unresolved。
- bb6e-05_svd_rho1_16_minus_attention_d128: +0.44 pp；同时区间 [-4.049849131470882, 4.92873857206914]；unresolved。
- bb6e-05_svd_rho1_16_minus_attention_d256: +1.80 pp；同时区间 [-2.5432979525294153, 6.147766318229977]；unresolved。
- bb6e-05_svd_rho1_8_minus_concat_mlp: -0.46 pp；同时区间 [-5.677589482566049, 4.748511897529312]；unresolved。
- bb6e-05_svd_rho1_8_minus_gated_mil: +0.59 pp；同时区间 [-4.259412808408653, 5.444673004803141]；unresolved。
- bb6e-05_svd_rho1_8_minus_mmtm_r4: -0.02 pp；同时区间 [-5.58433981196066, 5.548916128533548]；unresolved。
- bb6e-05_svd_rho1_8_minus_mmtm_r8: +0.36 pp；同时区间 [-4.761201974527503, 5.4767408561358275]；unresolved。
- bb6e-05_svd_rho1_8_minus_attention_d128: -0.75 pp；同时区间 [-5.398699960206143, 3.8892931626670144]；unresolved。
- bb6e-05_svd_rho1_8_minus_attention_d256: +0.61 pp；同时区间 [-4.2753248951844665, 5.49149702274765]；unresolved。
- bb6e-05_svd_rho1_4_minus_concat_mlp: -0.28 pp；同时区间 [-6.0930305668265685, 5.523120560872874]；unresolved。
- bb6e-05_svd_rho1_4_minus_gated_mil: +0.77 pp；同时区间 [-4.67497511111312, 6.2194028865906645]；unresolved。
- bb6e-05_svd_rho1_4_minus_mmtm_r4: +0.16 pp；同时区间 [-5.545590985786535, 5.869334881442486]；unresolved。
- bb6e-05_svd_rho1_4_minus_mmtm_r8: +0.54 pp；同时区间 [-5.434654857284504, 6.509361317975871]；unresolved。
- bb6e-05_svd_rho1_4_minus_attention_d128: -0.58 pp；同时区间 [-5.33453122870024, 4.184292010244175]；unresolved。
- bb6e-05_svd_rho1_4_minus_attention_d256: +0.79 pp；同时区间 [-4.2329489911633145, 5.80828869780954]；unresolved。

## 解释边界
同一原生网络、输入、三项CE、两档学习率、三种子和选优标准；参数量、家族配置数、历史筛选预算并不相等。MIL为深层空间token的门控汇聚，不是原论文2D B-scan系统完整复现。R&B在此配通用融合头；原独立预测器机制证据另报告。
本实验不能保证方法优势，也不能代替独立测试和外部验证。包含所有负结果、3418单列和逐种子波动；不按新结果筛选展示配置。
