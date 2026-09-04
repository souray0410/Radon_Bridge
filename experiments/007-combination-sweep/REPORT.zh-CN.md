# 007组合筛查完整结果

38/38次正常完成，304epoch，耗时11.717分钟，进程采样峰值2578MiB。全部76组最终预测的macro-F1/precision/recall和混淆矩阵已在ws独立重算一致。

结论：未发现稳定的桥接收益。预定规则入选的两个候选是stage3全S与半S，它们是相对较好的诊断候选，并非有效性已成立。两个新种子下OCT都下降。增加H/M没有解决问题，一些大配置出现明显退化。

## 所有固定最后epoch结果

| 试验 | CFP macro-F1 | OCT macro-F1 |
|---|---:|---:|
| baseline_head_1e4 | 55.44% | 60.10% |
| baseline_head_3e4 | 53.56% | 57.44% |
| baseline_s4_1e6 | 51.27% | 56.58% |
| baseline_s4_3e6 | 56.43% | 56.01% |
| baseline_s4_1e5 | 58.14% | 54.98% |
| baseline_s34_1e6 | 53.56% | 56.53% |
| baseline_s34_3e6 | 58.56% | 56.94% |
| baseline_s4_wd01 | 56.43% | 56.01% |
| screen_rb_s2 | 50.27% | 57.21% |
| screen_rb_s3 | 54.24% | 59.01% |
| screen_rb_s23 | 50.18% | 56.76% |
| screen_rb_h16_s3 | 51.67% | 58.02% |
| screen_rb_h16_s23 | 47.41% | 55.16% |
| screen_rb_h8_s3 | 45.15% | 51.87% |
| screen_rb_h8_s23 | 51.57% | 56.67% |
| screen_rb_mhalf_s3 | 54.47% | 56.58% |
| screen_rb_mhalf_s23 | 56.26% | 56.58% |
| screen_rb_mdouble_s3 | 55.56% | 52.71% |
| screen_rb_mdouble_s23 | 40.12% | 49.19% |
| screen_rb_shalf_s3 | 53.98% | 58.16% |
| screen_rb_shalf_s23 | 49.57% | 56.53% |
| screen_rb_kernel1 | 56.67% | 56.28% |
| screen_rb_kernel5 | 56.00% | 54.24% |
| screen_self_s3 | 54.47% | 58.56% |
| screen_self_s23 | 54.37% | 56.28% |
| screen_scrambled_s3 | 47.89% | 55.69% |
| screen_scrambled_s23 | 52.63% | 55.33% |
| screen_rb_mdouble_h16_s3 | 39.92% | 32.98% |
| confirm_3408_independent | 50.98% | 59.13% |
| confirm_3408_rb_s3 | 50.59% | 54.51% |
| confirm_3408_rb_shalf_s3 | 52.56% | 53.43% |
| confirm_3408_matched_self | 52.63% | 55.44% |
| confirm_3408_matched_scrambled | 53.97% | 54.29% |
| confirm_3409_independent | 54.51% | 60.10% |
| confirm_3409_rb_s3 | 53.77% | 51.95% |
| confirm_3409_rb_shalf_s3 | 55.72% | 53.77% |
| confirm_3409_matched_self | 53.55% | 58.59% |
| confirm_3409_matched_scrambled | 53.26% | 55.56% |

## 入选与限制

无桥接入选head_1e4（主干冻结，head lr1e-4，wd0.01），CFP55.44%、OCT60.10%。这只是8epoch/256训练样本下的预定排名，不能称为已经建立强基线。

桥接组使用同一头训练配方，但会新增约142万可训练通信参数。虽然对照初始化和优化步骤一致，容量变化仍是解释结果的因素。self/scrambled对照与对应桥接匹配通信容量。

008学习率选择规则：为避免把head-only的backbone_lr=0转移为假解冻，选007中adapt_stages非空且固定最后epoch两分支F1均值最高的配方，转移非零主干lr、head lr和wd，再比较18/34层及解冻范围。这个规则在008结果出现前固定。

两个新种子只检验初始化/训练顺序敏感性，不能代替独立验证集。128人开发验证曾被反复检查，包含筛选偏差；所有bootstrap为探索性。完整macro precision/recall、混淆矩阵、裁剪比例与区间见summary.json及ws history.jsonl。

后续：扩大到1264训练、296开发验证；先检查ResNet18/34基线，之后按已固定候选进行5种子对照。保持测试集未评估。
