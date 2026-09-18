# 研究覆盖审查：哪些完成了，哪些还没有

2026-09-18核对批准协议、当前研究矩阵、ws02两个已完成包及本仓库累计页面。不是重新启动训练，也不是全部历史科学资产已迁移。机器清单：[coverage.json](coverage.json)。

|研究问题/范围|目前证据|尚缺什么/下一动作|
|---|---|---|
|小队列六臂：桥是否有附加收益|[6/6，单3416](small_cohort/README.md)|正式多种子与大队列不能由此替代|
|SVD与QR/可学习通道是否不同|[有限六项匹配包](channel_compression/README.md)|6/6已独立验收：原SVD两项复用、四项新增；中心化方向另待验收|
|全局线性分解是否保留几何收益|[24/24，三种子](factorized/README.md)，本次重新核验全部原产物和F1|补入累计页，精确复用，不重训；容量与几何因素分开解释|
|固定SVD+分组卷积|[右侧完整执行包](grouped_linear/README.md)：G=1严格复用，G=2/4/8/16×Radon/普通通信8/8结束，右侧SHA/296人/F1/profile/audit通过|等待左侧按原产物做最终独立科学验收；当前不把右侧自检直接升级为最终accepted|
|不压缩+分组与SVD+稠密比较|原批准补充保留|先核验等参数/近似等计算的可行匹配；不通过填通道或暗改r/M/S制造匹配|
|大队列六臂/三疾病/三架构|有限162位置；源状态保留最近运行核验时间|父模型、实际pair/runtime/恢复/逐臂验收尚未齐；13:40UTC复核Ibex仍待资源：Dense3D暂停、新32无正式更新|
|参考组49机制×三种子|147位置，下面逐臂列出|实现/CPU测试不是科研接受；不能把ws02的SVD/普通通信直接计为这里的ResNet50大队列结果|
|大队列分解补充|另计于309之外|正式case/feed/GPU/报告接入仍缺，不能用小队列24项填充|
|老师：已有方法A与A+桥|六臂中的MMTM/attention是适配基线|尚未完成匹配A/A+桥研究；先核验宿主、初始化及训练身份，不能把换骨干或适配基线冒充附加性比较|
|跨器官/三层迁移/729六网络|后续阶段保留|按眼科→心脏→眼心及明确临床/数据门槛推进，不是本周全部展开|
|更早的广泛历史研究与旧dev/test报告|保留历史源码和授权档案|本轮未全量逐项重验，不读取封存test，不宣称全历史已覆盖|

## 线性结构不要混为一谈

- 固定SVD降维后做稠密卷积：当前核心的主桥。
- 全局A/K/B可学习分解：本页24项，R是全局瓶颈容量。
- 分组卷积：限制直接连接范围；各组含CFP/OCT来源。WS02有限包已由右侧8/8执行并完成技术audit，左侧最终独立验收尚待完成。
- 旧learned_projected是按来源分别学习两端映射，与全局分解不同；learned_channel是另一个已登记机制因素。代码支持不代表已经训练。

原批准依据：[分组补充历史协议](https://github.com/souray0410/Radon_Bridge/blob/91beacfdbc3173993dbe84e11a824633b98b28c4/docs/linear_grouping_supplement_20260914.zh-CN.md)、[分解协议](https://github.com/souray0410/Radon_Bridge/blob/91beacfdbc3173993dbe84e11a824633b98b28c4/docs/factorized_ws_20260914.zh-CN.md)、[年末有限范围](../../semester_delivery_2026.zh-CN.md)、[研究矩阵](../../../src/radon_bridge/studies/research_matrix.py)。

## 本次漏报的原因与修复边界

旧检查只检查各有限包是否完成，没有把不同包与总入口合并核对。导致24项分解证据保存在服务器却未呈现在当前页面；总入口和机器摘要仍有过时表述。结果没有丢失，也没有因此失效。

现在使用显式清单，发布前运行 `python -m radon_bridge.analysis.publication_coverage --root docs/reports/current`。检查已接受包是否缺行/重复、运行身份是否改变、复用模型与预测SHA是否一致、未完成项是否说明下一步、报告链接是否存在；结果缺失时拒绝验收本次发布并保留旧证据。CI也运行同一门槛。每个新增包先登记后派发，不能只改一个局部报告。

它不能自动发现所有未登记的口头想法，也不能判断所有科学设计是否合理。负责人仍须每次包完成和周报前审查原协议、老师问题、反例与替代解释；这项责任不能转给用户。

## 参考组49个补充臂（均未在本轮接受为大队列结果）

下列来自当前锁定研究矩阵；保留每个身份，不因未运行或负结果删减。

|实验臂|状态|
|---|---|
|`qr_radon`|待对应大队列完整验收|
|`qr_linear_resample`|待对应大队列完整验收|
|`qr_self`|待对应大队列完整验收|
|`centered_radon`|待对应大队列完整验收|
|`centered_linear_resample`|待对应大队列完整验收|
|`learned_channel_radon`|待对应大队列完整验收|
|`learned_channel_linear_resample`|待对应大队列完整验收|
|`svd_spatial_scramble`|待对应大队列完整验收|
|`svd_s_axis_scramble`|待对应大队列完整验收|
|`svd_oct_to_cfp`|待对应大队列完整验收|
|`svd_cfp_to_oct`|待对应大队列完整验收|
|`svd_frozen_radon`|待对应大队列完整验收|
|`svd_frozen_linear_resample`|待对应大队列完整验收|
|`svd_frozen_self`|待对应大队列完整验收|
|`qr_spatial_scramble`|待对应大队列完整验收|
|`qr_s_axis_scramble`|待对应大队列完整验收|
|`qr_oct_to_cfp`|待对应大队列完整验收|
|`qr_cfp_to_oct`|待对应大队列完整验收|
|`qr_frozen_radon`|待对应大队列完整验收|
|`qr_frozen_linear_resample`|待对应大队列完整验收|
|`qr_frozen_self`|待对应大队列完整验收|
|`r16_radon`|待对应大队列完整验收|
|`r16_linear_resample`|待对应大队列完整验收|
|`r64_radon`|待对应大队列完整验收|
|`r64_linear_resample`|待对应大队列完整验收|
|`M16_radon`|待对应大队列完整验收|
|`M16_linear_resample`|待对应大队列完整验收|
|`M64_radon`|待对应大队列完整验收|
|`M64_linear_resample`|待对应大队列完整验收|
|`S32_radon`|待对应大队列完整验收|
|`S32_linear_resample`|待对应大队列完整验收|
|`S128_radon`|待对应大队列完整验收|
|`S128_linear_resample`|待对应大队列完整验收|
|`k1_radon`|待对应大队列完整验收|
|`k1_linear_resample`|待对应大队列完整验收|
|`k5_radon`|待对应大队列完整验收|
|`k5_linear_resample`|待对应大队列完整验收|
|`depth23_radon`|待对应大队列完整验收|
|`depth23_linear_resample`|待对应大队列完整验收|
|`depth23_self`|待对应大队列完整验收|
|`depth234_radon`|待对应大队列完整验收|
|`depth234_linear_resample`|待对应大队列完整验收|
|`depth234_self`|待对应大队列完整验收|
|`mmtm256_plus_continue`|待对应大队列完整验收|
|`mmtm256_plus_radon`|待对应大队列完整验收|
|`mmtm256_plus_linear_resample`|待对应大队列完整验收|
|`attention256_plus_continue`|待对应大队列完整验收|
|`attention256_plus_radon`|待对应大队列完整验收|
|`attention256_plus_linear_resample`|待对应大队列完整验收|

## 2026-09-18通道压缩补充验收

[SVD／随机QR／可学习通道，各配Radon与普通通信](channel_compression/README.md)六项已齐全并独立重核。两项SVD结果引用旧核心包，新增四次训练。三个接受包去重32次执行；早先28是新增包之前的历史计数。分组、中心化、A/A+桥及大队列门槛仍开放，不称全项目完成。

## 2026-09-18 新增完整包

ws02 MMTM加桥三臂3/3已独立接受，当前四包去重35次执行；[详细比较](augmentation/README.md)。本适配不覆盖所有外部方法加桥，分组、中心化及大队列等原范围保留。\n\n## 2026-09-18 grouped 执行完成，待左侧最终独立验收\n\n[G=1/2/4/8/16完整执行报告](grouped_linear/README.md)已由右侧从WS02原产物复制并校验字节SHA。G=2/4/8/16共8个新臂均完成，manager无失败；最终技术audit核296人顺序、独立sklearn F1、profile/resume和全部资产SHA通过。此处仍保持coverage非accepted，直到左侧按原产物独立复核并显式接受。
