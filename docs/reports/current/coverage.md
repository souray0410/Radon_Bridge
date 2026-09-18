# 研究覆盖审查：哪些完成了，哪些还没有

2026-09-18核对批准协议、当前研究矩阵、ws02已完成包及本仓库累计页面。不是重新启动训练，也不是全部历史科学资产已迁移。机器清单：[coverage.json](coverage.json)。

|研究问题/范围|目前证据|尚缺什么/下一动作|
|---|---|---|
|小队列六臂：桥是否有附加收益|[6/6，单3416](small_cohort/README.md)|正式多种子与大队列不能由此替代|
|SVD与QR/可学习通道是否不同|[有限六项匹配包](channel_compression/README.md)|6/6已独立验收：原SVD两项复用、四项新增；中心化方向另待验收|
|全局线性分解是否保留几何收益|[24/24，三种子](factorized/README.md)，本次重新核验全部原产物和F1|补入累计页，精确复用，不重训；容量与几何因素分开解释|
|固定SVD+分组卷积|[已接受WS02单种子包](grouped_linear/README.md)：G=1严格复用，G=2/4/8/16×Radon/普通通信8个新增执行，左右两次独立核验通过|五项同时区间全跨0；G同时改变连接拓扑和参数量，不能把跨G点估计趋势当单一机制结论|
|中心化SVD基方向|[右侧完整执行包](centered_basis/README.md)：旧未中心化Radon/普通严格复用，中心化Radon/普通两新臂完成，独立basis数值gate、profiles、296人指标与3项10k区间均通过|等待左侧从原产物最终独立验收；当前仍不计入accepted包/去重执行数|
|不压缩+分组与SVD+稠密比较|原批准补充保留|先核验等参数/近似等计算的可行匹配；不通过填通道或暗改r/M/S制造匹配|
|大队列六臂/三疾病/三架构|有限162位置；源状态保留最近运行核验时间|父模型、实际pair/runtime/恢复/逐臂验收尚未齐；13:40UTC复核Ibex仍待资源：Dense3D暂停、新32无正式更新|
|参考组49机制×三种子|147位置，下面逐臂列出|实现/CPU测试不是科研接受；不能把ws02的SVD/普通通信直接计为这里的ResNet50大队列结果|
|大队列分解补充|另计于309之外|正式case/feed/GPU/报告接入仍缺，不能用小队列24项填充|
|老师：已有方法A与A+桥|[MMTM同宿主三臂](augmentation/README.md)已完成单3416探索：继续/+Radon/+普通匹配|同时区间跨0且宿主是本项目适配，不是作者完整系统；是否扩大到另一宿主由左侧另裁决|
|跨器官/三层迁移/729六网络|后续阶段保留|按眼科→心脏→眼心及明确临床/数据门槛推进，不是本周全部展开|
|更早的广泛历史研究与旧dev/test报告|保留历史源码和授权档案|本轮未全量逐项重验，不读取封存test，不宣称全历史已覆盖|

## 线性结构不要混为一谈

- 固定SVD降维后做稠密卷积：当前核心的主桥。
- 全局A/K/B可学习分解：本页24项，R是全局瓶颈容量。
- 分组卷积：限制直接连接范围；各组含CFP/OCT来源。WS02固定SVD单种子包已独立接受，但G增加同时降低参数量，不能把跨G趋势单独解释为拓扑效应。
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

[SVD／随机QR／可学习通道，各配Radon与普通通信](channel_compression/README.md)六项已齐全并独立重核。两项SVD结果引用旧核心包，新增四次训练。该阶段三个接受包去重32次执行；此后MMTM宿主加桥与grouped有限包也已分别接受。中心化与大队列门槛仍开放，不称全项目完成。

## 2026-09-18 新增完整包

ws02 MMTM加桥三臂3/3已独立接受；[详细比较](augmentation/README.md)。本适配不覆盖所有外部方法加桥，中心化及大队列等原范围保留。

## 2026-09-18 grouped 单种子包独立接受

[G=1/2/4/8/16完整报告](grouped_linear/README.md)已由左侧从WS02原始资产独立复核后接受：G=1两项严格复用核心包，G=2/4/8/16共8个新执行，manager无失败；89个源码文件、54份科研资产、296人顺序/F1、8个profile恢复与10,000次五项区间均重核通过。加入两条G1精确复用后，当前五个已接受发布包合计去重为43次实际执行。无压缩grouped预算匹配、中心化SVD和Ibex范围仍开放，不称全项目完成。
