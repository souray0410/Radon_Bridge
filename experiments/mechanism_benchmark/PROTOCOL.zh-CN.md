# R&B：结构化跨来源通信机制基准（用户预先批准）

## 范围与完整性

保留已有129项，新增57项，最终186项不同第二阶段结果。仅3416/3417/3418；不重新预训练，不重新拟合基。研究主线为Radon通信，压缩是实现和稳健性因素。现有原23页报告与全部证据只引用，不覆盖。

本轮为受控机制基准，历史开发集仍属于探索；预定分析并不使它成为独立确认性验证。未来数据、模型、任务与外部验证另立协议。不以实验数量保证NBE录用，不要求性能单调，不只展示较弱基线。

## 新增矩阵

- 三种子×三档rho=1/16、1/8、1/4×QR线性重采样：9项。
- 三种子×三档rho×SVD/QR×OCT→CFP、CFP→OCT：36项。
- MMTM reduction4/8（hidden256/128）×三种子：6项。
- 双向注意力dimension128/256、4头×三种子：6项。

Radon类固定stage3/M32/S64、r16/32/64、h512/1024/2048；rho是维数比例，不是能量阈值。QR重采样复用旧QR和原线性重采样前向/返回行L2范数匹配；32为打包维度。单向保留两个自身块，只屏蔽一个跨来源块；cross_edges=[source,destination]，权重行=目的、列=发送，未提供保持旧双向。存储和有效参数分别报告。

MMTM空间GAP→concat→FC/ReLU→两个FC→2sigmoid通道调制，末层weight/bias零初始化。作者公开代码使用sigmoid；本适配遵循论文2sigmoid并增加保持初始预测的零初始化，只接stage3，不声称完整系统复现。

注意力使用完整196/288 tokens、每来源LayerNorm、两个方向独立带偏置Q/K/V和输出投影，head维度d/4，softmax(QK^T/sqrt(d/4))V；两方向从写回前同时计算，输出weight/bias零初始化。无FFN、dropout、显式位置编码。family默认radon，基线rho/M/S不适用，不称等宽。

两网络仍使用MHD V4前后向、原Node ID、各自头/CE/预测。stage3、backbone6e-5、head/bridge1e-4、全参数/BN更新、batch16、AdamW WD.01、每分支/桥裁剪5。至少8轮，连续6轮无>0.001改善判平台，3轮LR×0.3，60轮仅保护上限；按两分支平均F1选择共同检查点。无桥及每个方法均复用对应种子的相同父检查点与数据顺序。OOM/失败/未平台需处理，不改batch或标准。

## 只读分析

新增57项初始/选中模型：全部1264训练参与者能量与残差；前128稳定排序训练参与者、batch16、eval，按参与者加权累积两个CE梯度后计算stage3与桥子模块范数/余弦。无固定子空间的MMTM/注意力能量比例N/A；零梯度未定义，固定卷积行块正交核验不用于非线性基线。保持参数、BN、参数梯度及外部随机状态。

54个标准Radon选中模型=两基×三方向×三rho×三种子：正确配对，分别关掉两方向及全部跨来源项，分别打乱两方向及双方。20组全开发集296人无自配对置换，NumPy独立default_rng(202609051)拒绝采样，标签无关，跨模型共用，双眼整体移动；双方分别使用置换/逆置换。仅替换跨来源消息，自身/接收者/标签保持。缓存写回前特征并核对完整MHD输出。置换次数不是新增参与者，不作正式因果证明，不参与训练与选优。

全部186项计算0.5p_CFP+0.5p_OCT晚期融合，固定argmax；融合F1与分支F1平均严格分开。额外报告AUROC、宏precision/recall、混淆矩阵、NLL、二分类Brier=mean((p_case-y)^2)。

## 预定统计

7项机制比较对三种子三rho等权平均：SVD/QR各Radon减重采样（2），上述几何收益之差（1），两基下双向减删除OCT→CFP对CFP影响（2），双向减删除CFP→OCT对OCT影响（2）。

24项基线比较=两基×三rho×四基线配置，以两分支F1平均、再跨三种子平均。先每个模型计算F1，不先平均不同种子/rho概率。

10000次296位参与者配对bootstrap，default_rng(20260905)，所有模型共享索引。普通95%百分位区间与31项bootstrap max-|t|近似同时区间并列：中心化偏差(theta*−theta_hat)，各比较bootstrap样本SD标准化，对31项max绝对值取95%分位数，再theta_hat±q*SD。零方差记未定义，不除零。辅助逐种子/分支/rho完整展开但不混入31项校正家族。

±1pp研究参考范围：同时区间完全>+1支持实质提升，完全<−1支持下降，完全在[-1,1]支持实际接近，其余尚不能分辨。并非临床阈值；所有区间条件于固定模型和开发集选优。3418单列；不能把3×296或20×296当成独立参与者。

## 执行与交付

隔离检出、CPU旧路径和新增结构验收、GitHub main及时间戳分支；取得项目锁，9种最大配置batch16完整前后向/优化/诊断预检，再部署。每GPU项目≤10GiB，不影响LOOK，无GPU时长上限但持续记账。验收129项配置、平台、父/基/预测/选中检查点与摘要SHA，新唯一目录，不覆盖/重复运行。

33个代表选中模型=三种子×（无桥+两基三rho+四基线）。GPU1、固定前16训练样本、10预热/50同步计时，报告中位/IQR、设备、其他负载；有其他进程标受干扰，不宣称速度优势。训练/拟合/诊断/预检/计时分别记账。

生成完整CSV、所有预定配对表、逐种子/区间/方向/扰动/诊断/成本图（PDF/SVG/PNG）、英文16:9新整合PDF与中文解读、环境/协议/偏离记录、图源数据。最终PDF逐页渲染审核后交付；旧23页保留。自动收取，健康运行不通知，完成/失败/需处理通知，最终暂停收尾自动化。

数据：1264训练、296开发，病例/对照632/632和148/148，同一青光眼目标两来源预测。UKB记录导出临床表型，非新增专家影像金标准。数据卡报告能核实的抽样/缺失/年龄/性别/中心/标签来源与时间，伦理/许可未知列待补。平衡样本校准/预测值不推广真实患病率。GitHub仅聚合指标/代码/报告/图源数据；参与者级概率、特征、索引保留授权环境。

来源：
- https://openaccess.thecvf.com/content_CVPR_2020/html/Joze_MMTM_Multimodal_Transfer_Module_for_CNN_Fusion_CVPR_2020_paper.html
- https://github.com/haamoon/mmtm/blob/master/mmtm.py
- https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html
- https://www.nature.com/natbiomedeng/submission-guidelines/about/aims
- https://www.nature.com/natbiomedeng/editorial-policies/reporting-standards
- https://www.nature.com/natbiomedeng/editorial-policies/clinical-research
