# R&B 项目介绍讲解稿

## 1. R&B: Radon Bridge

开场：我们研究如何把两个已经能够独立预测的影像网络连接起来，让它们交换有用信息。R&B指Radon Bridge，核心是固定Radon几何、中间可学习线性通信、普通反投影和残差写回。当前受控任务是UKB配对CFP与OCT的同一青光眼标签。该页是结构示意，不是真实影像或实验数据。来源：项目协议、radonbridge/bridge.py与model.py，训练源码f850929及补充实现6ada3a3。

## 2. Two independent prediction networks

网络：两条完整ResNet18分别处理CFP二维图像和OCT三维体积。每人双眼，网络内部将眼维展平，所以stage3张量第一维是2B。CFP输入[B,2,3,224,224]，stage3为[2B,256,14,14]；OCT输入[B,2,1,32,96,96]，stage3为[2B,256,8,6,6]。两条分支保留自己的stage4、双眼池化、分类头、CE损失和预测。R&B通过MHD V4在原Node ID上残差写回，两个方向都使用写回前的特征。为了读图，图中把stage1-2合并显示，代码中它们保持独立可见。来源：radonbridge/model.py、graph.py、bridge.py。

## 3. Inside a fixed-channel Radon Bridge

模块图展示固定通道版本：X先经Q转置降通道，在每个空间位置共享同一投影。固定EEM方向和Householder Radon将不同空间维度表示为方向×采样坐标。两来源分别得到[2B,r_i M_i,S]，沿通道拼接，通过kernel=3、无偏置、零初始化Conv1d，再分开。普通直接反投影后乘Q恢复通道，残差写回各自原MHD节点。Q与几何是持久化buffer，梯度可以经过固定算子；固定版本只有中间卷积可学习，原生网络仍全参数训练。桥内没有激活、门控或BN。图中蓝紫两行共享中间mixer，残差旁路的X保持本来源。来源：radonbridge/bridge.py、projector.py、svd_basis.py。

## 4. Compression controls communication width

降维是控制通信规模的实现因素，不是R&B唯一核心。旧可学习CM投影在Radon以后压缩C×M；通道投影在Radon前压缩C，固定SVD/随机QR用Q转置及Q，可学习通道使用独立编码/恢复1×1线性卷积。中心化与非中心化SVD都是已有子空间对照，此介绍只突出固定非中心化主路径。固定通道r=max(1,floor(rho*C))，h=r*M。在C256、M32下，rho=1/16、1/8、1/4对应r16/32/64和h512/1024/2048。rho不是能量阈值。独立来源SVD不构成跨模态语义对齐。来源：bridge.py、learned_channel.py、svd_basis.py及qr_nested_supplement/PROTOCOL.zh-CN.md。

## 5. Two-stage training on paired participants

第一阶段分别监督训练CFP和OCT至平台，各自恢复开发集最佳检查点。第二阶段从这两个检查点加入桥，仍训练原来的两条网络，没有额外第三个特征提取网络。SVD只用这些父网络的训练集stage3特征拟合一次并复用，开发集不拟合基。数据1264训练、296开发，同一UKB记录导出的青光眼临床表型，双眼按参与者整体处理。当前主比较stage3/M32/S64，种子3416-3418，batch16，backbone6e-5，head/bridge1e-4。原规则至少8轮、6轮无>0.001实质改善平台、3轮停滞LR乘0.3、60轮保护上限，未平台不视为完成。该介绍页省略优化细项，完整协议与最终报告保留。来源：experiment.py、convergence.py及已验收manifest。

## 6. Experiments that test the mechanism

实验逻辑：自身处理保持自身块但屏蔽跨来源块，检验跨来源项是否重要。空间打乱与普通重采样检验Radon空间结构的额外贡献，普通重采样匹配卷积宽度和算子行范数，不能称几何或秩完全匹配。两个单向训练都保留自身块，分别删除一个跨来源方向。开发集关闭通信或打乱配对只改变跨来源消息，保持接收者特征和标签；20组无自配对置换，双眼按参与者移动。所有此类干预的结果都需要分支、种子和区间解释，不是临床因果证明。来源：mechanism_benchmark/PROTOCOL.zh-CN.md、communication_analysis.py及qr_nested_supplement/PROTOCOL.zh-CN.md。

## 7. Study map

截至2026-09-05，原186项第二阶段结果及其队列分析已完成。它们由129项既有压缩和机制研究加57项机制/常用方法补充组成。57项包括45项QR普通通信与两基单向训练，及12项MMTM和交叉注意力适配。新A是18项QR自身/空间打乱；新B是9次三方法多宽度联合训练，每个模型评估三档rho得到27个视图。新增训练总27次。全部完成后213个训练配置、231个评估视图，不能把视图数写成独立训练数。页上将补充标为scheduled，表示已授权、排队及验收中的计划，不宣称它们完成。源：两个实时queue_status、manifest及已批准补充协议。

## 8. Research scope and evidence

收尾：研究主线是R&B结构化跨来源通信。需要依次回答连接是否有用、Radon结构是否带来额外收益、解释是否跨基成立、共享宽度训练是否改善表示。所有性能和统计结论留在后续详细结果报告；本介绍不展示精选最佳分数。当前仍是历史参与过配置筛选的开发集探索性证据，同一UKB青光眼目标和ResNet18二维/三维模型，未来需要独立队列、其他任务/模型验证。不能把控制实验或漂亮图示当成完整NBE投稿或录用保证。来源：项目协议、实验接受清单和开发集数据边界。
