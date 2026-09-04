# 新拓扑验收与启动记录

验收源码08807b8。以下是实现正确性证据，不是疾病预测性能结果。

- 同一attach_to_nodes接口通过2/3/4/5参与者、异构通道、1D/2D/3D/4D和不同网络深度的自动接入。
- 每组只新增一个通信结果节点，原生Node ID和所有原Edge ID/端点保留；返回超边共处通信之后的下一level。原下游仍读原节点，无updated替代节点。
- 通信组连续调用三次、网络重复forward/backward通过；MHD/native梯度一致；初始化零桥与无桥预测精确相同。
- self控制阻断跨网络通信，random控制新拓扑验收通过；CFP独立训练与配对无桥的两次AdamW更新参数差均为0。
- 全解冻H1/8真实batch4的全部51,350,916个参数可训练，两个方向跨任务梯度到达另一stem；MHD/native最大梯度差0，BN统计更新。峰值allocated2230.71MiB、reserved2426MiB。稳定step约0.054–0.055秒。
- 旧展开布局H1/8、1/16、1/32验收保留在acceptance_h*.json；当前拓扑证据为topology.json、requirements.json、full_h8.json、random.json、optimization.json。

正式协议：5种子×7臂×16epoch；H1/8主配置，H1/16和1/32带宽对照，self/scrambled/random均匹配H1/8；stage3、M/S固定，两条网络所有层联合训练。统一轮数在新性能结果出现前固定。

预算记账更新：资格9.603、旧中止1.738、所有验收保守预留2分钟，共13.341；正式最多226.5分钟，累计最多239.841GPU分钟。新布局略快，仍用之前更慢的step估计约213.15分钟，不提高任务量。

数据1264训练/296开发验证，290测试未读。旧冻结与部分解冻试验不作为本批方案性能证据，也不继承其检查点。开始状态须看/data/mengh/RadonBridge/runs/2026_09_04_15_23_46/status.json。
