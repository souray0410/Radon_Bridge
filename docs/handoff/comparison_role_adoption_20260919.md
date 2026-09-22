# LOOK / Radon_Bridge 比较角色采用记录（R&B侧）

依据 PHD standards/research.md 2026-09-19“外部方法对照必须对应所研究的问题”及本周 comparison_scope_correction.zh-CN.md。本记录只澄清跨项目比较角色，不改R&B既有结果、coverage、scientific cutoff、训练队列或test策略。

- LOOK 的 A：真正处理整模态缺失的既有策略；主问题是同一个合理训练 A 与同一个 A + LOOK。MMTM在LOOK中的既有结果只作宿主兼容/组装证据，不计缺失方法A/B/C。
- Radon_Bridge 的 A：跨维/跨模态交流方法；主问题是同一个交流方法 A 与同一个 A + bridge。R&B已有MMTM加桥包仍按其原科学问题和原验收范围有效，不因LOOK角色纠正而失效。
- 两项目不得用同一个“A”标签混合解释结果；必须先写清所研究对象是“缺失处理”还是“跨维/跨模态通信”。
- GAN、缺失鲁棒微调等只是LOOK缺失方法的类别例子，不构成R&B比较清单，也不是LOOK机械必跑表。
- 本记录不授权任何新训练、GPU任务或方法扩展；具体候选和公平匹配协议由各项目新的有限任务包锁定。

LOOK侧采用记录与一手候选盘点保存在LOOK研究分支 webcodex/probability-readiness-20260919。
