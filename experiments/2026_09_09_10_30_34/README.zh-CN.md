# Radon_Bridge：2026_09_09_10_30_34

本时间戳只建立统一命名及Ibex运行基线，不新增训练或重新使用旧test选型。GPU申请按用户“等下”暂停；新数据、模型和任务协议尚未锁定。

## 名称与路径

- 仓库与项目目录：`Radon_Bridge`；Python包：`radon_bridge`；展示：Radon Bridge（R&B）。
- Ibex代码：`/ibex/project/c2377/souray/home/mengh/Radon_Bridge/2026_09_09_10_30_34`。
- Ibex本轮输出：`/ibex/project/c2377/souray/data/mengh/Radon_Bridge/runs/2026_09_09_10_30_34`。
- ws02对应：`/home/mengh/Radon_Bridge`、`/data/mengh/Radon_Bridge`。
- 新目录不加projects层。代码项目目录下按时间戳保存隔离检出，本轮代码目录为本时间戳分支的检出，实验配置位于本文件夹，后续启动训练前记录精确提交并冻结执行源码。
- 共享原始眼科影像仍为`.../souray/data/mengh/UKBiobank/ophthalmology/raw`，改名不触碰正在传输的文件。

`ibex_workspace.json`和`ws02_workspace.json`显式指定代码/数据根目录。新入口不依赖当前工作目录或机器的系统HOME。历史脚本仍属于原协议，不能未经参数适配直接作为新Ibex任务执行。

## 兼容规则

实现目录迁至`radon_bridge`。旧`radonbridge`包仅提供兼容导入，同一模块和类保持对象身份；旧命令行和序列化模块名仍可解析。类名遵循Python的CamelCase习惯，不因目录改名改变数学含义。

ws02旧代码/数据路径保留指向新目录的符号链接，确保历史绝对路径仍有效。历史报告、JSON、预测、摘要和检查点不做文本替换，不改变SHA。GitHub保留仓库历史，历史分支不重新生成。备份盘中的历史证据不迁移。

## MHD依赖

MHD_Project固定至核查时main的`f3f5228d1cddfd35c93e61184cdc65419b8c7712`。其V4实现与旧依赖`8f6651ce9af96f54585bf32bbfd70afe0315867d`逐文件一致，因此本次使用最新仓库中的兼容V4接口。该仓库还包含V5；V5改变节点记忆与反向执行语义，迁移V5需独立验收，不能将本次更名伪装成已完成V5适配。

## Ibex环境与检查

Python 3.11；torch2.8.0、torchvision0.23.0与既有实验主版本相同，其余依赖在安装验收后保存实际版本清单。环境位于`.../home/mengh/environments/radon_bridge_2026_09_09_10_30_34`。不复制ws02的虚拟环境。

```bash
python -m radon_bridge --workspace experiments/2026_09_09_10_30_34/ibex_workspace.json --check-framework
```

此命令只检查目录、依赖和记录已有迁移状态，不加载参与者影像、不训练、不申请GPU。加`--require-verified-images`后，影像尚未SHA验收会返回非零；即使影像已通过，也不代表标签、样本划分、新实验协议已通过。

`ibex_a100_2_48h.sbatch`仅为2张A100/48小时申请模板，不已提交。启动需存在经确认的实际workload，不能用空闲进程占卡。Slurm设置的CUDA_VISIBLE_DEVICES保持不变，不套用ws02的物理GPU编号。未来训练要在实际分配的A100上重新验证显存、梯度、退出与检查点恢复，本轮CPU检查不能替代。

## 尚未完成的科学前置条件

全量影像正在复制和校验；完整标签来源待补。扩大队列、临床标签时间定义、模型替换及新评价协议待与用户确定。当前已完成的UKB研究保持原证据边界，不自动重跑。
