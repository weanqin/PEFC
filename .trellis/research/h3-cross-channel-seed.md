# H3 跨通道交互假设 · 下一任务 seed（2026-10-16，来自任务 10-12-gpfm-mamba-edra-f0 收尾移交）

> 供下一个 study1 任务（H3 变体探究）的 brainstorm/research 阶段起步用；非正式 design，最终口径由下一任务与用户收敛。

## 假设定义（用户 2026-10-16 定义与更正）

**H3 = 跨通道交互假设**：v1（d_model=16）GC 一致性（正式口径 0.2755）> v2（d_model=62，0.1227）的可能原因是——v1 的 d16 构成信息瓶颈，**强制跨通道混合**；v2 的 d_model=62 恰等于通道数，in_proj/head 均 62↔62，可学成通道对齐的近似独立映射，跨通道滞后依赖在结构上不被鼓励。

勘误随登：此前 Run 016 曾把「损失侧频带项」误标为 H3 方向，早期高频归因列表的「H3=tv 权重」编号同样作废——两者均不占 H3 编号（见 10-12 experiment.md 判定与收尾块）。

## 证据基础（正式口径 Run 017）

| 维度 | v1 (d16) | v2 (d62) |
|---|---|---|
| GC 一致性 r（补充参考） | **0.2755** | 0.1227 |
| GC 幅值 gen/real | 92% | 89% |
| MAE / CRPS / 逐通道 r / FC r / 空间 | 均次优 | **均最优** |
| R1 终点监控 | 安静（1.243） | 触发（1.526，口径效应注记） |

结构性事实（`models/gpfm-mamba-v1/model.py`）：`in_proj Linear(62+1+16→d_model) → 4×MambaBlock(d_model) → head Linear(d_model→62)`。跨通道（EEG 通道轴）混合只发生在 in_proj 与 head 两个线性层；Mamba 块内部 conv 为 depthwise、SSM 逐特征。v2 唯一超参变化即 d_model 16→62（单变量消融，config 有注记），因此 v1/v2 差异 = 「瓶颈混合 × 容量」的联合变化——**H3 变体设计的核心是把「混合」从「容量」中解耦**。

## 两篇参考论文的机制提取（PDF 存档 `docs/research/papers/`）

1. **Tank et al. 2021, Neural Granger Causality（TPAMI, arXiv:1802.05842）**——联合网络共享隐层无法辨析「哪个输入影响哪个输出」；提出 **component-wise 结构**：每个输出通道 i 一个独立子模型 g_i，输入为**全部** p 个通道的滞后历史 `x⁽⁻ᐟ⁾₁..ₚ`；首层权重按「源通道 × 滞后」列分块 W¹_:j，group lasso / sparse-group lasso / hierarchical lasso 将整列组收缩至零 ⇒ j 不 Granger-cause i（预测器不变性定义）。要点：跨通道影响**结构上可用**（每个输出的预测器看得见全部通道）且**可辨识**（列组结构），稀疏罚用于选择而非注入。
2. **Li & Zhou 2025（iCONECCT, sparse cMLP for iEEG EZ localization）**——同一组件式 + group lasso 用于 iEEG：权重聚合出有向连接矩阵，图特征（outflow 最优，F1 0.941）定位致痫区；消融证明稀疏约束是关键（无稀疏 F1 降至 0.824）。

移植启示：cMLP 的本质 = 「每个输出的预测路径显式地从全部输入通道读出」+ 影响可按源通道分组辨识；对生成式骨干（共享 Mamba trunk）的对应物不是逐通道 62 个独立网络（成本高），而是**在接口处强制混合、在输出处组件式读出**。

## v2 基础上的变体设计空间（推荐 A 起步；命名注意：仓库 v3 已被 v3lp 占用，**新变体应编 v4**——用户口头称「v3」，登记别名即可）

- **A（推荐首选，单变量）输入瓶颈解耦**：保持 d_model=62（容量不变），in_proj 分解为 62+1+16 → r → 62 的低秩瓶颈（r=16，可扫 {8,16,32}）。强制全部通道信息经 r<C 共享潜表示 ⇒ 混合不由容量买单；与 v1（混合+容量同变）/v2（皆无）构成 2×2 因子中的缺失格。实现面小（model.py in_proj 一处），符合薄变体协议。
- **B（cMLP 组件式移植）输出头分组结构**：head 改为组件式——每输出通道 i 独立读出（可从 pooled 全局上下文），读出权重按源通道分组登记，**可选** group lasso 提供「模型内部有向结构 readout」作免费诊断副产品（注意：主轨仍是 Study II 扰动 EC，此 readout 仅诊断用）。
- **C（表示层强制混合）跨通道混合块**：Mamba 块间插入通道轴混合层（逐时间步线性/门控）。混合最深但改动面与归因难度大于 A。
- 评估：与 v1/v2 完全同口径（正式输出侧 40 Hz 低通、evaluate+fc_metric+gc_psd、fold0 单 seed、Run-001 式跑前登记）；核心判据 = GC 一致性能否在保持 v2 五维度优势的同时回升（v1 的 0.276 为参照带）。
- 风险登记：瓶颈 r=16 或伤 MAE（v1 已示容量代价）；GC 提升若以 FC/MAE 恶化为代价，回到「混合-容量-质量」三角权衡，交用户拍板。

## 与研究框架的衔接

- 上游：10-12 任务已证 GPR+Mamba 可行（三检验前置验证通过，v1/v2 均有 5-fold 资格）；5-fold 候选拍板被有意推迟到 H3 变体产出后（候选集 v1/v2/v4 一并取舍）。
- 下游：GC 一致性维度与 Study II 扰动 EC（EC_pert）目标最相关——H3 变体若兑现，θ* 候选大概率落在 v2 系跨通道强制变体上。
- 论文：`docs/research/papers/2021-Tank-NeuralGrangerCausality-TPAMI.pdf`、`docs/research/papers/2025-LiZhou-sparse-cMLP-iEEG-EZ-localization.pdf`。
