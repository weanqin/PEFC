# 科研进度看板

> **维护规则**：SessionStart hook 只注入顶部「当前状态」区；AI 在节点完成/决策后当场更新（保持 ≤20 行）；历史细节追加到下方里程碑区（只增不改），完整过程见 journal。

## 当前状态

- **阶段**：双变量 GC 观测参考落地（任务 10-17-bivar-gc-ec-fmt，2026-10-17 完成）——edra-fmt 48 例成对双变量 GC EC 矩阵 + split-half 天花板 0.9243±0.0447 已落盘可复用；**不做 EC_pert 比较**（用户第四轮拍板），比较器接入与 fmt 条件 GC 滞后决策
- **5-fold 候选集（用户 2026-10-16 拍板）：{v1, v4a}**；fmt 范式与 adaLN 消融后排序均不变
- **Study II 状态（沿用户 2026-10-17 定论）**：当前方法（v4a-fmt θ\*）EC 完全不可用、不在此 θ\* 上继续，优化留待后续任务；流程/四要素契约/规范首版已沉淀可复用
- **下一任务（待建）**：①5-fold 主实验——范式选择待拍板（edra 旧口径 vs edra-fmt vs 双轨）、GC 阻塞是否延续；②或 Study I 损失侧迭代（跨通道特异性，Study II 缺口判据驱动）——先后关系待拍板
- **最近节点**：2026-10-17 bivar-gc-ec-fmt 完成（B0–B5 全链 + trellis-check PASS-with-minors + evaluation-calibers 双变量口径条目；BIC 100% 顶格 p\*=24 实测登记）
- **待拍板**：①下一任务先后关系（5-fold vs Study I 跨通道迭代）；②指标2 视觉人工判；③GC 阻塞解除时点；④r\*→连接强度最终映射；⑤fmt 条件 GC/比较器接入时点 + 双变量 BIC 网格扩展（滞后决策）

## 里程碑（只增不改）

- 2026-10-10 · 研究框架文档定稿（用户提供）；Trellis 科研工作流装配完成；研究记忆初始化 · docs/research/research-direction.md
- 2026-10-16 · GPR+Mamba 可行性肯定（任务 10-12 归档：三检验通过、Q1 v2 最优 MAE 0.701 未超 GPR 基线）· 完整记录见 journal Session 2
- 2026-10-16 · H3 判「不支持」：混合不可局部移植为 GC 载体（接口/读出两臂证伪），5-fold 候选集 {v1, v4a}（任务 10-16 归档）· 完整记录见 journal Session 3
- 2026-10-17 · edra-fmt 新范式 fold0 前置验证通过（指标1/3 过线、v4a-fmt 全维度最优、GC 阻塞中；任务 10-16-gpfm-mamba-fmt-f0）· 完整记录见 journal Session 4
- 2026-10-17 · adaLN(t) 消融中性结果（全维度持平、fork_rng 契约兑现 epoch-0 逐位相等、排序不变；任务 10-17-gpfm-mamba-adaln-f0）· 完整记录见 journal Session 5
- 2026-10-17 · Study II 流程首验证：四要素首版 + 端到端跑通 + AC6 如实阴性（缺口=θ\* 跨通道弥散，四重证据）；流程规范首版沉淀 spec/research/perturb-ec-pipeline.md（任务 10-17-perturb-ec-f0）· 完整记录见 journal Session 6
- 2026-10-17 · 双变量 GC 观测参考落地：48 例 EC 矩阵 + 天花板 0.9243±0.0447（BIC 顶格 p\*=24 如实登记；不做 EC_pert 比较）· 完整记录见 journal Session 7
