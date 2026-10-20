# 科研进度看板

> **维护规则**：SessionStart hook 只注入顶部「当前状态」区；AI 在节点完成/决策后当场更新（保持 ≤20 行）；历史细节追加到下方里程碑区（只增不改），完整过程见 journal。

## 当前状态

- **阶段**：study2 NI-MSE-50-q0.90 路径首执行完成（任务 10-20-ni-mse-v5fmt-f0，2026-10-20）——θ\* = v5-fmt-01（fmt 管线，与 refs 同管线 caveat 消除），三期 null/perturb/report 跑通（2.04 h），流程契约全兑现（R2 同批恒等 ×10 例×4 臂、split-half 0.914–2.000%≪10%、per-例断点续跑、种子树四支含 η）；**结构结论阴性**：G 门控全被试 0/3,782、S offdiag 中位 ~0.45（跨通道响应低于噪声地板、剂量曲线全平）、自响应亦低于地板；refs 相关微弱 r≈0.05–0.10 且 flip≈direct（方向信息≈0）⇒ **该 θ\* × 该路径 × 钉定参数（c≤0.5/0.1s 窗）下 EC_pert 不可用**；方法层沉淀（可交换 null 灵敏度权衡 + 设计 §4.2 勘误 + RNG 契约成本归因）入 spec/research/perturb-ec-pipeline.md §7
- **Study I 候选现状**：gpfm 5-fold 候选集 {v1, v4a}（10-16 拍板）；v5/v5-fmt 轴分解交替变体完成（v5-main-01 0.73155 / v5-fmt-01 0.70242，θ\* 候选）；latent-flow edra_fmt θ 候选就绪（10-19 训练 run）但**三项合格性检验未做**——三个候选族均未取得代理脑资格
- **Study II 状态**：两条 EC_pert 路径均阴性（perturb-ec-v1 增益算子 @v4a-fmt、NI-MSE 噪声注入 @v5-fmt-01），共同指向代理脑侧跨通道传导弱；流程/契约/规范（含 NI-MSE §7）可复用
- **暂停任务**：10-19-gpfm-mamba-v4b-gnn-f0（in_progress 暂停待恢复，未竟 = S4'-S6' 收尾 + Run 017 同口径评估）
- **下一任务（待建，先后待拍板）**：①Study I 迭代（损失侧跨通道监督——两条阴性路径的共同判据指向）；②latent-flow 合格性检验（θ 候选就绪）；③5-fold 主实验（范式/口径待拍板）
- **待拍板**：①下一任务先后关系；②指标2 视觉人工判；③GC 阻塞解除时点；④r\*→连接强度最终映射；⑤NI-MSE 若复跑的优化方案（D 预取零数值变化可直接用；A/B/C 数值口径变化须变更登记）；⑥10-19 暂停任务恢复时点

## 里程碑（只增不改）

- 2026-10-10 · 研究框架文档定稿（用户提供）；Trellis 科研工作流装配完成；研究记忆初始化 · docs/research/research-direction.md
- 2026-10-16 · GPR+Mamba 可行性肯定（任务 10-12 归档：三检验通过、Q1 v2 最优 MAE 0.701 未超 GPR 基线）· 完整记录见 journal Session 2
- 2026-10-16 · H3 判「不支持」：混合不可局部移植为 GC 载体（接口/读出两臂证伪），5-fold 候选集 {v1, v4a}（任务 10-16 归档）· 完整记录见 journal Session 3
- 2026-10-17 · edra-fmt 新范式 fold0 前置验证通过（指标1/3 过线、v4a-fmt 全维度最优、GC 阻塞中；任务 10-16-gpfm-mamba-fmt-f0）· 完整记录见 journal Session 4
- 2026-10-17 · adaLN(t) 消融中性结果（全维度持平、fork_rng 契约兑现 epoch-0 逐位相等、排序不变；任务 10-17-gpfm-mamba-adaln-f0）· 完整记录见 journal Session 5
- 2026-10-17 · Study II 流程首验证：四要素首版 + 端到端跑通 + AC6 如实阴性（缺口=θ\* 跨通道弥散，四重证据）；流程规范首版沉淀 spec/research/perturb-ec-pipeline.md（任务 10-17-perturb-ec-f0）· 完整记录见 journal Session 6
- 2026-10-17 · 双变量 GC 观测参考落地：48 例 EC 矩阵 + 天花板 0.9243±0.0447（BIC 顶格 p\*=24 如实登记；不做 EC_pert 比较）· 完整记录见 journal Session 7
- 2026-10-18 · 双变量 GC 观测参考 250 Hz 原始全长档落地：refs-01 48 例 + 天花板 0.8981±0.1414 + rev-01 指纹 + AC6 raw bytes 级复现（条件数越限代决策续跑如实登记；不做跨口径对比）· 完整记录见任务 `.trellis/tasks/10-18-bivar-gc-fmt-250/experiment.md`
- 2026-10-18 · latent-flow 先前原型移植完成（study1 新候选路径）：OC-LatentFlow 代理 + bisection σ 归因代码面入 `models/latent-flow-v1/`，裁剪 A+bundle 备份、零源码改动、G1/G2 全过、trellis-check AC1–AC5 通过；MODMA 勘误（modma_resting_process.zip 可读，已解压在位）· 完整记录见任务 `.trellis/tasks/10-18-latent-flow-port/experiment.md`
- 2026-10-19 · latent-flow × edra_fmt 首个正式训练 run：npz shim + manifest 只读复用 + 两阶段 50 epoch 稳定收敛（stage1 1.5526 / stage2 mexhat CFM 0.5977），θ 候选产出、产物隔离、AC1–AC7 全过（D0/D1/D2/D4 四项用户拍板；S2.2 几何判定门如实触发后真实拍板 npy 基座）· 完整记录见任务 `.trellis/tasks/10-19-latent-flow-edra-fmt-train/experiment.md`
- 2026-10-20 · v5-fmt 变体完成（任务 10-20-gpfm-mamba-v5-fmt-f0）：v5 结构 × fmt 口径 + 早停 10（用户拍板），152,957 参数/单测 11/11，主 Run v5-fmt-01 ep39 早停 best 0.70242@ep29，A_eff/W 快读出随档，θ* 候选就绪待 study2 NI-MSE 接入
- 2026-10-20 · （10-19-gpfm-mamba-v4b-gnn-f0 会话补档）v5 变体落地 + v5-main-01 直训完成（best 0.73155@ep64）；GC 计算主路径拍板 = 原库 GC + refs 归档启用 `npi/refs/`（首批 fmt 48 例 EC 矩阵）；NI-MSE-50-q0.90 设计定稿 v1.5（用户五要素逐项拍板）
- 2026-10-20 · NI-MSE-50-q0.90 首执行（任务 10-20-ni-mse-v5fmt-f0）：v5-fmt-01 θ\* × c{0.05,0.1,0.3,0.5} × Q90 门控，流程契约全兑现（R2/split-half/续跑）；**阴性主读出**（G 全 0/3782、S~0.45、剂量全平、refs r≈0.05–0.10 flip≈direct）⇒ 该配置下 EC_pert 不可用；可交换 null 灵敏度权衡与 RNG 契约成本归因入 spec §7 · 完整记录见任务 `.trellis/tasks/10-20-ni-mse-v5fmt-f0/experiment.md`
