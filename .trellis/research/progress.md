# 科研进度看板

> **维护规则**：SessionStart hook 只注入顶部「当前状态」区；AI 在节点完成/决策后当场更新（保持 ≤20 行）；历史细节追加到下方里程碑区（只增不改），完整过程见 journal。

## 当前状态

- **阶段**：Study II 流程首验证**完成**（任务 10-17-perturb-ec-f0，2026-10-17，v4a-fmt fold0 冻结 θ\*·前置验证级偏离经用户签认）——四要素首版落地（A 案增益算子 + CRN-RMS 度量 + τ 双口径 v2 操作性 τ̃=0.25 + readout 三件套），流程端到端跑通（10 例 27.7 min、r=0 同批逐位恒等、跨 Run 逐位复现、100% 单调）；**AC6 如实判定：该 θ\* 上未获结构收敛初步证据**（slope×GC ρ̄=−0.007 / slope×gPDC ρ̄=+0.022 Holm p=0.112），缺口四重证据定位 **θ\* 跨通道传导弥散无特异性**（非流程/readout/参考侧，ceiling 0.96、可检 ρ̄≈0.028 power 足够）——为 Study I 损失侧迭代提供直接判据；流程规范首版已沉淀 spec/research/perturb-ec-pipeline.md
- **5-fold 候选集（用户 2026-10-16 拍板）：{v1, v4a}**；fmt 范式与 adaLN 消融后排序均不变
- **下一任务（待建）**：①5-fold 主实验——范式选择待拍板（edra 500 Hz 旧口径 vs edra-fmt vs 双轨）、GC 阻塞是否延续；②或 Study I 损失侧迭代（跨通道特异性，Study II 缺口判据驱动）——先后关系待拍板
- **最近节点**：2026-10-17 任务 perturb-ec-f0 收尾：B0-B6 全链（smoke/bench/主 Run/readout+诊断七件套/参考侧+split-half/比较器）+ trellis-check 通过（1 Major 数值笔误修复）+ spec 首版沉淀
- **待拍板**：①Study II 缺口应对路径（换代理脑复跑 / B 案算子 / 其他）与 5-fold 先后关系；②指标2 视觉人工判（figs/ 三套 + Study II 诊断图 8 张）；③GC 阻塞解除时点；④r\*→连接强度最终映射（框架第 14 步）

## 里程碑（只增不改）

- 2026-10-10 · 研究框架文档定稿（用户提供）；Trellis 科研工作流装配完成；研究记忆初始化 · docs/research/research-direction.md
- 2026-10-16 · GPR+Mamba 可行性肯定（任务 10-12 归档：三检验通过、Q1 v2 最优 MAE 0.701 未超 GPR 基线）· 完整记录见 journal Session 2
- 2026-10-16 · H3 判「不支持」：混合不可局部移植为 GC 载体（接口/读出两臂证伪），5-fold 候选集 {v1, v4a}（任务 10-16 归档）· 完整记录见 journal Session 3
- 2026-10-17 · edra-fmt 新范式 fold0 前置验证通过（指标1/3 过线、v4a-fmt 全维度最优、GC 阻塞中；任务 10-16-gpfm-mamba-fmt-f0）· 完整记录见 journal Session 4
- 2026-10-17 · adaLN(t) 消融中性结果（全维度持平、fork_rng 契约兑现 epoch-0 逐位相等、排序不变；任务 10-17-gpfm-mamba-adaln-f0）· 完整记录见 journal Session 5
- 2026-10-17 · Study II 流程首验证：四要素首版 + 端到端跑通 + AC6 如实阴性（缺口=θ\* 跨通道弥散，四重证据）；流程规范首版沉淀 spec/research/perturb-ec-pipeline.md（任务 10-17-perturb-ec-f0）· 完整记录见 journal Session 6
