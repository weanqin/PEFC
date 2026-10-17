# 科研进度看板

> **维护规则**：SessionStart hook 只注入顶部「当前状态」区；AI 在节点完成/决策后当场更新（保持 ≤20 行）；历史细节追加到下方里程碑区（只增不改），完整过程见 journal。

## 当前状态

- **阶段**：adaLN(t) 调制归一化消融**完成（中性结果）**（任务 10-17-gpfm-mamba-adaln-f0，2026-10-17）——v4a-fmt 基座上 5 个 LayerNorm 位换 adaLN（emb(t)→逐 site SiLU→Linear 零初始化，A 案 scale/shift，fork_rng 恒等初始化），与 v4afmt-main-01 **全维度持平**（MAE 0.70571 vs 0.70565、FC 0.9797 vs 0.98147、指标3 0.30605 vs 0.30134 均 ✓，Δ≤0.005）；54ep 早停 best 44；epoch-0 损失与基线逐位相等（fork_rng 契约兑现）→ **该规模下 t 条件调制无增量收益，候选集 {v1, v4a} 排序不变**
- **5-fold 候选集（用户 2026-10-16 拍板）：{v1, v4a}**；fmt 范式与 adaLN 消融后排序均不变
- **下一任务（待建）**：5-fold 主实验——**范式选择待拍板**（edra 500 Hz 旧口径 vs edra-fmt 新口径 vs 双轨）、折数 × seed 数、GC 阻塞是否延续（fmt 侧解除须先定 250 Hz GC 口径）
- **最近节点**：2026-10-17 任务 adaln-f0 收尾：主 Run（54ep 早停）+ 正式评估（中性结论）+ trellis-check 通过（6 处记录笔误修复）；spec 回填（reproducibility RNG 中性构建惯例 + cond_adaln 钩子链、context 技术路线登记）
- **待拍板**：①指标2 视觉人工判（figs/ 三套：v1fmt/v4afmt/v4afmt-adaln）；②5-fold 范式与口径；③GC 阻塞解除时点

## 里程碑（只增不改）

- 2026-10-10 · 研究框架文档定稿（用户提供）；Trellis 科研工作流装配完成；研究记忆初始化 · docs/research/research-direction.md
- 2026-10-16 · GPR+Mamba 可行性肯定（任务 10-12 归档：三检验通过、Q1 v2 最优 MAE 0.701 未超 GPR 基线）· 完整记录见 journal Session 2
- 2026-10-16 · H3 判「不支持」：混合不可局部移植为 GC 载体（接口/读出两臂证伪），5-fold 候选集 {v1, v4a}（任务 10-16 归档）· 完整记录见 journal Session 3
- 2026-10-17 · edra-fmt 新范式 fold0 前置验证通过（指标1/3 过线、v4a-fmt 全维度最优、GC 阻塞中；任务 10-16-gpfm-mamba-fmt-f0）· 完整记录见 journal Session 4
- 2026-10-17 · adaLN(t) 消融中性结果（全维度持平、fork_rng 契约兑现 epoch-0 逐位相等、排序不变；任务 10-17-gpfm-mamba-adaln-f0）· 完整记录见 journal Session 5
