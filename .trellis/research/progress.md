# 科研进度看板

> **维护规则**：SessionStart hook 只注入顶部「当前状态」区；AI 在节点完成/决策后当场更新（保持 ≤20 行）；历史细节追加到下方里程碑区（只增不改），完整过程见 journal。

## 当前状态

- **阶段**：Study I edra-fmt 新预处理范式 fold0 前置验证**完成**（任务 10-16-gpfm-mamba-fmt-f0，2026-10-17）——v1-fmt/v4a-fmt（LayerNorm、p=25、无低通、**genGC vs real GC 阻塞**口径，48 例 34/4/10）均过指标1（FC 0.970/0.981）与指标3（1–30 Hz 0.317/0.301）；v4a-fmt 质量全维度最优（MAE 0.70565 / 逐通道 r 0.12335 超 GPR-r 0.116），候选排序与旧范式一致；无低通生成侧 >40 Hz 占比 ≈ 真实 3×（大体在带内）
- **5-fold 候选集（用户 2026-10-16 拍板）：{v1, v4a}**；fmt 范式迁移后排序不变（判读权重向 v4a 侧）
- **下一任务（待建）**：5-fold 主实验——**范式选择待拍板**（edra 500 Hz 旧口径 vs edra-fmt 新口径 vs 双轨）、折数 × seed 数、GC 阻塞是否延续（fmt 侧解除须先定 250 Hz GC 口径）
- **最近节点**：2026-10-17 任务 fmt-f0 收尾：双变体主 Run（92/55 ep 早停）+ 正式评估（GC 阻塞验收逐文件过）、spec 回填（data/edra-fmt 契约、指标3 频段 1–30 Hz/40–Nyquist 适配、evaluation-calibers fmt 口径、reproducibility 钩子链、data-pipeline 整数分配语义）
- **待拍板**：①指标2 视觉人工判（figs/ 两套）；②5-fold 范式与口径；③GC 阻塞解除时点

## 里程碑（只增不改）

- 2026-10-10 · 研究框架文档定稿（用户提供）；Trellis 科研工作流装配完成；研究记忆初始化 · docs/research/research-direction.md
- 2026-10-16 · GPR+Mamba 可行性肯定（任务 10-12 归档：三检验通过、Q1 v2 最优 MAE 0.701 未超 GPR 基线）· 完整记录见 journal Session 2
- 2026-10-16 · H3 判「不支持」：混合不可局部移植为 GC 载体（接口/读出两臂证伪），5-fold 候选集 {v1, v4a}（任务 10-16 归档）· 完整记录见 journal Session 3
- 2026-10-17 · edra-fmt 新范式 fold0 前置验证通过（指标1/3 过线、v4a-fmt 全维度最优、GC 阻塞中；任务 10-16-gpfm-mamba-fmt-f0）· 完整记录见 journal Session 4
