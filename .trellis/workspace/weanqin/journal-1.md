# Journal - weanqin (Part 1)

> AI development session journal
> Started: 2026-10-10

---



## Session 1: baseline EC 观测参考管线（GC+gPDC）与扰动 EC 主轨比较器交付
<!-- trellis-session: v=2 fp=a93cf8635350ae72 -->

**Date**: 2026-10-16
**Task**: baseline EC 观测参考管线（GC+gPDC）与扰动 EC 主轨比较器交付
**Branch**: `main`

### Summary

任务 10-13-baseline-ec-v1 完成（已归档）：EDRA 46 例 125Hz 降采全量产出 GC（ridge 条件 ln SSR 比）与 gPDC(|gPDC|²) 两参考矩阵；关键定标=OLS 在 62ch 下 Gram 条件数 ~1e18、split-half 天花板≈0 → ridge λ_rel=1e-2 + 固定 p=8（ASCT 锚点），天花板救回 gc 0.900±0.030 / gpdc 0.734±0.039；mne-c 二元交叉核对 vs 条件 GC 0.069±0.080（estimand 分歧量化）。主轨比较器（距离偏相关 Spearman 主指标 + t×Holm + bootstrap + 天花板比值）以合成 EC_pert 联调通过；EC_pert readout 约束（低剂量斜率主/AUC 次/r* 降级删失）与术语纪律（observational references）登记入 evaluation-calibers/surrogate-qualification。TE 全系取消（用户拍板，16 核 cgroup 实测 S=50/100 需 35-70 天）；环境契约新增 cgroup 16 核事实 + _blas_guard 线程护栏惯例。真实 EC_pert 对比与三角诊断留待 Study II。

### Git Commits

| Hash | Message |
|------|---------|
| `d951ed9` | feat(baseline-ec-v1): 观测参考 EC 管线（GC+gPDC 两参考法）+ 扰动 EC 主轨比较器 + split-half 天花板 |
| `a034135` | docs(baseline-ec-v1): Run 002 配置块回填代码提交哈希 d951ed9（amend 后自引用修正） |

### Status

[OK] **Completed**


## Session 2: gpfm-mamba-edra-f0 收尾：三检验判定通过、H3 勘误与移交、任务归档
<!-- trellis-session: v=2 fp=7439ce763720d59a -->

**Date**: 2026-10-16
**Task**: gpfm-mamba-edra-f0 收尾：三检验判定通过、H3 勘误与移交、任务归档
**Branch**: `main`

### Summary

任务 10-12-gpfm-mamba-edra-f0 完成并归档（GPR+Mamba 可行性立项问题收肯定回答）。①三检验判定（用户 2026-10-16 拍板）：指标 1 FC 一致性通过（标准=组平均 r≥0.95，v1 0.9586/v2 0.9763）、指标 2 视觉 v1/v2 均过、指标 3 PSD 通过（标准=低频段 δ–β |Δlog10|≤0.5+目视；θ/β 补登 0.30–0.38，α 随边界漂移 0.12–0.18 但判定稳健）——通过标准回填 surrogate-brain-qualification.md 消除占位。②H3 勘误（用户更正定义）：H3=跨通道交互假设（v1 d16 瓶颈强制混合 vs v2 d62 通道对齐可独立映射，解释 GC 一致性 0.276>0.123）；三处历史误标勘误（Run 016 损失侧频带项误挂 H3、Run 012 tv-权重假设让位、Run 016 幅度括注训练期口径误用于生成侧——实测过冲 +23%/+29%）。③H3 移交下一任务：读 Tank 2021（cMLP 组件式+列组 group lasso）与 Li & Zhou 2025（sparse cMLP iEEG EZ 定位）两篇论文提取强制跨通道机制，写 seed（.trellis/research/h3-cross-channel-seed.md，v4 设计空间 A 输入瓶颈解耦/B 组件式输出头/C 混合块，推荐 A；论文存档 docs/research/papers/）。④5-fold 暂缓（用户拍板，候选集待 H3 变体扩充为 v1/v2/v4 一并取舍）。trellis-check 全量检查 PASS（1 Major 幅度表述+2 Minor 已修）。附带：前会话遗留 trellis 设施（1.0b/1.0c 路由门禁/data spec/nature-figure）补入库。

### Git Commits

| Hash | Message |
|------|---------|
| `aafb013` | chore(trellis): 科研工作流设施装配（前会话完成未入库）——1.0b/1.0c 子研究路由门禁（workflow.md+config.yaml hook+seed-research-routing 播种脚本）、data spec 接入（edra 数据集契约 index+data-pipeline.md）、nature-figure 技能 |
| `7a1ab8a` | docs(task-10-12): 前置验证三检验判定通过、H3 勘误、任务收尾——指标 1/3 通过标准回填 spec（组平均 r≥0.95 / 低频段 δ–β ≤0.5+目视，用户 2026-10-16 拍板）；H3 更正为跨通道交互假设（v1 d16 瓶颈强制混合 > v2 d62 通道对齐；三处历史误标勘误），seed+两篇 cMLP 论文存档移交下一任务；任务总结论：GPR+Mamba 可行性肯定（Q1 v2 最优 MAE 0.701 未超 GPR 基线、生成 std_t 过冲 +23%/+29% 勘误登记；Q2 三检验通过）；5-fold 暂缓（候选集待 H3 变体扩充） |
| `11f8a20` | chore(task): 归档 10-12-gpfm-mamba-edra-f0（GPR+Mamba 可行性任务收尾：三检验通过、H3 移交下一任务、5-fold 暂缓） |

### Status

[OK] **Completed**

### Next Steps

- 下一任务（待用户启动）：H3 跨通道交互假设变体探究（v2 基础 cMLP 式 v4，seed 就绪）；随后 5-fold 候选集拍板（v1/v2/v4）
