# fold0 AIC GC 基线现算 ×10 例 + OC↔GC 锚点补全

## Goal

补全 10-27-lf-globaltau-f0-unbiased-repro（07-05 无偏 global-τ 复现）的 OC↔GC 锚点：
现算 MODMA fold0 test 10 例 GC 基线（**AIC 口径 = 原始 07-05 锚点同参**），与复现的
OC conn 矩阵算逐例 Spearman 并与原始锚点 **+0.29**（9 例均值）并置。

## Background（侦察已钉死）

- **口径依据**：旧库残留 `results/baseline_connectivity/gc/subject_0.json`（fold0
  test[0]，2026-07-23）method_params = `granger_bivariate_spectral, order_select: aic,
  max_order: 8, n_freq: 65, freq_band: full, 1–125 Hz, fs250, 全长 42501 样本`——即
  07-05 时代基线口径（gc-corr-f2nosc 任务已实证该时代为 AIC）。
- **代码路径**：移植版 `latent_flow/baseline/run_baseline_connectivity.py` +
  `granger.py::granger_connectivity`；`fixed_order=0` 落入逐对 AIC sweep
  （granger.py:98 `fixed_order is not None and fixed_order >= 1`），参数 echo
  `order_select: aic` 与残留一致。**零代码改动**（纯调 CLI + 一个只读核对脚本）。
- **对照物**：OC = `outputs/10-27-lf-globaltau-f0-unbiased-repro/run_01/subject_{0..9}.json`
  的 connectivity_matrix（10 例齐全）；GC subject_0 残留可用于**管线对拍验证**。
- **上轮已得的 n=1 预览**（残留 GC × 复现 subject_0）：Spearman raw aligned +0.3125
  （vs 原始 +0.29）——本任务把它做全 10 例。

## Requirements

- **R1 GC 现算 10 例（AIC 口径）**：`run_baseline_connectivity --fold 0 --subjects 10
  --methods gc --gc-fixed-order 0 --n-jobs 16`，结果/图件目录重定向到本任务 outputs；
  预计 ~9 min/例（AIC ≈ fixed8 的 ~2.25×，16 workers cgroup 上限）≈ 1.5 h CPU。
- **R2 管线对拍**：subject_0 现算结果 vs 旧库残留 JSON（同数据同参同代码系）——
  数值应高度一致（float 级），max|Δ| 与相对差登记；若显著偏离须停跑归因。
- **R3 OC↔GC 10 例并置**：逐例 Spearman（raw aligned 主读出 + abs/flipped 附带）
  + 组均值±std + 组均值矩阵读出，与原始锚点 +0.29 并置（跨栈 caveat 沿用）。
- **R4 登记**：experiment.md + 研究记忆回填（progress 里程碑 + 上一任务锚点缺口
  状态更新）。

## Acceptance Criteria

- [ ] **AC1**：10 例 GC JSON 齐全，method_params 逐参 = 残留口径（aic/max8/nf65/
  1–125Hz/fs250/全长）。
- [ ] **AC2**：subject_0 对拍结论明确（max|Δ|/相对差数值登记，同源判定成立）。
- [ ] **AC3**：OC↔GC 10 例并置表落档（逐例 + 组均值 vs +0.29）。
- [ ] **AC4**：experiment.md + 研究记忆回填完成。

## Boundaries

- 只算 GC（TE +0.28 锚点不在本令范围，如需另拍板）；不动旧库/已有产物；
  不构成代理脑资格；描述性并置不外推。

## 子研究路由（1.0b / 1.0c）

- **分类结果**：`study1`
- **依据**：为 latent-flow MODMA 线复现任务补全对照锚点，属 Study I 候选族口径档案。
- **新路径/方法检测（1.0c）**：`已有路径：latent-flow（复用移植版 granger 主路径，无新方法）`
- **research 产物**：`无`
