# latent-flow global-tau fold0 无偏复现（07-05 原始口径 10 例）

## Goal

在当前库本机（RTX 4080 SUPER / torch 2.8）以**配置 100% 对齐**的方式复现旧库 2026-07-05 的
global-tau σ* 扰动归因运行（旧库任务 `07-04-global-tau-sigma`）：fold0 `graph_mexhat` 原版
ckpt + **无偏零均值噪声** + `global_tau_quantile` 两阶段法，fold0 test 10 例；产物入 outputs
随档，并把**本机可复算的锚点**与 domain.md 仅存的原始聚合数值并置核对。

## Background（侦察已钉死的事实）

- **原始运行**（旧库 `.trellis/tasks/archive/2026-07/07-04-global-tau-sigma/` + `spec/domain.md` §07-05）：
  - ckpt：`checkpoints/fold0_oc_latent_flow_flow_best_graph_mexhat.pt`（PRD R6 点名；本机
    `/root/autodl-tmp/latent_ode/checkpoints/` 在位 ✓，5,085,003 B）
  - 方法：两阶段 global-τ σ*，`SignalProbeSite` 信号空间，**无偏 ξ~N(0,1)**（有偏噪声 07-23
    才引入，本口径早于其存在）；method 名 `global_tau_quantile`
  - 配置：fold0、q=0.5、**n_grid=16**（非脚本默认 24）、n_trials=32、seed=42、
    σ∈log[0.01,0.3]（脚本默认）
  - 被试：fold0 test 10 例；**原跑 subject_9 挂起被杀，验证集实为 9/10**（domain.md 明文）
- **原始逐被试产物已被清理**（`results/oc_connectivity_global_tau_signal_q0.5/`、
  `de_diag_global_tau_q0.5/` 均不在旧库快照）→ 无逐位对比基准。仅存锚点（domain.md 表，9 例）：
  | 锚点 | 原始值 |
  |---|---|
  | OC↔GC Spearman | +0.29 |
  | OC↔TE Spearman | +0.28 |
  | OC↔SC Spearman | +0.31 |
  | source（行）方差 % | 5.7% |
  | a→b 残差方差 % | 65.5% |
  | signal-space vs A_eff | +0.36 |
- **锚点可复算性**（本机盘点）：
  - ✓ SC：`/root/autodl-tmp/latent_ode/results/sc_prior/sc_elec_directed.pt` 在位
  - ✓ 行/残差方差分解：自身矩阵即可算（median-polish 型行/列/残差分解，旧库 diag 口径）
  - ✓ vs A_eff：A_eff json 不在位，但可从 ckpt graph_mixer 权重 + geometry.pt 重建
    （pairwise_d 已由 10-21-nosc-check 验证入位 `/root/autodl-tmp/modma/geometry.pt`）
  - ✗ GC / TE：fold0 逐被试基线矩阵本机只剩 PNG（outputs/baseline_connectivity_fold0/），
    锚点 +0.29/+0.28 本任务**不核**，如实注明（重算基线属另行拍板）
- **跨栈 caveat（预期管理，写入交付物）**：原始 = Windows/Python 3.13/torch 2.12/5070 Ti，
  本机 = Linux/Python 3.12/torch 2.8/4080 SUPER。同 seed 同代码路径下数值应统计一致，
  **不承诺逐位相等**（f2-nosc 先例即"同型"复现）。"100%" 的口径 = 配置/代码路径/seed 100%
  对齐 + 可算锚点统计级核对。
- **实现差异点**：当前库移植版 runner
  (`models/latent-flow-v1/latent_flow/scripts/run_oc_connectivity_global_tau.py`) 硬编码
  `use_biased_noise=True`（07-23 后 HEAD 形态）→ 需变体层无偏入口；JSON 键按 07-05 原版契约
  （connectivity_matrix/sigma_matrix/method=global_tau_quantile/tau/q/raw_response_matrix/
  hparams/checkpoint，无 sign/mean_diff/mu_bias 字段——07-05 早于符号提取存在）。

## Requirements

- **R1 无偏变体入口（包体零改动）**：变体层单文件（`models/latent-flow-v1/` 根，先例
  `ec_mild_unbiased.py`）：复用移植包 runner 的加载/两阶段逻辑，仅改
  `SignalProbeSite(use_biased_noise=False)`、method 名与 JSON 键按 07-05 原版契约；
  `--fold 0 --subjects 10 --n-trials 32 --q 0.5 --n-grid 16 --seed 42`，
  `--checkpoint` 显式指向 graph_mexhat 原版 ckpt。
- **R2 主跑 10 例**：GPU 后台，逐例 JSON 落 `outputs/10-27-lf-globaltau-f0-unbiased-repro/run_01/`；
  预计 ~4.5–6.5 min/例。subject_9 若复现挂起：如实记录挂起现象并按当年 9/10 口径汇总，
  **不得静默跳过**。
- **R3 锚点核对并置表**（能算的全算）：OC↔SC Spearman、行/列/残差方差分解、钳边诊断
  （@σ_min/@σ_max%）、以及 R4 的 vs A_eff；与 domain.md 原始值并置，跨栈 caveat 注明。
- **R4 A_eff 重建**：从 ckpt graph_mixer（graph_mexhat 原版，γ_sc=0，具体系数以 ckpt 实际
  buffer 为准）+ `geometry.pt` pairwise_d 重建 128×128 A_eff，Spearman offdiag vs conn →
  与 +0.36 并置。
- **R5 本机确定性自检**：同配置复跑 subject_0 一遍，两 JSON 数值逐位一致（同机同栈应可达成；
  若不一致如实登记并分析）。
- **R6 登记**：experiment.md（格式契约）+ 研究记忆回填（context/progress 路线行 + journal）。

## Acceptance Criteria

- [ ] **AC1**：10 例（或 9 例 + subject_9 挂起如实记录）JSON 产物齐、每例 128/128 源成功 0 失败、
  method=`global_tau_quantile`、hparams 与原始口径逐键一致（q=0.5/n_grid=16/n_trials=32/seed=42）。
- [ ] **AC2**：R3 并置表落档（SC/方差分解/钳边三项至少），GC/TE 不可复算的缺口如实注明。
- [ ] **AC3**：R4 完成：A_eff 重建 + vs A_eff Spearman 并置。
- [ ] **AC4**：R5 确定性自检结论明确（逐位一致 / 不一致+原因）。
- [ ] **AC5**：experiment.md + 研究记忆回填完成；跨栈 caveat 与"不构成代理脑资格"边界写入交付物。

## Notes

- Lightweight 任务，PRD-only（先例：10-21-latent-flow-globaltau-f2-nosc）。

## 子研究路由（1.0b / 1.0c）

- **分类结果**：`study1`（EEG 代理脑，框架 Study I）
- **依据**：任务主体是 latent-flow 代理脑（旧库 ckpt）上的扰动归因复现存证，服务于 Study I
  候选族的口径档案，不引入新 EC 推断算法。
- **新路径/方法检测（1.0c）**：`已有路径：latent-flow（context.md 技术路线表 Study I 候选行）`
  ——复现 07-05 已有口径，无新方法。
- **research 产物**：`无`
