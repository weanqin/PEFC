# 实验日志 — 07-05 口径无偏 global-τ fold0 复现（graph_mexhat ckpt × 10 例）

> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的 outputs 目录。

## Run 001 — 冒烟（1 例 × 16 源）（2026-10-27）

### 目的与假设

- 验证 pre-SC `graph_mexhat` 原版 ckpt 在移植包 `load_model` 的加载路径、无偏
  `SignalProbeSite(use_biased_noise=False)` 通路与 07-05 JSON 键契约（无
  sign/mean_diff/mu_bias 字段）。

### 配置

- 代码：`models/latent-flow-v1/ec_modma_globaltau_unbiased.py`（变体层单文件，包体零改动）
- 参数：口径常量不动，仅 `--subjects 1 --sources 16`
- 环境：base conda @ 本机 GPU 4080 SUPER（cu128），BLAS 守卫 ≤8，`MODMA_DATA_ROOT=/root/autodl-tmp/modma`
- 命令：`cd models/latent-flow-v1 && OMP_NUM_THREADS=8 ... python3 ec_modma_globaltau_unbiased.py --mode main --out-dir <smoke_01> --subjects 1 --sources 16`

### 结果

- 加载 ✓（1.24M params，stage-2 mexhat-graph flow，dataset=modma 自动检测）；
  `μ_bias is None` 断言过；0.57 min/16 源 → 外推 ~4.5 min/例（与实际一致）；
  键契约核验过（method=`global_tau_quantile (signal-space, sigma* preserved)`、
  hparams q=0.5/n_grid=16/n_trials=32/seed=42/use_biased_noise=False，无
  sign/mean_diff/mu_bias 字段）。
- 产物：`outputs/10-27-lf-globaltau-f0-unbiased-repro/smoke_01/subject_0.json`。

### 结论

- 支持开跑：加载/通路/契约三关全过。

## Run 002 — 主跑（fold0 graph_mexhat × 10 例 × 128 源，07-05 全口径）（2026-10-27）

### 目的与假设

- 100% 配置对齐复现旧库 2026-07-05 运行（任务 07-04-global-tau-sigma）：
  fold0 test 10 例 × 32 窗、q=0.5、σ_ref=0.3、σ∈log[0.01,0.3]×**16**、
  n_steps=10、batch=32、seed=42、无偏 ξ~N(0,1)（同源跨 σ 同 ξ）、
  conn=−(ln σ*+½ln2πe)（无 +5/7 平移——v4 为 mild 线决策）。
- 假设：跨栈（Windows/torch2.12/5070Ti → Linux/torch2.8/4080S）数值统计级一致；
  可算锚点（SC/方差分解/A_eff）落在原始值附近。

### 配置

- 代码：同 Run 001 文件（git_head 随 JSON 落档；与移植版 runner 仅三处差异 =
  无偏 site + 07-05 method/键契约 + 无 +5/7 平移，其余逐行同口径）
- ckpt：`/root/autodl-tmp/latent_ode/checkpoints/fold0_oc_latent_flow_flow_best_graph_mexhat.pt`
  （只读；07-04 PRD R6 点名的原版，pre-SC 形态）
- 命令：`cd models/latent-flow-v1 && OMP_NUM_THREADS=8 ... MODMA_DATA_ROOT=/root/autodl-tmp/modma python3 ec_modma_globaltau_unbiased.py --mode main --out-dir <run_01> --subjects 10`

### 结果

- **AC1 兑现**：10/10 例、每例 128/128 源成功 0 失败、45.1 min 总耗时
  （4.51 min/例）；τ_global ∈ [7.67e-5, 9.74e-5]；**subject_9 本机未复现挂起**
  （原 07-05 挂起为环境性问题，本次 10 例全量为原始 9 例验证集的超集）。
- 钳边型态（10 例聚合）：**@σ_min 0.79% / @σ_max 49.8% / interior 49.4%**——
  无偏 μ=0 下近平衡 ~50/50（与 domain.md「q=0.5 is the balanced default」定性
  一致）；对照有偏口径 f2-nosc（10-21）@min 47.3%+@max 44.2% 近二元钳边，
  μ_bias 底噪消除后 @σ_min 塌缩至 <1%（与 mild 线 unbiased 先例同向）。
- 产物：`outputs/10-27-lf-globaltau-f0-unbiased-repro/run_01/subject_{0..9}.json`。

### 结论

- 复现运行成功执行（AC1）；结构性诊断与 07-05 范式定性一致。

## Run 003 — 确定性复跑（subject_0 同配置重跑）（2026-10-27）

### 结果

- **AC4 兑现**：τ、sigma_matrix、connectivity_matrix、raw_response_matrix
  **全部逐位一致**（np.array_equal），diagnostics 相等 → 同机同栈确定性成立。
- 产物：`outputs/10-27-lf-globaltau-f0-unbiased-repro/run_02_determinism/subject_0.json`。

## Run 004 — 锚点核对并置（离线；对照 domain.md §07-05 仅存原始值）（2026-10-27）

### 目的与口径

- 原始逐被试产物已被清理（旧库快照无 `results/oc_connectivity_global_tau_signal_q0.5/`），
  锚点 = domain.md §07-05 聚合值。SC = `sc_prior/sc_elec_directed.pt` 的
  `sc_elec`（128×128 有向，OSF 88HC 组均值 |L|·SC·|L|ᵀ 投影）；A_eff = ckpt
  graph_mixer 经 `effective_connectivity()` 重建（原始 A_eff json 不在快照）；
  方差分解 = 交替行列均值消元（house 口径，同旧库 diag 型）；offdiag Spearman。
- 命令：`... python3 ec_modma_globaltau_unbiased.py --mode anchors --out-dir <anchors> --main-dir <run_01>`

### 结果

| 锚点 | 07-05 原始（9 例） | 本次复现（10 例，mean±std） | 判读 |
|---|---|---|---|
| OC↔SC Spearman（raw, aligned） | +0.31 | **+0.3011±0.0147** | 差 0.009，被试散布内 |
| 行（源）方差 % | 5.7% | **5.61%±1.21** | 差 0.1pp |
| 列（靶）主效应 % | ~29%（domain.md 定性） | **29.10%±5.57** | 吻合 |
| a→b 残差方差 % | 65.5% | **65.29%±4.91** | 差 0.2pp |
| signal-space vs A_eff（重建） | +0.36 | **+0.3238±0.0414**（范围 [0.269, 0.385]） | 原值落在本次被试分布内 |
| 组均值矩阵 vs SC / vs A_eff | —（原始未记） | +0.333 / +0.337 | 补充登记 |
| OC↔GC Spearman | +0.29 | **不可复算**（fold0 逐被试 GC 基线矩阵本机已删，仅存 PNG） | 缺口如实注明 |
| OC↔TE Spearman | +0.28 | **不可复算**（同上） | 缺口如实注明 |

- SC 读出 recipe 侧证：raw aligned（+0.301）与原始值吻合而 abs 变体（+0.004）
  不吻合 ⇒ 原始口径为 raw-vs-raw Spearman（flipped +0.302 ≈ aligned，该 SC
  投影近对称，无方向判别力——描述性登记）。
- **AC2/AC3 兑现**；产物：`outputs/10-27-lf-globaltau-f0-unbiased-repro/anchors/{anchors_summary.json, aeff_rebuilt.npz}`。

### 结论

- **复现成功（统计级全锚点吻合）**：四项可算锚点差值 ≤0.036，方差分解三项
  几乎逐点重合（5.61/29.10/65.29 vs 5.7/~29/65.5）——07-05 矩阵的结构性统计量
  跨栈稳定；复现口径 = 配置/代码路径/seed 100% 对齐（含 n_grid=16 非默认值与
  无偏噪声契约）；跨栈逐位等同不承诺且不可验（原始产物已清理）；同机确定性
  逐位成立（Run 003）。
- 边界重申：复现存证不构成代理脑资格判定；不外推；GC/TE 锚点缺口待 fold0
  基线重算（另行拍板）方可补核。
- 下一步（同步 progress.md）：无强制派生项；GC/TE 补核与 10-25 symw EC 线
  θ 拍板的联动关系不变。

## 环境与溯源

- 本机 RTX 4080 SUPER（32GB）/ torch 2.8.0+cu128 / Python 3.12 / BLAS 守卫 ≤8
  （environment.md 已登记环境）；数据 `MODMA_DATA_ROOT=/root/autodl-tmp/modma`。
- 原始运行溯源：旧库 `.trellis/tasks/archive/2026-07/07-04-global-tau-sigma/`
  （prd/implement）+ `.trellis/spec/domain.md` §07-05（聚合值唯一在档来源）。
