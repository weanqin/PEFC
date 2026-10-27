# 实验日志 — fold0 AIC GC 基线现算 ×10 例 + OC↔GC 锚点补全

> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的 outputs 目录。

## Run 001 — subject_0 AIC GC 现算 + 旧库残留对拍（2026-10-27）

### 目的与假设

- 补全 10-27-lf-globaltau-f0-unbiased-repro 的 OC↔GC 锚点（原始 +0.29）；先以
  subject_0 验证 GC 再生管线 = 原始口径（旧库残留 `results/baseline_connectivity/
  gc/subject_0.json` 恰为 fold0 test[0] 同被试、AIC 口径）。

### 配置

- 代码：移植版 `latent_flow/baseline/run_baseline_connectivity.py` +
  `granger.py::granger_connectivity`（零改动；`--gc-fixed-order 0` 落入逐对 AIC
  sweep，granger.py:98 哨兵）；CLI 与图件目录重定向本任务 outputs。
- 参数：`--fold 0 --subjects 1 --methods gc --gc-fixed-order 0 --n-jobs 16`
  （AIC max8/nf65/1–125Hz/fs250/全长逐通道 z-score——与残留 method_params 逐参一致）
- 环境：base conda @ 本机（CPU 16 workers，BLAS=1/worker：OMP/OPENBLAS/MKL=1），
  `MODMA_DATA_ROOT=/root/autodl-tmp/modma`
- 命令：`cd models/latent-flow-v1 && OMP_NUM_THREADS=1 ... python3 -m latent_flow.baseline.run_baseline_connectivity --fold 0 --subjects 1 --methods gc --gc-fixed-order 0 --n-jobs 16 --results-dir <gc_aic> --outputs-dir <figs>`

### 结果

- **AC2 兑现（对拍完美）**：现算 subject_0 与旧库残留 **逐位一致**——
  `np.array_equal=True`、max|Δ|=0.0、n_samples 42501 与 method_params 逐参相同；
  4.06 min/例。跨机器（旧库 Windows 生成 vs 本机 Linux 再生）比特级一致 ⇒
  GC 路径（numpy 确定性）跨栈逐位可复现，再生管线即原始口径。

### 结论

- 支持全量开跑：管线身份已被残留件证明，其余 9 例现算值即原始口径值。

## Run 002 — 其余 9 例 AIC GC 现算（2026-10-27）

### 配置

- 同 Run 001，`--subjects 10 --resume`（跳过已完成 subject_0）；33.72 min 完成
  9 例（~3.7 min/例）。

### 结果

- **AC1 兑现**：10/10 例 JSON 齐全（`gc_aic/gc/subject_{0..9}.json`），每例
  128×128 有向矩阵、method_params 全部 = aic/max8/nf65/1–125Hz/fs250/全长。

## Run 003 — OC↔GC 十例并置（vs 原始锚点 +0.29）（2026-10-27）

### 目的与口径

- OC = 10-27 复现 run_01 的 connectivity_matrix（无偏 global-τ，graph_mexhat
  fold0，10 例）；GC = 本任务 AIC 口径 10 例；offdiag Spearman raw aligned 主读出
  （recipe 经 SC 锚点与 subject_0 预览双重侧证）；被试逐配对（subject_file 断言）。

### 结果

| 被试 | Spearman raw aligned |
|---|---|
| s0 `02010013EOG` | +0.3125 |
| s1 `02010025EOG` | +0.2872 |
| s2 `02010002EOG` | +0.3835 |
| s3 `02010028EOG` | +0.3019 |
| s4 `02030007EOG` | +0.2793 |
| s5 `02020023EOG` | +0.3543 |
| s6 `02020020EOG` | +0.3781 |
| s7 `02020026EOG` | +0.0953 |
| s8 `02030011EOG` | +0.2388 |
| s9 `02020018EOG` | +0.0371 |

- **9 例验证集口径（与原始锚点同为 subject_0–8，s9=原挂起例剔除）：
  均值 +0.2923 ± 0.0879 vs 原始 +0.29 —— 差 0.002，精确吻合（AC3）**
- 10 例全量口径：+0.2668 ± 0.1157（s9 +0.037 全场最低，拉低均值——原跑恰在该例
  挂起，锚点本就不含它）；组均值矩阵 raw +0.3787；abs aligned +0.1151 / Pearson
  r_abs +0.1999（再次侧证原始口径为 raw Spearman）；GC directedness ~0.013
  近对称，flipped ≈ aligned（无方向判别力，与 f2-nosc 结论一致）。
- 产物：`outputs/10-27-gc-fold0-aic-anchor/gc_anchor_summary.json`（逐例 + 双口径
  聚合 + 并置 verdict）+ `gc_aic/gc/*.json` + `figs/`（每例热图/直方图）。

### 结论

- **OC↔GC 锚点补全完成并吻合**：至此 07-05 复现的可算锚点从四项扩到五项全吻合
  （SC +0.301 vs +0.31、行方差 5.61% vs 5.7%、靶主效应 29.10% vs ~29%、残差
  65.29% vs 65.5%、**GC 9 例口径 +0.292 vs +0.29**）；仅 TE（+0.28）因基线矩阵
  全删且 TE 无残留件可验而保持缺口。s9 现算成功且为最弱相关例，描述性提示原跑
  挂起例缺失未系统性抬高原始锚点。
- 下一步（同步 progress.md）：无强制派生项；TE 补核需现算 symbolic TE（另拍板）。
- 边界：描述性并置不构成代理脑资格判定。

## 环境与溯源

- 本机 CPU（16 workers cgroup 上限）；GC 路径零 GPU。原始锚点溯源 = 旧库
  `domain.md` §07-05（9 例均值）+ 残留 subject_0 JSON（07-23 再生版，本任务对拍
  逐位命中 ⇒ 07-23 再生版与 07-05 时代生成器同参成立）。
