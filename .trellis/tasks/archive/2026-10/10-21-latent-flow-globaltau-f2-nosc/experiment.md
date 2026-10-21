# 实验日志 — latent-flow global-tau fold2 nosc 本机重跑

> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的 outputs 目录。

## Run 001 — global-tau σ\* 归因 EC 推断首跑（fold2 nosc × 1 例 × 32 窗）（2026-10-21）

### 目的与假设

- 用 07-04-global-tau-sigma 方案（global-quantile-tau σ\* 归因，旧库 EC 推断主口径）在本机以 fold2 nosc（MODMA SOTA）ckpt 首跑 1 例，验证本机端到端可复跑并存证诊断指标（τ、边界钳位、耗时）。
- 假设：几何/加载堵点解除后（前任务 10-21-latent-flow-modma-nosc-check），该流水线应直接可跑；诊断量级应与旧库 fold0 历史例同型（σ\* 近二元钳边）。

### 配置

- 代码：`latent_flow/scripts/run_oc_connectivity_global_tau.py` + `latent_flow/attribution/bisect_sigma.py`（零改动，经 `run.py` 包装）
- 参数：`--fold 2 --subjects 1 --n-trials 32`；其余 07-04 默认：q=0.5、σ_ref=0.3、σ∈log[0.01,0.3]×24 点、SignalProbeSite 信号空间 + `use_biased_noise=True`（μ_bias=0.5×通道 std，方向符号）、n_steps=10、batch=32、splitter seed=42
- 环境：base conda @ 本机 GPU 4080 SUPER（cu128），BLAS 守卫 OMP/OPENBLAS/MKL=8，`MODMA_DATA_ROOT=/root/autodl-tmp/modma`
- 数据：fold2 test 第 1 例 `02010010EOG.set_61_.mat`（169 窗 → 子采样 32，`subsample_trials` seed=42 逐例确定性）
- 命令：
  ```bash
  cd /root/npi/models/latent-flow-v1 && OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  MODMA_DATA_ROOT=/root/autodl-tmp/modma MPLBACKEND=Agg PYTHONUNBUFFERED=1 python3 run.py \
  run_oc_connectivity_global_tau --fold 2 \
  --checkpoint /root/autodl-tmp/latent_flow/checkpoints/fold2_oc_latent_flow_flow_best_graph_mexhat_screg_l0.01_w0.1_nosc.pt \
  --subjects 1 --n-trials 32 --output-dir /root/npi/outputs/10-21-latent-flow-globaltau-f2-nosc/run_01
  ```

### 结果

- **AC1 兑现**：`n_sources_succeeded=128 / n_source_failures=0`，JSON 落盘；被试 = fold2 test[0] `02010010EOG.set_61_.mat`（169 窗 → 32，与 probe 首跑同被试）
- **核心诊断**：τ_global(q=0.5) = **5.594e-4**；边界钳位 **@σ_min 47.3% + @σ_max 44.2% = 91.5% 近二元**（旧库 fold0 历史例 44.4%+47.9%=92.3%，同型）；σ\* offdiag 中位 = 0.0100（= σ_min 最强钳位）
- **读出矩阵**：connectivity offdiag 中位 3.186（= −(log 0.01+½log2πe)，σ_min 钳位值）/ min −0.215（σ_max 钳位值）/ max 3.186；signed 后负 57.6% / 正 42.4%（μ_bias 均值偏移方向符号）
- raw response @σ_ref offdiag：中位 6.10e-4 / max 5.81e-3；spatial：corr(|conn|, 电极距离) = **−0.015（无距离衰减结构）**
- 耗时 **6.52 min/例**（32 窗口径；旧库 5070 Ti 同口径 17.7 min）
- 产物：`outputs/10-21-latent-flow-globaltau-f2-nosc/run_01/subject_0.json`（含 σ\* 矩阵/connectivity/signed/raw response/sign/μ_bias/全超参）

### 结论

- 支持：07-04 方案本机端到端跑通（AC1–AC3 兑现），诊断量级与旧库历史同型（近二元钳边 ~92%）；本机算力下 6.5 min/例，10 例对标旧库档预计 ~1.1 h，可行
- 描述性并置（不做对比交付）：单例无空间衰减结构与 npi 侧两条阴性路径的「跨通道传导非特异」判读同向；但本任务未跑 GC 对照（r_abs 类指标需 10 例档 + baseline refs 接入，属后续拍板）
- 边界重申：单例存证不外推；不构成代理脑资格
