# design — 07-04-global-tau × gpfm-mamba-v5-main

> 评审修订（trellis-review 2026-10-21，阻断 B1/B2 + 建议项已吸收）：θ\* 数值口径钉定 = **EMA 权重 + bf16 autocast**（briefing 拍板 2026-10-21，镜像 07-04 精度口径；对 gpfm 正典 fp32 为登记偏离）；
> 入模过去窗统一记 `x_0`（= `x_past`/`x_past_pert`），并显式区分 EulerSampler 内部 ODE 起点 x0（先验采样变量，非本任务语义）。

## §1 架构

新路径包 `models/ec-globaltau-v1/`（3 文件，import-only）：
- `adapter.py`：双 path 遮蔽加载 gpfm v5（config/model/graph）+ v1（data/gpr_prior/sample），
  `build_runner(cfg, run_dir)` → (sampler, test_ds, ema)，加载模式沿 `outputs/10-21-v5main-ec-explore/gen_ec_explore.py` 先例；
  **θ\* 数值口径 = EMA**：全部前向（baseline + 扰动）在 `with ema.swap(model):` 内执行
  （正典口径，v5-main-01 best_val 0.73155 为 EMA 口径；注意 explore 先例实跑未做 swap = raw weights，
  其 m1/m2 数字与本任务**不可数值直比**——research 文档已补记）。S1 含 swap 生效断言。
- `probe_site.py`：`GpfmProbeSite`——实现 07-04 `SignalProbeSite.perturb_and_measure` 同款契约
  （`set_source(a)`：播种 ξ_a (N,384) + μ_bias；`perturb_and_measure(σ, a)` → (mse_vec (62,), mean_diff_vec (62,))）；
  baseline `x^f_base = sample(x^p, eps, eta)` 构造期算一次（swap 域内）。
- `run_ec_globaltau.py`：镜像 `run_oc_connectivity_global_tau.py` 两阶段主流程
  （Stage1 σ_ref 扫描 → `compute_global_tau` → Stage2 网格 σ\* → conn/signed/JSON），
  原语 import 自 `latent_flow.attribution.bisect_sigma`。
  **数值精度 = bf16 autocast（briefing 拍板 2026-10-21：镜像 07-04 的 autocast 结构——autocast 域包裹
  前向、扰动算术在输入侧；对 gpfm 正典 fp32 为登记偏离；τ/σ\* 为阈值穿越量，精度口径单点声明）**。

## §2 数据流（用户钉定，PRD 硬约束 3/4/5 的展开）

```
                z-score 后（数据管线 channel_mean/std）
x_raw(62,N) ────────────────→ 滑窗 x_0 窗 = [x^p (62,384) | x^f* (62,128)]
                                   │
   ┌───────────────────────────────┤  扰动施加点（唯一）：x^p_pert = x^p + σ·ξ_a + μ_bias^a
   │                               │   （仅通道 a；ξ_a ~ N(0,1) 按源播种 seed+source，跨 σ 复用；
   │                               │     μ_bias^a = 0.5·std_a，std 母体 = 实际入模 x^p 窗集
   │                               │     std(dim=(0,1))——同 07-04 ch_std 定义，随 n_trials 取值）
   │                               ▼
   │                    x_0 ≡ x^p_pert（入模恒为扰动版；注意 ≠ EulerSampler 内部 ODE 起点 x0）
   │                               │
   │                    [v5-main θ* EulerSampler(euler_steps=16, sigma_min=1e-4)，EMA swap 域内，bf16 autocast]
   │                    sample(x_0, K=1, eps, eta)  ← (eps,eta) 被试级固定，跨源/跨 σ 逐位同
   │                               ▼
   │                          x^f_pert (62,128)
   │
   └─（无扰动分支）x_0 ≡ x^p ─→ [同 θ*，同 (eps,eta)，同 swap/autocast 域] ─→ x^f_base

MSE_b^a(σ) = mean_{win,time}(x^f_pert − x^f_base)²,  mean-shift_b^a 同构（μ 差）
   ↓ Stage1（σ=σ_ref=0.3 全源扫）           ↓ Stage2（log_grid(0.01,0.3,24) 逐源）
   τ_global = Quantile_{q=0.5}(全矩阵)  →  σ*_ab = walk_monotone_curve（标量 τ，clamp [0.01,0.3]）
   ↓
conn_ab = −(log σ*_ab + ½log 2πe)；EC_pert = sign(mean-shift) ⊙ conn
```

断言（写进 runner）：①ξ_a 跨 σ 逐位一致；②(eps,eta) 跨源/跨 σ 逐位一致；③扰动仅落在 x^p 通道 a
（x^p 其余通道与任何 x^f 切片在入模前不被触碰——x^f 本就不入模，由数据管线切窗保证）。

## §3 设置复用对照

见 `research/globaltau-v5main-adaptation.md` §3 映射表（prd R2 引用）。唯一开放拍板项 = n_trials
（briefing：32 / 50 / 全量 930）；其余非「同」项均为代理脑契约映射：`euler_steps=16`、窗 384/128、
fold0 test 10 例、**EMA 权重、bf16 autocast**（阻断 B1/B2 修法，briefing 拍板）。

## §4 确定性契约

- `torch.backends.cudnn.deterministic=True / benchmark=False`；BLAS 守卫 OMP/OPENBLAS/MKL=8。
- **θ\* 口径**：EMA swap 全前向生效（S1 断言 swap 前后输出不同、swap 后 = EMA 正典）；**bf16 autocast**（07-04 同构：autocast 域包裹前向，扰动算术在 fp32 输入侧）。
- 种子：ξ_a = seed+a（07-04 同式）；(eps,eta) = 被试级 `seed*1e4+sid` 播种一次；同 seed 复跑 σ\* 矩阵逐位一致（AC4）。
- **同输入同批形跨调用逐位一致**断言（替代 perturb-ec-pipeline.md D(r=0)=0 恒等断言；本任务按
  07-04 形态为 baseline 构造期一次 + 每 (源,σ) 单独调用，属**有意偏离**该规范 §1 的批内成对
  调用惯例，偏离随档声明）。
- 采样行分块 cap=1000（explore 同款，reflect-pad grid 上限内）。

## §5 产物与对照

- per-subject JSON：字段对齐 **07-04 schema 全集**（subject_idx/subject_file/n_trials/
  n_sources_requested/n_sources_succeeded/n_source_failures/method/tau_global/q/sigma_ref/
  sigma_matrix/connectivity_matrix/connectivity_formula/raw_response_matrix/sign_matrix/
  signed_connectivity_matrix/mean_diff_matrix/**mu_bias_per_channel**/noise_model/grid/hparams/
  checkpoint/diagnostics/elapsed_min/timestamp），新增 `surrogate` 块（θ\* 路径、euler_steps、
  sigma_min、(eps,eta) 种子、**ema_weights: true、amp: "bf16-autocast"**）。
- GC 对照（若 briefing 纳入）：`refs/.../edrav4-46-.../refs_s{sid}.npz`（行=源），aligned 主 Pearson/Spearman
  + flipped 诊断，描述性登记。

## §6 成本预算

62 源 × (1+24) 前向 × N 窗，Euler 16 步，cap 1000 行/批：32 窗 ≈ 3–6 min/例（10 例 ≈ 0.5–1 h）；
930 窗 ≈ 1.5–3 h/例（10 例 ≈ 15–30 h，须 nohup 后台化 + 日志落 `outputs/<任务>/run_01/logs/`）。
冒烟 1 例先行。

## §7 读写路径清单（path-isolation）

- **写**：`models/ec-globaltau-v1/`（新包）、`outputs/10-21-ec-globaltau-v5main/run_01/`（JSON/logs）、
  本任务 `.trellis/tasks/10-21-ec-globaltau-v5main/`。
- **只读**：`models/latent-flow-v1/`（bisect_sigma 原语 import）、`models/gpfm-mamba-v5/` +
  `models/gpfm-mamba-v1/`（模型/采样/数据 import）、`/root/autodl-tmp/edra_process_v4`（数据）、
  `outputs/gpfm-mamba-f0/v5-main-01/`（ckpt/config/channel_mean/std）、
  `refs/granger_bivariate_spectral/edrav4-46-resample250-.../`（对照）。
- 两写落点均为本任务新占（评审已实测无冲突）；两源包零逻辑改动。
