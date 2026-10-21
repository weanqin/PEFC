# research 适配文档 — 07-04-global-tau 方案 × gpfm-mamba-v5-main 代理脑

> 1.0c 新路径适配结论（本会话侦察产出，2026-10-21）。源锚点：
> 07-04 = `models/latent-flow-v1/latent_flow/scripts/run_oc_connectivity_global_tau.py` +
> `latent_flow/attribution/bisect_sigma.py`（零改动 import 复用）；
> 代理脑 = `models/gpfm-mamba-v5/`（θ* = `outputs/gpfm-mamba-f0/v5-main-01/checkpoints/gpfm-mamba-v5-best.pt`，best_val 0.73155@ep64）。

## 1. 可行性结论

**可行，且有直接先例**：`outputs/10-21-v5main-ec-explore/gen_ec_explore.py`（2026-10-21 归档）已实现
「加性高斯扰动 x^p 通道 → 固定 (eps,eta) 采样 → 逐靶响应」的完整适配模式（其方法1/方法2），
其中方法2 的 +0.5·std 常量偏移即 07-04 μ_bias 的特例。本任务把该模式补齐为 07-04 的完整两阶段
global-tau 流水线（σ 网格 + 全局 τ + σ\* 搜索 + 熵映射 + 方向符号），σ 搜索原语**直接 import
latent_flow 的 `bisect_sigma.py`（log_grid / walk_monotone_curve / compute_global_tau / _HALF_LOG_2PIE），
不自实现**，保证与 07-04 逐位同构。

## 2. 两源包接口事实（已核实）

### gpfm-v5 侧（数据与采样契约）
- 数据：`data_root=/root/autodl-tmp/edra_process_v4`，62ch @250 Hz，窗 = past 384 + future 128、stride 64；
  fold0、`subject_filter=status_ok`（46 例）、split seed 42、划分固化校验 `models/gpfm-mamba-v1/splits/`；
  数据管线逐被试逐通道 z-score（run 目录 `channel_mean.npy/channel_std.npy`）。
- fold0 test 10 例（`iterate_test_batches` 断言清单）：**[12, 17, 18, 24, 30, 38, 44, 55, 58, 61]**。
- 采样：`gpfm_v1_sample.EulerSampler(model, prior, euler_steps=cfg.euler_steps, sigma_min=cfg.sigma_min,
  gen_lowpass_hz=…)`；`sample(x_past (B,C,384), K, eps (B,K,C,128), eta (B,K,C,128))` → (B,K,C,128)；
  **给定 (x_past, eps, eta) 完全确定**（Euler 确定性），v5-main-01 run 口径 `euler_steps=16, sigma_min=1e-4`；
  sampler 构造沿 explore 先例（`gen_lowpass_hz=getattr(cfg,'gen_lowpass_hz',40.0)`）。
- 加载：v1+v5 双 path 遮蔽（`import config/model` 命中 v5，data/gpr_prior/sample 命中 v1），
  `load_eval_model(CKPT, cfg, device)`；`cudnn.deterministic=True`。

### latent-flow 侧（07-04 原语，模型无关）
- `bisect_sigma.compute_global_tau(response_matrix, q)`、`log_grid(σ_min, σ_max, n)`、
  `walk_monotone_curve(mse_curve, grid, τ, σ_min, σ_max)`、`_HALF_LOG_2PIE`——全部只依赖 numpy/torch 张量
  形状约定，与代理脑无关，**逐字复用**。
- `SignalProbeSite.perturb_and_measure` 的契约（返回 (mse_vec (C,), mean_diff_vec (C,))）作为
  本任务 `GpfmProbeSite` 的实现规范。

## 3. 设置映射表（07-04 → 本任务；「同」= 逐字复用）

| 07-04 设置 | 07-04 值 | 本任务 | 依据 |
|---|---|---|---|
| 扰动算子 | 加性高斯 σ·ξ + 固定偏置 μ_bias=0.5·ch_std（`use_biased_noise=True`） | **同** | 用户拍板「完全复用」 |
| 扰动位置 | z-score 后、入模前、x_obs 段（源通道 a） | **同**（x^p 段 384 样本） | 用户拍板 |
| 度量 | per-target MSE（对 batch,time 取均值） vs 确定性基线 | **同**（x^f 段 128 样本） | 用户拍板 |
| τ 标定 | σ_ref=0.3 一遍扫源 → 全局 q=0.5 分位 | **同** | 用户拍板 |
| σ 搜索 | log_grid[0.01,0.3]×24 + 单调走查 + 线性插值 + 边界 clamp | **同**（import 原语） | 用户拍板 |
| 强度映射 | −(log σ\* + ½log 2πe)；sign=mean-shift ±1/0 | **同** | 用户拍板 |
| CRN | ξ 按源播种（seed+source）跨 σ 复用；基线确定性 | **同** + gpfm 生成噪声 (eps,eta) 跨源/跨 σ 固定（被试级播种）| 幅度-only 比较前提 |
| Euler 步数 | OC `n_steps=10` | gpfm `euler_steps=16`（v5-main-01 训练/评估口径） | 代理脑自身口径，映射登记 |
| 窗口 | MODMA 500(375+125) @128ch | edra v4 512(384+128) stride 64 @62ch | 代理脑数据契约 |
| 被试/折 | splitter seed=42 | fold0 test 10 例（v1 固化 manifest 断言） | 代理脑划分契约 |
| n_trials | 默认全量（历史跑 32） | **briefing 拍板**（32 / 50 / 930） | 成本-口径权衡 |

## 4. 风险与对策

- **生成噪声混入**：若 (eps,eta) 随 σ 变动，MSE(σ) 混入生成方差 → 曲线非幅度-only。对策：被试级固定
  (eps,eta)（探索先例同款），并在 runner 内断言其跨 σ 逐位一致。
- **σ 量纲**：x^p 已 z-score（std=1），07-04 网格 [0.01,0.3] 语义不变，无需重标定。
- **确定性**：cudnn deterministic + 固定种子全链（explore 先例）；验收含同 seed 复跑逐位一致检查。
- **GC 对照口径**：refs = `refs/granger_bivariate_spectral/edrav4-46-.../refs_s{sid}.npz`（ec 行=源），
  10/10 test 例在档；aligned 主口径 + flipped 诊断（explore 惯例）；ρ 仅描述性。
- **成本**：62 源 × (1+24) 前向 × N 窗，Euler 16 步/前向，行分块 cap=1000（explore 同款）。
  32 窗估 ~3-6 min/例；930 窗估 ~1.5-3 h/例。

## 5. 代码落点（path-isolation）

新路径包 `models/ec-globaltau-v1/`（runner + GpfmProbeSite 适配层）：
**import-only 复用** latent_flow-v1（bisect_sigma 原语）与 gpfm v5/v1（模型/采样/数据），
两源包零逻辑改动；产物落 `outputs/10-21-ec-globaltau-v5main/`。

## 6. 评审修订补记（trellis-review 2026-10-21）

- **【阻断 B1】explore 先例实跑为 raw weights**：`gen_ec_explore.py` 取 `ema` 后从未 `swap`，其 m1/m2
  数字系 raw-weights 口径——与本任务（EMA 正典口径）**不可数值直比**。本任务全部前向在
  `with ema.swap(model):` 内执行，S1 加 swap 生效断言。
- **【阻断 B2】数值精度口径**：07-04 全程 bf16 autocast；gpfm 正典与 explore 均 fp32。本任务经 briefing 拍板（2026-10-21）钉定
  **EMA + bf16 autocast**（镜像 07-04 精度口径；对 gpfm 正典 fp32 为登记偏离），τ/σ\* 为阈值穿越量，
  精度口径单点声明并随 evaluation-calibers 登记。
- 其余建议项（schema 字段补全/μ_bias 母体/x_0 术语消歧/perturb-ec-pipeline §1 偏离声明/
  calibers 登记/后台化/路径清单/R2 措辞）已并入 design/implement/prd 相应节。
