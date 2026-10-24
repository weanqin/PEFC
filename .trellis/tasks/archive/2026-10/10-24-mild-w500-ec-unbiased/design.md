# design — latent-flow × mild-w500 无偏噪声扰动归因 EC（global-τ 两阶段）

任务：10-24-mild-w500-ec-unbiased。PRD 见 prd.md；执行清单见 implement.md。
本设计 = 用户 2026-10-24 指令的「完整扰动实验方案」技术版（范式/参数/流程/判据）。

## 0. 引用清单（直接输入，全部只读）

| 来源 | 用途 |
|---|---|
| `models/latent-flow-v1/latent_flow/attribution/bisect_sigma.py` | 全部算法原语：`SignalProbeSite`（无偏扰动+逐靶 MSE probe）、`log_grid`、`compute_global_tau`、`sigma_star_with_global_tau`、`per_target_mse` |
| `models/latent-flow-v1/latent_flow/scripts/run_oc_connectivity_global_tau.py` | 两阶段流程结构模板（stage1/stage2 的 autocast 包裹与循环形态） |
| `models/ec-globaltau-v1/run_ec_globaltau_unbiased.py` + `run_unbiased_response_curve.py` | 无偏口径先例（μ=0、q=0.25@0.3、conn 式、响应曲线诊断与 q ladder） |
| `models/latent-flow-v1/train_mild_depress.py` | `load_manifest` / `compute_mild_train_norm` / `MildTrialDataset` / `build_model_mild` / 常量 |
| `models/latent-flow-v1/viz_mild.py` | `_model_args` / `load_flow_ckpt`（strict 加载） |
| `models/latent-flow-v1/quality_check_mild.py` | 口径断言与 provenance 登记形态模板 |
| `outputs/10-24-latent-flow-mild-w500/run_w500_fold0/checkpoints/` | 代理脑 ckpt（只读） |
| `.trellis/spec/research/visualization.md` | smoke 曲线 PNG 风格契约 |
| `.trellis/research/environment.md` | BLAS 线程/GPU/数据盘契约 |

## 1. 总体数据流

```
manifest(fold0) ─→ train_f ─→ compute_mild_train_norm(window=500) ─→ (mean,std)
test 8 例 × 30 trial ─→ MildTrialDataset ─→ ds.trials[i] (500,128) z 域
                                                │ 取前 375 点
                                                ▼
build_model_mild(stage=2, obs=375, fc=125) + load_flow_ckpt(w500 best)
                                                ▼
SignalProbeSite(model, x_obs (30,375,128), use_biased_noise=False)
   ├─ compute_baseline（隐式，构造时）: sample(x_obs) → x_fc_base (30,125,128)
   ├─ stage1: 逐源 a: perturb_and_measure(σ_ref=0.3, a) → response[a,:] (128,)
   │           → τ_global = quantile_q(flatten(response))
   └─ stage2: 逐源 a: 逐 σ∈log[0.01,0.48]×24: perturb_and_measure → mse_curve (24,128)
              → sigma_star_with_global_tau → σ*[a,:] → conn = −(ln σ*+ln 2πe)+1
```

## 2. 扰动算子（与 07-04 正典逐字对齐点）

- **施加位置**：z-score（train 折统计）之后、入模之前的 x_obs 段（375 点）源通道 a；
  生成侧输出全窗的 fc 尾（125 点）为测量对象；真实 x_f 不参与（EC 为代理脑内部量）。
- **方式**：加性 `x_pert[:,:,a] = x_obs[:,:,a] + σ·ξ`，ξ~N(0,1) 形状 (N_trials, 375)，
  `torch.Generator(seed + a)` 播种；**同一 ξ 跨该源全部 σ 与 σ_ref 复用**（幅度-only
  比较，噪声样本效应不混入）。其余通道逐位不动（clone + 单通道赋值构造保证）。
- **时序安排**：扰动覆盖整个 obs 段 375 点（1.5 s @250 Hz）逐点持续注入；响应在 fc 段
  125 点（0.5 s）全程测量（对 (trial, time) 取均值）。无扰动基线与扰动前向共享同一
  确定性采样链路（确定性采样 ⇒ 无 eps/eta 自由度，基线唯一、delta 干净——这正是
  bisect_sigma 设计前提，见其 docstring「OC-LatentFlow stage-2 flow path is
  deterministic (z_obs = mu_obs)」）。
- **μ_bias 关闭**（`use_biased_noise=False`）：无偏口径定义；mean_diff/sign 字段随档
  保存但不解读（无 07-23 方向性设计依据——unbiased 先例同款登记）。
- **与 gpfm 适配层（GpfmProbeSite）的差异**：gpfm 采样随机（eps/eta 被试级固定注入），
  latent-flow 确定性采样 ⇒ 无需该自由度；channel 布局原生 channel-last，无需 (B,C,F)
  摆位。除此之外口径逐字一致。

## 3. 两阶段 global-τ 算法与 conn 映射

- **stage1（每被试）**：每源 1 次前向 @ σ_ref=0.3 → response (128,128)；
  `τ_global = compute_global_tau(response, q)`（flatten 后 q 分位，finite-only）。
  **q = 0.5（用户 2026-10-24 拍板钉死：保持默认 0.5，不测试不计算其他取值；原方案
  q=0.25 + ladder 校准作废）**。
  全矩阵共享同一阈值 ⇒ σ* 跨 (a,b) 对可直接比较（07-04 global-τ 修正的要点：
  消除 per-source 阈值导致的源主效应主导）。
- **stage2（每被试）**：`grid = log_grid(0.01, 0.48, 24)`；每源 24 次前向 →
  mse_curve (24,128)；`sigma_star_with_global_tau(mse_curve, grid, τ, 0.01, 0.48)`：
  单调非减走查 + 跨界线性插值；边界语义：MSE@σ_min>τ ⇒ σ*=σ_min（最强连接）；
  MSE@σ_max<τ ⇒ σ*=σ_max（最弱连接）。
- **conn 映射**：`conn[a,b] = −(ln σ*[a,b] + ln 2πe) + 1`（unbiased 口径登记式，
  与 v5-main unbiased 逐字一致；σ* 小 ⇒ conn 大 = a→b 强）。σ* 原矩阵随档保存 ⇒
  `1/(σ*+1e-4)` 等单调重映射可纯离线重算（Spearman 不变，先例已内置断言）。
- 对角线（自响应）随档保存但不解读；offdiag 128×127 = 16256 对为 EC 主体。

## 4. 执行形态（校准门取消，q 已由用户钉死）

- **工程 mini 干跑（S2，非科学口径测试）**：`--mode main --sources 8 --n-grid 4` 单例
  ~1 min，校验代码正确性：输出形状（8×8 σ*/conn）、口径断言链、无 NaN、conn 值域
  合理、凭据字段齐全。通过即进主跑，**不做任何 q 取值测试**。
- **主跑顺带曲线落盘（后置诊断，非 gate）**：main 模式每例保存完整 mse_curve
  (128, 24, 128) + resp_ref npz（~1.6 MB/例，零额外前向——stage2 本就逐点计算）。
  Run 002 从落盘曲线**事后重算**健康诊断：floor/ceil 分位、dyn_ratio（v5-main
  unbiased 参照 137.9×）、nonmono_pair_frac、interior/@min/@max（q=0.5 下实测值照登）、
  **P(MSE@σ_ref>τ) 实测值**（区分「τ 位置性钳边」vs「响应动态范围不足」——评审 A
  建议采纳）。任一指标退化的结论照实登记（负证据链惯例），**不回改 q、不熔断重跑**
  （用户口径）。**τ 分位口径含对角自响应**（flatten 全 128×128，与 unbiased 先例
  一致；占比 0.8% 可忽略）——payload/README 注明，防与 offdiag-only 口径误比（评审 A）。
- **主跑规模（用户拍板）**：8 例 × 30 trial 全量直接使用；预算 ≤ 30 min 量级（MODMA
  Run 002 同族参照），护栏 60 min；GPU 显存 ≪ 32 GB（模型 1.24M 参数 + 30 trial 批）。
  **逐例断点续跑**：subject_{i}.json 已存在即跳过（评审 B 采纳）。
- **实施硬断言（评审 B 采纳）**：脚本内钉死 window/obs/fc = 500/375/125 三元组断言
  （防 train_mild_depress 模块常量 1500/1125/375 误用——`compute_mild_train_norm`
  缺省 window_len=1500 是静默错统计路径）；重算 norm 的 mean_std SHA256 与训练
  config 登记值对拍（不等即拒跑）。stage1/2 循环自行包 autocast（site 不管理 dtype，
  模板 run_oc_connectivity_global_tau.py）。

## 5. 数值口径

- bf16 autocast 包裹采样前向（与训练 AMP 及既有 EC 线一致）；扰动算术在输入侧 fp32
  （autocast 域外 clone+加法，先例同款）；`nan_to_num` 护栏（sample 内置 + 测量侧
  float() 提升）；`cudnn.deterministic=True`、`benchmark=False`。
- 统计（分位/中位/方差分解）一律 float64、纯 CPU 离线。
- BLAS 线程上限 8 在任何 numpy/torch 导入前设置（16 核 cgroup 契约，quality_check_mild
  同款守卫）。

## 6. 输出 schema（subject_{i}.json，对齐 07-04/unbiased 先例并扩展凭据）

核心字段：`subject_idx/subject_file(sid)/n_trials/n_sources(128)/method/tau_global/q/
sigma_ref/sigma_matrix/connectivity_matrix/connectivity_formula/raw_response_matrix/
sign_matrix/mean_diff_matrix/noise_model("unbiased_gaussian_mu0 (sigma*xi additive on
z-scored x_obs, before model)")/grid("log_spaced")/hparams(q,sigma_ref,sigma_min,
sigma_max,n_grid,n_steps=10,batch_size,seed)/checkpoint`；
诊断：`diagnostics{n_at_sigma_min/max, pct_*, pct_interior, sigma_median/q25/q75}`；
凭据扩展：`provenance{manifest_path+sha256, norm_files_hash+mean_std_sha256, window
(500/375/125), ckpt_path, git_head, surrogate_stage2_best_val(0.8356@E150 登记借注)}`。

## 7. 新增文件与复用清单

- **唯一新增**：`models/latent-flow-v1/ec_mild_unbiased.py`（变体层单文件，三模式：
  `--mode main|summary|gccorr`；CLI 断言口径 obs+fc==window==ckpt 训练口径、subjects ⊆
  fold test、out-dir 必填无默认防误写旧树；mini 干跑 = `main --sources 8 --n-grid 4`）。
- import 复用（零拷贝）：`bisect_sigma.{SignalProbeSite, log_grid, compute_global_tau,
  sigma_star_with_global_tau}`；`train_mild_depress.{load_manifest,
  compute_mild_train_norm, MildTrialDataset, build_model_mild, MANIFEST_DEFAULT,
  _sha256_file, WINDOW/OBS_LEN/FORECAST_LEN}`；`viz_mild.{_model_args, load_flow_ckpt}`；
  `latent_flow.training.seeding.set_seed`。
- summary / gccorr 并入同文件（纯离线，读 main 产物重算），不另立文件。

## 8. 风险与缓解

| 风险 | 判据/缓解 |
|---|---|
| R-1 跨通道传导弱（多条 EC_pert 阴性路径共同指向）→ 响应平/钳边退化 | 曲线后置诊断（Run 002）如实登记；阴性结果照登（负证据链惯例）；用户口径 = 不熔断不回改 q |
| R-2 stage2 顶格未收敛 → EC 结构可能随续训漂移 | caveat 全档登记（payload 借注）；不以此定论代理脑 |
| R-3 bf16 对小 MSE delta 精度 | 房内既有口径（v5main/latent-flow 线均 bf16）；扰动算术 fp32；dyn_ratio 后置照登 |
| R-4 每源单条 ξ 的样本方差 | 房内设计（同 ξ 跨 σ 幅度-only 比较）；多 ξ 重复登记为边界外 |
| R-5 与 10-24 任务共享变体目录（其 in_progress + 未提交脏路径） | 仅新增文件零改旧文件；commit 只圈本任务文件；其脏路径零触碰 |
| R-6 geometry 依赖（`/root/autodl-tmp/Mild Depression/geometry_mild.pt`） | build_model_mild 内置存在性断言（缺即拒跑）；w500 训练已用过，在档已确认 |
| R-7 GC 参考与 EC 不可比风险（GC = 观测数据量，EC = 代理脑内部量；实现/口径不同） | 对照为描述性并置（先例同款登记）；口径隔离不做跨实现结论 |

## 9. 运行预算与产物树

```
outputs/10-24-mild-w500-ec-unbiased/
├── dryrun_mini/  工程干跑存证（8 源 × 4 网格，README 注明非正式 Run）
├── ec_main/      Run 001：subject_{0..7}.json（σ*/conn/sign/response/诊断/凭据）
│                        + subject_{0..7}_curves.npz（mse_curve + resp_ref）
├── summary/      Run 002：group_summary json（分布 + 方差分解 + 稳定性 + 重映射
│                        对照 + 曲线后置健康诊断）
└── gccorr/       Run 003：gccorr_summary json（|EC|/raw × GC aligned/flipped ×
                         Spearman/Pearson，逐例 + 组汇总）
```

GPU 侧仅 mini 干跑 + 主跑（合计预算 ≤ 35 min，护栏 60 min）；其余纯离线。

## 10. mild GC 参考注册设计（R6）

- **来源**：`/root/autodl-tmp/mild-depression-gc.zip`（2026-10-24 用户提供，15.2 MB，
  SHA256 随 meta 登记）→ staging 解包校验后注册。
- **目录与 schema（对齐房内 refs 契约）**：
  `refs/granger_bivariate_spectral/mild40-trial500-zscore-fs250-fixed8-f1-45-favg/`
  内 40 × `refs_s{sid}.npz`：`ec` float32 (128,128)（**ec[a,b]=a→b 行=源/列=靶，
  对角 0**，源 mat[i,j]=GC(i→j) 同向直拷）+ `subject_id` int64 + `group`（MDD/HC）+
  `params` 串（方法/阶数/频带/窗长/归一化/池化）；`meta.json`（来源 + 方法/参数 +
  文件清单 + 映射验证记录）+ `SHA256SUMS`（40 npz）+ `README-source.md`（zip 内
  README 原文归档）+ `register_refs.py` / `register.log`（注册脚本与日志，可重放）。
- **subject 映射（已验证）**：zip 按 EEG_ALL.mat 顺序编号 s0–s39（前 20 MDD 后 20
  HC）；EEG_ALL 第 i 片与 manifest 文件四锚点（m_1/m_20/h_1/h_20 ↔ sid 1/20/21/40）
  **bitwise 相等** ⇒ `sid = index + 1`；fold0 test 8 例 ↔ 索引 {0,4,6,9,20,24,26,29}。
- **注册校验**：40 npz 全载入断言 shape/dtype/对角 0/无负值（源 finalize 日志
  min=0 max=0.482 复核）+ sid↔group 与 manifest 一致（MDD 1–20 / HC 21–40）。
- **MANIFEST.md**：新增「mne-connectivity 谱 GC」小节条目（与 latent_flow 原库 GC
  两档口径隔离，不做跨实现/跨数据集对比交付）。
- **对照协议（Run 003）**：EC_pert（conn 与 1/(σ*+1e-4) 重映射的 |abs| 与 raw 两口径）
  × GC refs（aligned = 行=源 vs ec；flipped = vs ec.T）× Spearman/Pearson，offdiag
  16256 对，逐例 + 组汇总 + 组均值矩阵——协议逐字对齐 `gen_unbiased_gc_corr.py`。
