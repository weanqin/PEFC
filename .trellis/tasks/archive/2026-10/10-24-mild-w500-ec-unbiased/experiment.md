# 实验日志 — 10-24-mild-w500-ec-unbiased

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 000 — GC 参考注册（2026-10-24）

### 目的与假设

- 将用户提供的 mild GC 参考（mne-connectivity 谱 GC，VAR 阶 8 固定、1–45 Hz faverage）
  注册入 refs，供 Run 003 EC×GC 对照。

### 配置

- 代码：`refs/granger_bivariate_spectral/mild40-trial500-zscore-fs250-fixed8-f1-45-favg/register_refs.py`（注册脚本随档）
- 来源：`/root/autodl-tmp/mild-depression-gc.zip`（SHA256 随 meta.json）→ staging `/root/autodl-tmp/mild_gc_staging/`
- 映射：sid = EEG_ALL index + 1（四锚点 m_1/m_20/h_1/h_20 ↔ sid 1/20/21/40 bitwise 相等，规划期已验）
- 命令：`cd <refs 子目录> && OMP_NUM_THREADS=4 python3 register_refs.py | tee register.log`

### 结果

- 40 × `refs_s{sid}.npz`（ec float32 (128,128)，对角 0）+ SHA256SUMS + meta.json + README-source.md + register.log
- sanity：diag_max=0.0 min=0.0 max=0.4820 offdiag_mean=0.01454（与源 finalize 日志 min=0/max=0.482 一致）；sid↔group 与 manifest 全 40 一致
- refs/MANIFEST.md 追加「mne-connectivity 谱 GC」小节条目
- 产物：`refs/granger_bivariate_spectral/mild40-trial500-zscore-fs250-fixed8-f1-45-favg/`

### 结论

- 注册完成，口径与用户说明逐项一致（阶 8 固定 / 1–45 Hz / trial 500 点 / 逐 trial z-score / 30 trials 池化）。
- 派生决策项：Run 003 直接引用本档；与 edra 两档口径隔离不做跨数据集对比。

## Run 001 — 无偏噪声 global-τ EC 主跑 8 例 × 30 trial（2026-10-24）

### 目的与假设

- 在 mild-w500 fold0 代理脑上以无偏噪声（μ=0，σ·ξ）扰动 obs 段源通道，两阶段 global-τ
  （q=0.5 钉死）计算逐被试 128×128 有向 EC（σ*/conn）；假设 = 代理脑跨通道传导结构
  可被 σ* 编码（阴性亦如实登记，负证据链惯例）。

### 配置

- 代码：`models/latent-flow-v1/ec_mild_unbiased.py`（本任务唯一新增，git head 随 payload）
- 代理脑：`outputs/10-24-latent-flow-mild-w500/run_w500_fold0/checkpoints/fold0_mild_flow_best_graph_mexhat.pt`（stage2 best 0.8356@E150 顶格未收敛，caveat 随档）
- 参数：q=0.5（用户拍板钉死）、σ_ref=0.3、σ 网格 log[0.01,0.48]×24、seed 42、Euler 10 步、batch 32、bf16 autocast 采样域 + fp32 输入侧扰动、μ_bias 关闭
- 数据：manifest `splits/mild_5fold_seed42.json` fold0 test 8 例（sid 1,5,7,10,21,25,27,30）× 30 trial 全量（用户拍板 8×30）；归一化 = train 折前 500 点统计（SHA 对拍 d184e647… 通过即跑）
- 环境：conda npi @ 本机 4080S 32GB（.trellis/research/environment.md）
- 命令：`OMP_NUM_THREADS=8 /root/miniconda3/bin/python models/latent-flow-v1/ec_mild_unbiased.py --mode main --out-dir /root/npi/outputs/10-24-mild-w500-ec-unbiased/ec_main`

### 结果

- 8/8 例成功（n_source_failures=0），46.9 min 总耗时（5.85–5.86 min/例稳定），逐例：

| 被试 | 组 | τ(q=0.5) | interior% | @σ_min% | @σ_max% | P(resp>τ) |
|---|---|---|---|---|---|---|
| s1 | MDD | 1.660e-4 | 54.2 | 12.8 | 33.0 | 0.500 |
| s5 | MDD | 8.340e-5 | 64.1 | 1.6 | 34.3 | 0.500 |
| s7 | MDD | 7.913e-5 | 58.8 | 7.6 | 33.6 | 0.500 |
| s10 | MDD | 9.883e-5 | 64.9 | 1.7 | 33.4 | 0.500 |
| s21 | HC | 1.010e-4 | 65.4 | 1.6 | 33.0 | 0.500 |
| s25 | HC | 9.968e-5 | 42.3 | 19.1 | 38.6 | 0.500 |
| s27 | HC | 7.409e-5 | 66.5 | 2.6 | 31.0 | 0.500 |
| s30 | HC | 8.900e-5 | 65.9 | 1.0 | 33.1 | 0.500 |

- P(resp>τ)=0.500 逐例精确成立（τ=响应中位数 ⇒ q=0.5 构造性自洽校验）；对角 σ*=σ_min
  自响应最强（语义正确）。
- 产物：`outputs/10-24-mild-w500-ec-unbiased/ec_main/`（subject_{0..7}.json 全 07-04/unbiased
  schema + 凭据 + subject_{0..7}_curves.npz）；干跑存证 `dryrun_mini/`（8 源×4 网格，
  非正式口径）。

### 结论

- 口径健康度（q=0.5 钉死下如实登记）：interior 组均 60.3%±8.4%（双侧钳边 @min 6.0%/
  @max 33.7%——中位阈值下 ~1/3 对在 σ_max 内响应不达中位，属 q=0.5 位置性钳边而非
  动态范围耗尽，见 Run 002 P(resp>τ) 判读）；conn offdiag 均值 −0.36±0.19、std 0.91。
- 主读出：σ* 矩阵可计算、跨被试稳定 ρ=0.576，但响应曲线平（dyn_ratio 2.6×，Run 002）
  ⇒ σ* 判别力压缩；叠加代理脑 stage2 顶格未收敛 caveat ⇒ **EC 结果 provisional**。
- 下一步 / 派生决策项：GC 对照见 Run 003；q=0.5 外取值不测（用户拍板）；w500 续训与否
  的既有拍板项仍悬置（EC 结论随其联动）。

## Run 002 — 离线组级汇总与曲线后置诊断（2026-10-24）

### 目的与假设

- 无 GPU 离线重算组级诊断：分布/方差分解/跨被试稳定性/重映射不变性/响应曲线健康度
  （评审 A：P(resp>τ) 判读 τ 位置性钳边 vs 动态范围不足）。

### 配置

- 命令：`--mode summary --main-dir .../ec_main --out-dir .../summary`（读 Run 001 json+npz，float64 纯 CPU）

### 结果

- 组级：interior 60.3±8.4%；@min 6.0±6.7%；@max 33.7±2.2%；**dyn_ratio 中位 2.62±0.33×**
  （v5-main unbiased 参照 137.9×）；nonmono_pair_frac 0.992（平响应 + bf16 数值抖动下
  逐对曲线近平平台；walk 单调走查按设计容 plateau，σ* 判别受网格分辨率下限约束）；
  σ* offdiag 中位 0.3030±0.0008（≈σ_ref 构造性自洽：τ=响应中位 ⇒ 中位对在 σ_ref 附近跨界）。
- 方差分解（offdiag conn）：**行(源) 0.037 / 列(靶) 0.641 / 残差(边特异) 0.322**——
  global-τ 修正使源主效应消除（07-04 教训口径），但**靶主效应主导**：σ* 主要编码
  per-target 灵敏度而非 a→b 边特异传导。
- 跨被试稳定性：offdiag Spearman 均值 0.576 [0.173, 0.827]。
- 重映射 1/(σ*+1e-4) Spearman 不变断言 8/8 PASS。
- 产物：`summary/group_summary.json` + `group_mean.npz`（conn/σ* 组均值矩阵）。

### 结论

- dyn_ratio 2.6× + 列主效应 0.64 ⇒ 该代理脑上 σ* 的边特异信号弱（残差 0.32），
  与 w500 未收敛 caveat 一致；不构成 q 取值回改依据（用户口径），如实登记。
- 派生决策项：若后续 w500 续训/波形判读通过，可同命令重跑复核。

## Run 003 — EC × mild GC 参考相关对照（2026-10-24）

### 目的与假设

- 描述性并置：|EC_pert| 与 raw conn × GC refs（mild40 档，aligned=行=源 / flipped=vs
  ec.T）× Spearman/Pearson，offdiag 16256 对，协议逐字对齐 gen_unbiased_gc_corr.py
  先例；不构成代理脑资格判定。

### 配置

- refs：`refs/granger_bivariate_spectral/mild40-trial500-zscore-fs250-fixed8-f1-45-favg/`
  （Run 000 注册；sid↔subject_file 逐例断言对齐）
- 命令：`--mode gccorr --main-dir .../ec_main --out-dir .../gccorr`

### 结果

- 逐例（|EC| aligned entropy）：s1 +0.031 / s5 −0.192 / s7 −0.080 / s10 −0.159 /
  s21 −0.089 / s25 −0.100 / s27 −0.116 / s30 −0.085（7/8 负）。
- 组汇总：**|EC| aligned ρ = −0.0986±0.0656 / r = −0.0692±0.0611；raw aligned ρ = +0.1178；
  flipped ≈ aligned（|EC| ρ −0.044）⇒ 无方向信息；组均值矩阵 |EC| ρ = −0.2317**。
- inv 重映射 1/(σ*+1e-4)：raw Spearman 与 entropy raw 逐位一致（同族单调，断言校验）；
  abs 口径因负值折叠序变（部分例转正 +0.29）——两式并置登记，解读以 |ρ|≤0.12 量级为准。
- 产物：`gccorr/gccorr_summary.json`。

### 结论

- **EC_pert × GC 基本不相关**（|ρ| 组均 ≤0.12，与 v5-main unbiased 先例 ρ≈−0.04 同量
  级；raw 微正 +0.118 与 flipped 差距小不构成方向证据）——按边界登记为描述性阴性，
  与该代理脑 provisional 状态一致。
- 派生决策项：无（用户边界内）；若未来代理脑资格任务立项，本档数字可作基线引用。

