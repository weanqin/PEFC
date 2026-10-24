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


## Run 004 — A_eff 与逐被试 EC 读出可视化（2026-10-24，用户指令「可视化 A_eff 以及被试扰动归因 EC」）

### 目的与假设

- 读出型检验图（不落库代码，PNG 交付）：训练后 A_eff（γ_m·M(d)+γ·W，行=靶/列=源）×
  组均值 EC 对照 + 8 例逐被试 EC 矩阵网格。

### 配置

- 数据：Run 001 subject_{0..7}.json conn 矩阵 + summary/group_mean.npz；A_eff 自 w500 ckpt
  `flow.graph_mixer.effective_adjacency()` 参数层读取（CPU，无前向）
- 命令：内联 matplotlib（RdBu_r 双向色，diag 掩蔽，q99 饱和，E1–E128 刻度）

### 结果

- `outputs/10-24-mild-w500-ec-unbiased/viz/aeff_vs_mean_ec.png`（a: A_eff；b: 组均值 EC）+
  `viz/ec_subjects.png`（2×4：行 1 MDD s1,5,7,10 / 行 2 HC s21,25,27,30，共享色标）
- 读出数字：γ_m=1.0、γ=0.1；W offdiag −0.063±0.559、相对不对称 0.149（M 对称核主导）；
  **Spearman(|EC_mean|, |A_eff|) offdiag = −0.478**（结构先验与扰动 EC 中等负关联，
  描述性登记——与 EC 靶主效应主导 0.64 及 provisional caveat 联动解读）

### 结论

- A_eff 呈 mexhat 距离环带结构（对称核主导）；EC 逐被试有个体结构（跨被试 ρ=0.576）
  但主要由靶灵敏度差异塑形；两者负关联进一步支持「该配置 EC_pert 未编码结构先验意义上的
  有向传导」。派生决策项：无（读出登记）。

## Run 005 — conn 公式 v2 重映射 + GC 相关重算（2026-10-24，用户指令）

### 目的与假设

- 用户指定新连接强度公式 conn = −(ln σ\* + 0.5·ln 2πe) + 2/3；基于 Run 001 在档 σ\* 纯离线
  重算全部 conn（无新前向、不动模型与流程），同协议重算 EC×GC 相关并与原公式对照。

### 配置

- 数据：ec_main/subject_{0..7}.json σ\*（float64 重算；归档 conn 为 float32 域，精度效应
  max|Δρ_raw| = 2.1e-08 随档）；refs mild40（同 Run 003）；相关性协议同 Run 003。
- 数学事实：v2 相对原式为严格常数平移（+1.085605）⇒ raw 口径 Spearman/Pearson 逐位不变
  （同精度断言 8/8 PASS）；零点 σ\* 由 0.1592 移至 0.4713（贴近 σ_max=0.48）⇒ 仅 |·| 口径重排。

### 结果

- conn v2 分布：值域 [−0.0183（=conn(0.48)）, +3.8529（=conn(0.01)）]；offdiag 均值
  +0.58~+1.09；负值占比 31.8–39.3%（即 @σ_max 钳边平台，仅 −0.018 微负）。
- GC 相关：raw aligned ρ = +0.1178±0.1537（与原式逐位一致，断言在档）；**|EC| aligned
  ρ 由 −0.0986±0.0656 翻为 +0.1184±0.1529、r 由 −0.0692 翻为 +0.1674、组均值 ρ 由
  −0.2317 翻为 +0.1394**（逐例 6/8 转正；s25 仍 −0.153、s1 −0.001）。
- 产物：`recompute_v2/`（subject_{0..7}.json 重映射档 + gccorr_v2_summary.json 含不变性
  断言与新旧对照）。

### 结论

- |·| 口径的符号翻转是**零点移动的算术结果**（v2 下 66% 对为正 ⇒ abs≈raw ⇒ abs 读数
  ≈ raw +0.118），并非 EC×GC 关系的真实变化——真实对齐度即 raw 口径，两公式下同为
  +0.118 微弱正、flip≈direct 差距小，仍属「基本不相关」量级；旧 abs 的 −0.099 系
  σ\*=0.159 折叠伪影。判读口径提醒随档：跨公式比较一律用 raw 或 σ\* 序。

## Run 006 — conn 公式 v3 重映射 + GC 相关重算（2026-10-24，用户指令）

### 目的与假设

- 用户指定 v3 公式 conn = −(ln σ\* + ln 2πe) + 5/7；基于 Run 001 在档 σ\* 纯离线重算 +
  同协议 GC 相关（无新前向、不动模型与流程）。

### 配置

- 数学事实：v3 = 原式仅常数 1→5/7 ⇒ **严格常数平移 conn_v3 = conn_old − 2/7** ⇒ raw 口径
  Spearman/Pearson 逐位不变（同精度断言 8/8 PASS）；零点 σ\* = 0.1196（原 0.1592 / v2 0.4713）。

### 结果

- conn v3 分布：值域 [−1.3896（=conn(0.48)）, +2.4816（=conn(0.01)）]；offdiag 均值
  −0.28~−0.79；**负值占比 74.4–89.2%**（零点左移使绝大多数对落负区）。
- GC 相关（重点 Pearson）：**|EC| aligned Pearson = −0.1364±0.0809 [−0.2261,+0.0081]**
  （Spearman −0.1318±0.0816；flip Pearson −0.0804）；组均值矩阵 |EC| Pearson −0.2988；
  raw 口径 Pearson +0.1674±0.1661 / Spearman +0.1178±0.1537（与原式/v2 逐位一致）。
- 产物：`recompute_v3/`（subject_{0..7}.json + gccorr_v3_summary.json）。

### 结论

- 三公式对照：raw 恒 +0.118/+0.167（信息不变）；abs 口径随零点游走——原式 Pearson
  −0.069 / v2 +0.167 / v3 −0.136——均为折叠位置算术结果。v3 使 3/4 以上对为负，
  abs 幅度被折叠放大至强端，abs 相关比原式更负。判读纪律同 Run 005：跨公式一律
  raw 或 σ\* 序。

## Run 007 — conn 公式 v4 重映射 + GC 相关重算（2026-10-24，用户指令）

### 目的与假设

- 用户指定 v4 公式 conn = −(ln σ\* + 0.5·ln 2πe) + 5/7；基于 Run 001 在档 σ\* 纯离线重算 +
  同协议 GC 相关（无新前向、不动模型与流程）。

### 配置

- 数学事实：v4 = 原式严格常数平移（亦 = v2 + 1/21）⇒ raw 口径 Spearman/Pearson 逐位不变
  （同精度断言 8/8 PASS）；**零点 σ\* = e^(5/7−0.5·ln2πe) = 0.4943 落在网格上界 0.48 之外**
  ⇒ 全部 conn 严格为正（min>0 断言 8/8 PASS）⇒ **abs 口径 ≡ raw**（逐位一致断言 8/8 PASS）。

### 结果

- conn v4 分布：值域 [+0.0293（=conn(0.48)）, +3.9005（=conn(0.01)）]，**负值占比 0.0%**
  （8/8 例）；offdiag 均值 +0.63~+1.14。
- GC 相关（abs≡raw，Pearson 重点）：**aligned Pearson = +0.1674±0.1661 [−0.1546,+0.3471]**；
  aligned Spearman = +0.1178±0.1537；flipped Pearson +0.1080±0.0633；组均值矩阵
  Pearson +0.2155 / Spearman +0.1409。
- 产物：`recompute_v4/`（subject_{0..7}.json + gccorr_v4_summary.json）。

### 结论

- v4 是四个公式中唯一**全域非负**的标度（零点设在 σ_max 之外）⇒ 无 |·| 折叠歧义，abs 与
  raw 同一读数；携带的 EC–GC 信息与原式/v2/v3 完全相同（raw 不变断言），组均 Pearson
  +0.167 / 组均值矩阵 +0.215 的微弱正相关仍在「基本不相关」量级。后续若需网络化/阈值
  切边，v4 数值可直接使用。
