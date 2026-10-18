---
date: 2026-08-23
name: task-snr-scan
aliases:
  - band x horizon scan
  - 实验五
  - task-level SNR scan
task: 08-23-task-snr-scan
type: diagnosis (task-level SNR scan: band x horizon)
dataset: modma
fold: 0
status: "完成 (verdict BR-BAND: 12/12 带对比过线; bb horizon 同时过线被层级吸收)"
tags:
  - experiment
  - diagnosis
  - cross-channel
  - probe-matrix
  - modma
  - raw-domain
  - band-filter
  - horizon
  - result-positive-with-caveats
checkpoint: null   # 本实验不加载任何模型权重；冻结输入（raw pairs npz / FC_emp / 08-16 归档读数 / geometry）只读 sha1 戳于正文与 verdict.json
variant: null
related_spec: "[[feasibility-gate]]"
---

# 任务级 SNR 扫描：band × horizon（实验五）

> Task `08-23-task-snr-scan`。DySIB 线已按预注册终止归档
> ([[08-19-dysib-01-v22]] ZERO_FACE；08-21 addendum BR-B：EEG 窗口级预测状态
> **无低维充分统计量**，联合低维表征线关闭）。本实验是当时唯一存活的数据级
> 实验：把 [[08-16-raw-domain-probe]] 的 raw 域跨通道探针**原封不动**地铺到
> 5 频带 × 3 horizon 的 15 格网格上（纯 CPU、无训练、无扰动、`models/` 零
> 新增 diff），用预注册分支表裁决最后一章写「任务重定向」（BR-BAND /
> BR-HORIZON：substrate 或 horizon 层面还有可修的任务 SNR）还是「范式级
> 上限」（BR-FLAT：窗口级宽带转迁信息本质边际）。判读留给讨论。

> [!summary] 一句话结论
> 预注册判决 **BR-BAND**——12/12 个同 horizon 带对比全部过线（k8 CI_low>0
> ∧ 点增益 ≥+0.03；delta/theta/alpha 的 k8 增益 **+0.58～+0.68**、beta
> +0.04～+0.26），bb 自身的 horizon 放大同真（fc64 +0.0513 / fc32 +0.1209，
> BR-HORIZON 触发条件也满足但按冻结层级被 BR-BAND 吸收）；AC 硬门逐位复现
> （4 量 max|diff| = 0，地板软校验同为 0 偏差）。**最重要的诚实警示**：增益
> 的机械成因主要是「带限让目标信号本身变得可预测」——带格的自通道对角锚
> I[a,a] 从 bb 的 +0.075 跃到 **0.82～0.99**（近乎完全的自外推），而 k8 近箱
> 读数与对角锚的比值在所有带内恒定于 0.74～0.80（bb 也是 0.80）——近箱携带
> 的更像「自通道可读信息的固定比例」（窄带信号的空间平滑性），而非新增的
> 跨通道耦合；delta 格全程带边缘效应警示（1 Hz 周期 ≈ 整窗 1.024 s）且带外
> 泄漏 36%。按冻结分支表：substrate 路线 = **逐通道 band-split VAE**（联合
> 低维线维持 BR-B 关闭），以本探针为资格门重建 v1 式链路。

## 实验目的与假设

08-16 判 H-VAE（raw 清了地板但点估计低于 latent）后，剩余的解释空间在
任务设定层：跨通道转迁信息也许存在，但**弥散在宽带窗口里**，被 128 样点
horizon 的 SNR 稀释。两个旋钮可拆：**频带**（信息可能集中在 delta/theta/
alpha/beta 某带，宽带聚合把它平均掉）与 **horizon**（fc 段越短，近期动力
学 SNR 越高）。若任一旋钮把 k8 近箱的配对增益拉过 CI_low>0 ∧ 点 ≥+0.03 的
双线，任务值得重定向（band-split 或短时程重启）；若全部平坦（BR-FLAT），
则窗口级宽带转迁信息本质边际且高维弥散，论文以「范式级上限」收章。

## 预注册块（frozen before the registered run）

### 判决规则（`task_snr.decide_task_snr`，pure function，单测钉死）

- **存在性**（within 主判，cross 次要，逐格）：k8 CI_low > 自身地板 + 0.02。
- **放大性**（主判，配对差、同重采样、k8 判决箱、off-diag 并报）：
  - 带效应：Δ(band − bb 同 horizon)，4 带 × 3 horizon = 12 个对比；
  - horizon 效应：Δ(fc64−fc128)、Δ(fc32−fc128)，bb 内与各带内分别报告；
  - 联合读数：Δ(band×fc32 − bb×fc128)，**描述性，不进分支**；
  - 过线 = k8 CI_low > 0 **且** 点增益 ≥ +0.03（双线同时）。
- **分支判定**（层级互斥）：BR-BAND（任一带对比过线）→ BR-HORIZON
  （仅 bb 的 fc64/fc32 过线）→ BR-FLAT。
- 近阈值纪律 ±0.005（存在性 CI_low 对 floor+0.02、放大性 CI_low 对 0）；
  点估计-vs-margin 只进手工警示段。

### 分支字典（`TASK_SNR_BRANCH_DICTIONARY`，逐字冻结）

- **BR-BAND**：任一 (band, horizon) 的 k8 配对增益 CI_low>0 且点增益 ≥+0.03
  → 频带集中坐实 → substrate 路线 = 逐通道 band-split VAE（联合低维线已由
  BR-B 关闭），以本探针为资格门重建 v1 式链路；
- **BR-HORIZON**：仅 horizon 放大（bb 的 fc64/fc32 过线）→ 缩短预测 horizon
  重建 pairs，FM/DE 在短时程重启（EC 语义随 horizon 声明）；
- **BR-FLAT**：全部放大性判据不过 → 窗口级宽带转迁信息本质边际且高维弥散
  （"范式级上限"章节定稿）；记录相位类读出（PLV/相位传递熵）为唯一
  未测读出族、数据集/范式变更为唯一上游选项；
- 伴随判据（不进分支）：raw I 模式对 FC_emp 的相关随 band/horizon 的变化
  ——"这点信息像不像耦合"的描述性读数。

### 阈值与常量（frozen）

| 量 | 值 |
|---|---|
| 存在性 margin（k8 CI_low 超自身地板） | $0.02$（= 08-15 `BRANCH_MARGIN`） |
| 放大性实用线（点增益） | $0.03$（正式线 CI_low>0 之外的双线） |
| near_tol（形式 flag 容差） | $0.005$ |
| 近箱 k（自通道排除） | $8$（08-13 Teacher-L 约定） |
| 距离四分位箱数 | $4$ |
| AC 复现容差（bb×fc128 vs 08-16 归档，4 量三分量） | $1\times10^{-4}$ |
| 滤波 | Butterworth order 4 带通 `sosfiltfilt`（零相位、默认 odd padding） |
| 带定义 | bb=恒等；delta 1–4 / theta 4–8 / alpha 8–13 / beta 13–30 Hz |
| horizon | fc128 / fc64 / fc32（fc 段取滤波后前 64/32 样点，L_persist 按目标长度重算） |
| 采样率 | `get_dataset("modma").fs` = 250 Hz（数据集注册表读取，禁硬编码） |

### 种子（frozen）

| 种子 | 值 |
|---|---|
| probe 主种子（splits / boot 母 / shuffle 母） | $0$ |
| boot（`default_rng(seed+1)`，15 格同 seed → 同重采样 = 配对前提） | $1$ |
| shuffle 母（`default_rng(seed+2)`，floor 种子推导镜像 08-16） | $2$ |
| mlp_base（仅 `--mlp` 时；注册运行默认关） | $7$ |
| floor_within / floor_cross | 1798679647 / 561807779（与 08-16 逐位同推导） |

### 规模与协议（frozen）

- 数据 = `results/de/probe_raw/fold0_raw_pairs.npz` **只读**（N=8067
  train+test、C=128、obs/fc 各 128 样点；08-16 已做对齐硬校验）。
- 逐格滤波：逐通道 obs+fc 拼回 256 样点整窗 → `sosfiltfilt`（float64，
  chunk 512 窗，chunk 无关性有单测）→ 重切 obs（前 128）/ fc（fc 段前
  fc_len 样点）；bb 格恒等直通（slice，逐位等于归档数组）。
- 能量健全性（冻结定义）：256 样点窗 rfft 功率谱，带内保留率
  retention_in = ΣP_post[带内]/ΣP_pre[带内]，另记 leak_out（带外占比）、
  band_frac_pre（原信号带内占比）、energy_ratio（总能量比，附加描述量）；
  带下限周期 > 半窗（0.512 s）自动打 edge-warning。
- 探针 = `probe_matrix.run_probe` ridge-only（`DEFAULT_ALPHAS`、
  `build_splits(seed=0)`、n_boot=1000、n_shuffle=0），源 = 8 个 AP 轴审计
  通道 [70,61,56,41,38,37,26,9]，目标 = 全 128；地板 = 每格×每 split
  `shuffle_floor(..., return_rows=True, n_shuffle=5)` + `floor_from_rows`
  逐箱（off-diag/k8/q0–q3）；读出 = off-diag/k8 [CI+地板]、四分位剖面、
  对角锚 I[a,a] [CI]、Spearman(电极距离, R²_off) [CI]、伴随
  Spearman(I_off, FC_emp) [CI]。
- **AC 硬门**：bb×fc128 必须复现 08-16 归档（within off-diag +0.0261
  [−0.0024,+0.0573]、k8 +0.0605 [+0.0204,+0.1003]；cross off-diag +0.0169
  [+0.0007,+0.0250]、k8 +0.0563 [+0.0258,+0.0785]），容差 1e-4；超差即停、
  零产物。分箱地板软校验（WARN 不中止）。门在首格完成后、其余 14 格之前
  评估（坏锚不烧 13 h 算力）。
- $R^2$ 约定：$1-\mathrm{MSE}/L_{\mathrm{persist}}$（T-002D），L_persist 逐格
  按目标长度重算；MLP 对照降为 `--mlp` 可选（默认关，线性主导由 08-15/16
  确立）。

## 设计与执行

- **库**：`analysis/fm_gate/task_snr.py`（冻结常量/分支字典/滤波/能量/
  网格/AC 门/判决全为新增）；**runner**：`scripts/run_task_snr_scan.py`；
  核心机件（`run_probe`/`build_splits`/`shuffle_floor`/`persistence`/
  `paired_delta`/距离箱）全部复用 import，零复制粘贴。
- **唯一机件改动**：`probe_matrix.run_probe` 维度泛化——冻结实现假设 obs/fc
  逐通道同维（prediction reshape 用 obs 的 d），horizon 截断格（obs=128、
  fc=64/32）会崩。改为 reshape 用 `d_fc = z_fc.shape[2]` + (N,C) 形状守卫；
  对称调用（全部既有注册路径）数值逐位不变，由全量 pytest 零回归 + AC 硬门
  逐位复现双重锚定。与 08-16 给 `shuffle_floor` 加 `return_rows` 同类的
  最小 additive 能力扩展。
- **测试**：`tests/test_task_snr.py` 13 项（bb 直通逐位、滤波-重切 + chunk
  不变性、fc 截断 L_persist 重算、配对差同重采样（spy
  `cluster_bootstrap_indices` 两次 run_probe 索引逐位相同 + 手工差复算）、
  分支全分支+优先级+联合不触发、near flag、能量保留（带内正弦/带外抑制）、
  sos 缓存、AC 门（容差内/外/缺键即抛）、子集保块序、edge-warning、网格
  bb 优先）。全量 pytest **284 passed, 3 skipped**（基线 271 + 新增 13，
  零回归）。
- **执行序列**：PRD/预注册冻结 → 冒烟（alpha×fc64×5 被试，零产物，数字
  不入笔记）→ 注册运行 15 格全量（CPU 后台，stdout 落盘）→ 产物三件 +
  本笔记。
- **算力**：实际 **48,217 s ≈ 13.4 h CPU**（远超任务简报 3–5 h 预估，如实
  记录）：fc128 格 ≈ 5.2–5.6 ks + 地板 ~90 s；fc64 ≈ 2.6–2.7 ks + 45 s；
  fc32 ≈ 1.4 ks + 23 s；滤波 11–12 s/格；产物落盘 ~10 s。AC 门在启动后
  ~1.6 h 处通过，避免了坏锚情形下 13 h 空转。

## 结果

### 判决：BR-BAND（12/12 带对比过线；AC 硬门逐位复现）

| 判据组 | 结果 |
|---|---|
| 带对比过线（12 个中） | **12 全部**（delta/theta/alpha 各 horizon k8 点增益 +0.58～+0.68、CI_low ≥ +0.52；beta +0.0415～+0.2617、CI_low ≥ +0.0032） |
| bb horizon 过线 | fc64 +0.0513 [+0.0350,+0.0638]、fc32 +0.1209 [+0.0926,+0.1454] —— **同真**（BR-HORIZON 条件满足，层级上被 BR-BAND 吸收） |
| near flag | `band:beta_fc128:ci_low_vs_zero`（CI_low +0.0032，距 0 在 ±0.005 内，如实报原始值；点 +0.0415 已过实用线，不属点-vs-margin 张力） |
| 点-vs-margin 手工警示 | 无条目（所有判决量双线同侧） |
| AC 硬门 | 4 量三分量 max\|diff\| = **0.000e+00**（逐位复现）；地板软校验 4 项同为 0 偏差（无 WARN） |
| 存在性（15 格 × 2 split） | **30/30 全过**（含 bb 三格与 beta_fc128：k8 +0.1007 > −0.0136+0.02） |

### 头表：15 格 within（判决量 + 地板 + 对角锚 + 能量）

| band | horizon | off-diag R² [CI] (floor) | k8 near [CI] (floor) | diag I[a,a] [CI] | exist | ret_in |
|---|---|---|---|---|---|---|
| bb | fc128 | +0.0261 [−0.0024, +0.0573] (−0.0135) | +0.0605 [+0.0204, +0.1003] (−0.0151) | +0.0753 [+0.0296, +0.1187] | T | 1.000 |
| bb | fc64 | +0.0509 [+0.0133, +0.0908] (−0.0140) | +0.1127 [+0.0576, +0.1615] (−0.0158) | +0.1393 [+0.0759, +0.1926] | T | 1.000 |
| bb | fc32 | +0.0800 [+0.0363, +0.1207] (−0.0135) | +0.1829 [+0.1167, +0.2351] (−0.0157) | +0.2269 [+0.1507, +0.2833] | T | 1.000 |
| delta | fc128 | +0.2417 [+0.2271, +0.2548] (−0.0032) | **+0.7390** [+0.7197, +0.7567] (−0.0030) | +0.9651 [+0.9610, +0.9691] | T | 0.859* |
| delta | fc64 | +0.2472 [+0.2312, +0.2609] (−0.0033) | +0.7550 [+0.7351, +0.7725] (−0.0033) | +0.9876 [+0.9861, +0.9889] | T | 0.859* |
| delta | fc32 | +0.2481 [+0.2326, +0.2624] (−0.0034) | +0.7586 [+0.7385, +0.7764] (−0.0034) | +0.9947 [+0.9940, +0.9953] | T | 0.859* |
| theta | fc128 | +0.2147 [+0.2020, +0.2274] (−0.0044) | +0.6742 [+0.6522, +0.6944] (−0.0053) | +0.8649 [+0.8511, +0.8804] | T | 0.712 |
| theta | fc64 | +0.2392 [+0.2253, +0.2518] (−0.0042) | +0.7469 [+0.7277, +0.7642] (−0.0051) | +0.9567 [+0.9516, +0.9621] | T | 0.712 |
| theta | fc32 | +0.2509 [+0.2367, +0.2636] (−0.0039) | +0.7775 [+0.7582, +0.7939] (−0.0050) | +0.9954 [+0.9949, +0.9959] | T | 0.712 |
| alpha | fc128 | +0.3007 [+0.2322, +0.3517] (−0.0076) | +0.6536 [+0.6110, +0.6753] (−0.0091) | +0.8160 [+0.7765, +0.8382] | T | 0.860 |
| alpha | fc64 | +0.3445 [+0.2798, +0.3916] (−0.0078) | +0.7588 [+0.7347, +0.7751] (−0.0093) | +0.9512 [+0.9397, +0.9575] | T | 0.860 |
| alpha | fc32 | +0.3576 [+0.2957, +0.4048] (−0.0073) | +0.7946 [+0.7753, +0.8110] (−0.0085) | +0.9969 [+0.9962, +0.9972] | T | 0.860 |
| beta | fc128 | +0.0237 [+0.0205, +0.0264] (−0.0123) | +0.1007 [+0.0948, +0.1054] (−0.0136) | +0.1364 [+0.1294, +0.1430] | T | 0.836 |
| beta | fc64 | +0.0619 [+0.0561, +0.0667] (−0.0127) | +0.2200 [+0.2111, +0.2272] (−0.0142) | +0.2922 [+0.2833, +0.3001] | T | 0.836 |
| beta | fc32 | +0.1333 [+0.1233, +0.1417] (−0.0127) | +0.4420 [+0.4277, +0.4540] (−0.0146) | +0.5809 [+0.5699, +0.5910] | T | 0.836 |

（* = delta 格 edge-warning：1 Hz 周期 1.0 s > 半窗 0.512 s。cross 次要
全部同构：30/30 存在性过，带格 cross k8 +0.10～+0.75，见 verdict.json；
bb 三格 cross 即 08-16 归档复现值。）

### 配对差 Δ(band − bb 同 horizon)，k8 近箱（主判）与 off-diag

| 对比 | within k8 [CI] 过线 | within off-diag [CI] | cross k8 [CI] |
|---|---|---|---|
| delta − bb @fc128 | **+0.6795** [+0.6296, +0.7224] T | +0.2166 [+0.1869, +0.2440] | +0.6249 [+0.5423, +0.7069] |
| delta − bb @fc64 | +0.6442 [+0.5866, +0.6983] T | +0.1978 [+0.1608, +0.2316] | +0.5803 [+0.4828, +0.6822] |
| delta − bb @fc32 | +0.5784 [+0.5206, +0.6373] T | +0.1698 [+0.1316, +0.2070] | +0.5049 [+0.3946, +0.6205] |
| theta − bb @fc128 | +0.6146 [+0.5605, +0.6639] T | +0.1894 [+0.1579, +0.2181] | +0.5925 [+0.5344, +0.6472] |
| theta − bb @fc64 | +0.6359 [+0.5751, +0.6934] T | +0.1895 [+0.1523, +0.2242] | +0.5953 [+0.5270, +0.6655] |
| theta − bb @fc32 | +0.5969 [+0.5377, +0.6559] T | +0.1724 [+0.1345, +0.2079] | +0.5457 [+0.4762, +0.6232] |
| alpha − bb @fc128 | +0.5911 [+0.5598, +0.6255] T | **+0.2703** [+0.2319, +0.2969] | +0.5631 [+0.5285, +0.5953] |
| alpha − bb @fc64 | +0.6465 [+0.5924, +0.6933] T | +0.2901 [+0.2621, +0.3108] | +0.6061 [+0.5645, +0.6618] |
| alpha − bb @fc32 | +0.6132 [+0.5541, +0.6708] T | +0.2746 [+0.2508, +0.2968] | +0.5661 [+0.5154, +0.6381] |
| beta − bb @fc128 | +0.0415 [+0.0032, +0.0777] T† | −0.0015 [−0.0317, +0.0241] | +0.0431 [+0.0219, +0.0737] |
| beta − bb @fc64 | +0.1094 [+0.0622, +0.1601] T | +0.0124 [−0.0252, +0.0463] | +0.0985 [+0.0628, +0.1493] |
| beta − bb @fc32 | +0.2617 [+0.2099, +0.3192] T | +0.0548 [+0.0170, +0.0918] | +0.2316 [+0.1841, +0.2945] |

（† = near flag 格：CI_low +0.0032 距 0 在 ±0.005 内。beta 的 off-diag
增益在 fc128/fc64 含 0——beta 的过线完全由近箱支撑。）

### horizon 效应 Δ(fc64−fc128) / Δ(fc32−fc128)，k8 近箱

| band | fc64−fc128 within [CI] | fc32−fc128 within [CI] |
|---|---|---|
| bb | +0.0513 [+0.0350, +0.0638] **T** | +0.1209 [+0.0926, +0.1454] **T** |
| delta | +0.0160 [+0.0127, +0.0194] F（CI 过、点 <0.03） | +0.0198 [+0.0148, +0.0248] F（同） |
| theta | +0.0726 [+0.0637, +0.0805] T | +0.1031 [+0.0884, +0.1158] T |
| alpha | +0.1067 [+0.0890, +0.1289] T | +0.1430 [+0.1192, +0.1738] T |
| beta | +0.1192 [+0.1145, +0.1244] T | +0.3410 [+0.3303, +0.3517] T |

联合读数 Δ(band×fc32 − bb×fc128)，k8 within：delta +0.6993、theta
**+0.7177**、alpha **+0.7341** [+0.6844, +0.7772]、beta +0.3825（cross
同构，全部 CI_low > +0.33；描述性，不进分支）。

### 伴随判据（描述性）：Spearman(I_off, FC_emp) 与局部性

- 伴随 FC_emp 相关全部弱：within |ρ| ≤ 0.033（多数 CI 含 0）、cross
  ρ ∈ [+0.026, +0.066]（多数 CI 不含 0 但量级微小）；带格略高于 bb 格
  （如 alpha_fc64 cross +0.0651 [+0.0395, +0.0985] vs bb_fc64 +0.0260）——
  "像耦合"的证据至多是方向性的。
- 局部性出现弱梯度（与 08-16 的 bb 含 0 不同）：带格 Spearman(距离, R²)
  within −0.031～−0.064（CI 均不含 0），bb 格仍含 0——带限后远距轻微衰减。
- 四分位 U 形放大：delta_fc128 q0 +0.437 / q1 +0.075 / q2 +0.116 /
  q3 +0.340（bb_fc128 为 +0.039/+0.015/+0.014/+0.037）——近箱与最远箱
  同抬的结构在带格中放大 ~10×，不是单调距离衰减。

### 能量健全性（滤波警示的量化）

| band | retention_in | leak_out | band_frac_pre | energy_ratio | edge |
|---|---|---|---|---|---|
| delta 1–4 Hz | 0.859 | **0.361** | 0.099 | 0.133 | **T** |
| theta 4–8 Hz | 0.712 | 0.186 | 0.101 | 0.088 | F |
| alpha 8–13 Hz | 0.860 | 0.038 | 0.408 | 0.364 | F |
| beta 13–30 Hz | 0.836 | 0.074 | 0.131 | 0.118 | F |

（delta：带外泄漏 36%、原信号带内能量占比仅 9.9%、1 Hz 周期≈整窗——
该带读数受边缘效应污染最重。alpha：原信号能量的 41% 集中在 8–13 Hz，
是四带中信息占比最高者。）

## 分析与结论（事实性读法；判读留给讨论）

1. **判决函数按冻结规则输出 BR-BAND，且触发面是"全表面"而非边缘**：12/12
   带对比过线、30/30 存在性过、cross 与 within 同构——不存在"只有某带某
   horizon 边缘过线"的脆弱结构。唯一 near flag（beta_fc128）不影响双线
   判定（点 +0.0415 与 CI_low +0.0032 同侧过线）。
2. **但增益的主体是"可预测性转移"，不必然是"信息增量"**：带格对角锚
   I[a,a] 0.82–0.99（bb 0.075）说明带限信号近乎完全自外推；k8/对角比
   恒定 0.74–0.80（bb 0.80）说明近箱读数跟随自通道锚等比例放大。若把
   k8 读数按各自对角锚归一，带间差异大幅收窄（delta 0.739/0.965=0.77、
   theta 0.674/0.865=0.78、alpha 0.654/0.816=0.80、beta_fc128
   0.101/0.136=0.74、bb 0.0605/0.0753=0.80）。这是一个**描述性读数**
   （对角锚按预注册逐格报告），其判读——"带限创造的是可预测的窄带载体，
   跨通道结构比例未变"——不属于预注册判决量，留给讨论。
3. **beta 与 horizon 的交互是网格中最"任务级"的信号**：beta 的带增益随
   horizon 缩短单调放大（+0.0415 → +0.1094 → +0.2617），其 off-diag 增益
   在 fc128 含 0、fc32 才显著（+0.0548 [+0.0170, +0.0918]）——高频转迁
   信息存在于短时程；delta 相反（horizon 缩短不增益：Δfc 均不过实用线）。
   bb 自身 horizon 增益过线（+0.05/+0.12）说明 horizon 效应不依赖带限。
4. **对终止语境的含义**（按冻结分支字典）：substrate 路线 = 逐通道
   band-split VAE，以本探针为资格门重建 v1 式链路；联合低维线维持 BR-B
   关闭。BR-HORIZON 路线（短时程重启 pairs）条件同真但层级居次——两个
   重定向方向并非互斥，排序属判读。
5. **AC 逐位复现（4 量 + 4 地板全部 0 偏差）** 把本实验钉死在 08-16 的
   冻结管线上：同 seed、同重采样、同地板种子，配对差的可信度由锚定保证。

## 诚实警示（Honest caveats）

- **(a) 带限的可预测性混淆（最重要）**：滤波同时改变了预测目标与预测源，
  带格 R² 的跃升大部分来自目标本身变可预测（对角锚 0.82–0.99）。本实验
  的判决量（同 horizon 带对比的配对差）在预注册时即为此设计——它度量
  "任务重定向到该带后探针能读到多少"，不是"该带比宽带多携带多少独立
  跨通道信息"。按对角锚归一的读数（分析 2）不支持"信息比例增加"。
- **(b) delta 格边缘效应**：1 Hz 周期 1.0 s > 半窗 0.512 s（自动
  edge-warning），带外泄漏 36%、带内能量占比仅 9.9%——delta 三个格的
  绝对值最不可信；但其增益 (+0.58) 远超其余带受污染可解释的量级，
  方向读数仍有效。
- **(c) 窄带 bin 粗糙**：256 样点 @ 250 Hz 的 rfft bin 间距 0.977 Hz，
  delta 带内仅 ~3 个 bin——带边界由滤波器滚降而非 bin 网格定义，能量
  保留/泄漏的数值依赖该粗糙谱窗。
- **(d) beta_fc128 的近阈值 flag**：k8 CI_low +0.0032 距 0 仅 0.0032
  （±0.005 带内），且 beta@fc128 的 off-diag 增益含 0——beta 在参照
  horizon 上的过线是 12 个对比中最弱的一个；beta 的稳健过线在 fc32。
- **(e) k8 近箱的空间解释**：k8/对角比恒定 + U 形四分位（q0/q3 同抬）
  与"距离衰减的局部转迁"不相容；近箱读数更可能反映窄带信号的空间平滑
  （容积传导级的近邻共变）。
- **(f) 伴随判据弱**：Spearman(I_off, FC_emp) 全格 |ρ| ≤ 0.066——带限
  后 I 模式与经验 FC 的相关仅方向性升高，"像耦合"证据薄弱。
- **(g) 覆盖范围**：8 个 AP 轴代表源 × 128 目标、fold 0、恒定 obs 窗
  128 样点、Butterworth order 4、一次滤波实现（sosfiltfilt 默认 padding）；
  结论在此覆盖内成立。MLP 对照未开（线性主导已由 08-15/16 确立；若开，
  `--mlp` 逐格可选）。
- **(h) 算力**：48,217 s ≈ 13.4 h 纯 CPU（超任务简报 3–5 h 预估约 2.7×，
  逐格墙钟在 verdict.json `meta.cell_seconds`）；无 GPU、无训练。

## 可复现性

环境 `D:\program\anaconda\envs\npi\python.exe`，从 `D:\102-npi`（仓库父
目录）运行——直接调 env python，禁 `conda run`（Windows gbk 崩溃），打印
全 ASCII：

```bash
# 冒烟（1 带 × 1 horizon × 5 被试，零产物）
D:/program/anaconda/envs/npi/python.exe -u -m latent_flow.scripts.run_task_snr_scan --fold 0 --smoke

# 注册运行（15 格全量；实测 48,217 s ≈ 13.4 h CPU；写 results/de/task_snr_scan/）
D:/program/anaconda/envs/npi/python.exe -u -m latent_flow.scripts.run_task_snr_scan --fold 0

# 本任务测试（13 项）与全量（284 passed, 3 skipped = 基线 271 + 13）
D:/program/anaconda/envs/npi/python.exe -m pytest latent_flow/tests/test_task_snr.py -q
D:/program/anaconda/envs/npi/python.exe -m pytest latent_flow/tests/ -q
```

- **Provenance**（全量戳在 `verdict.json` `meta`）：raw pairs sha1
  `a035e14c7923`（只读）；FC_emp `864777b6d6bc`；08-16 归档读数
  `d0be28f423e0`；geometry `476c21310727`；git rev `9efe1e3`；fs=250 Hz
  （注册表读取）；种子全表与阈值见预注册块（与 `meta.seeds` /
  `meta.thresholds` 一致）；AC 对照逐分量入 `ac_gate`。
- **新增代码全部 additive**：`analysis/fm_gate/task_snr.py`、
  `scripts/run_task_snr_scan.py`、`tests/test_task_snr.py`（13 测试）；
  `probe_matrix.py` 仅维度泛化（对称路径逐位不变，AC 门 0 偏差锚定）；
  本任务对 `models/` 零新增 diff（工作区 `models/dysib.py` 的脏状态属
  08-17/19/21 未提交线遗留）；既有注册产物零改动；未 commit。

## 下一步（候选方向，非承诺）

- **判读讨论（最高优先）**：(a) 分析 2 的归一化读数（k8/对角比恒定）是否
  应升格为下一次预注册的判决量——"带限增加的是可预测载体还是跨通道信息
  比例"；BR-BAND 的行动含义（band-split VAE 资格门）是否受此修正。
- **BR-BAND 行动路径**：逐通道 band-split VAE（alpha 优先：带内能量占比
  41% 最高、off-diag 增益最大 +0.27～+0.29），以本探针逐格复检为资格门，
  重建 v1 式 DE→FM 链路。
- **BR-HORIZON 备选**（条件同真、层级居次）：短时程 pairs 重建与 FM/DE
  重启的排序 vs band-split 的排序——beta×fc32（+0.26/+0.34 增益）是两
  旋钮的交点。
- 相位类读出（PLV / 相位传递熵）仍为唯一未测读出族（BR-FLAT 语境的遗留
  项，BR-BAND 下变为 band-split 设计的对照面）。

## 产出物索引

| 类型 | 路径 |
|---|---|
| 判决（branch + 12+2+联合配对差 + 15 格全读数/地板/存在性/能量 + AC 门 + provenance） | `results/de/task_snr_scan/verdict.json` |
| 摘要（头表、带/horizon/联合对比、伴随判据、判决、警示） | `results/de/task_snr_scan/summary.md` |
| 原始数组（逐格 R²/boot/地板 rows/α/L_persist/读出/delta/masks/energy/meta） | `results/de/task_snr_scan/task_snr.npz` |
| 运行日志（stdout 捕获，含 AC 门逐分量行、逐格墙钟、判决行） | `results/de/task_snr_scan/run_registered.log` |
| 库模块 | `analysis/fm_gate/task_snr.py` |
| 运行器 | `scripts/run_task_snr_scan.py` |
| 测试（13 项） | `tests/test_task_snr.py` |
| 冻结输入（只读） | `results/de/probe_raw/fold0_raw_pairs.npz`、`results/de/probe_matrix/fc_emp.npz`、`results/de/probe_raw/probe_raw_r2.npz`、`data/modma/geometry.pt` |

## 关联笔记

- 协议母本与锚定来源：[[08-16-raw-domain-probe]]（本实验 AC 硬门逐位复现
  其归档读数与地板；探针机件原封复用）
- 终止语境：[[08-19-dysib-01-v22]]（ZERO_FACE）及其 08-21 addendum BR-B
  （无低维充分统计量 → 联合低维线关闭，分支后果按此书写）
- 审计链上游：[[08-15-substrate-cross-channel-probe]]（BRANCH 3 与探针
  机件）、[[08-14-frozen-fm-layerwise-perturbation-audit]]（8 个 AP 轴
  审计源）
- Spec：[[feasibility-gate]]（gate 纪律与预注册判据）
