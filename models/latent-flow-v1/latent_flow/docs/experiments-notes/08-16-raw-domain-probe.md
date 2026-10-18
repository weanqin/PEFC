---
date: 2026-08-16
name: raw-domain-probe
aliases:
  - raw probe
  - 实验四A
  - raw-domain cross-channel probe
task: 08-16-raw-domain-probe
type: diagnosis (raw-domain representation information probe)
dataset: modma
fold: 0
status: "完成 (verdict H-VAE, 经 k8 近箱单判据触发)"
tags:
  - experiment
  - diagnosis
  - cross-channel
  - probe-matrix
  - modma
  - raw-domain
  - result-positive-with-caveats
checkpoint: null   # 本实验不加载任何模型权重；冻结输入（pairs npz / 08-14 审计 / DEFlow ckpt / norm cache / geometry）只读 sha1 戳于正文与 verdict.json
variant: null
related_spec: "[[perturbation]] / [[feasibility-gate]]"
---

# Raw-Domain 跨通道信息探针（实验四A：H-VAE vs H-DATA）

> Task `08-16-raw-domain-probe`。[[08-15-substrate-cross-channel-probe]] 的
> BRANCH 3（无米之炊）判决留下两个未分离的解释：**H-VAE**——原始信号里有
> 可计数的 $x_{\mathrm{fc},b} \sim x_{\mathrm{obs},a}$ 跨通道转迁信息，是
> per-channel DEVAE（$D{=}8, T_{\mathrm{lat}}{=}16 \to d{=}128$）压缩把它
> 摧毁了；**H-DATA**——原始域在 $L{=}128$ horizon 上同样贫乏，与压缩无关。
> 本实验把 08-15 的探针机件**原封不动**搬到**原始归一化窗**上（per-channel
> 原始维度 $L{=}128$ 恰等于 latent $d{=}128$，探针几何完全同构），对两个
> 假设做预注册裁决。纯 CPU、无重训、无 `models/` 改动、val 全程不动。

> [!summary] 一句话结论
> 预注册判决 **H-VAE**——但只由 within 的 **k8 近箱单判据**触发
> （$\mathrm{CI\_low} = +0.0204 > \mathrm{floor} -0.0151 + 0.02 = +0.0049$），
> 聚合 off-diag 判据未过（$-0.0024 < +0.0065$）。**最重要的诚实警示**：
> 判决是**地板驱动**的，不是信号驱动的——raw 的被试内置换地板塌成负值
> （$-0.0135$，置换 refit 劣于持久性基线），使门槛降到 $\approx +0.005$；
> 而 raw 的点估计在所有箱都**低于** latent（off-diag within $+0.0261$ vs
> $+0.0379$），配对差 $\Delta(\text{raw}-\text{latent})$ 的 within 两箱 CI
> 均含 0、cross off-diag 显著为负（$-0.0333\ [-0.0656, -0.0048]$）。
> "raw 清了自己的地板"成立；"raw 比 latent 信息多"不成立。按预注册
> branch 字典的含义（表征修复优先：D-scan → t_lat scan → joint-latent
> minimal variant）与上述警示并读，判读留给讨论。

## 实验目的与假设

08-13（mixer 容量正常）→ 08-14（透传链 off sub-noise）→ 08-15（冻结 substrate
缺跨通道信息）的审计链把病因压到了表征层，但 08-15 的探针只测过 DEVAE 的
$\mu$。若信息在原始域存在而在压缩中丢失（H-VAE），修复方向是表征架构
（$D \uparrow$ / joint-latent / band-split）；若原始域同样缺席（H-DATA），
修复方向是任务设定本身（band-split raw probe 4b / horizon scan 4c）。
两假设共用同一判决量：与 08-15 完全一致的
$I[a,b] = R^2(\text{target}_b \sim \text{source}_a)$、同一 persistence
基线、同一 margin。

## 预注册块（frozen before the registered run）

### 判决规则（`probe_raw.decide_raw_probe`，pure function，单测钉死）

- **within 主判**（对齐 08-15 的 branch 顺序）：
  - `agg_ok` $= \mathrm{CI\_low}(\text{raw off-diag mean } R^2) > \mathrm{floor_{agg}} + 0.02$；
  - `near_ok` $= \mathrm{CI\_low}(\text{raw k8-near mean } R^2) > \mathrm{floor_{near}} + 0.02$；
  - **H-VAE $\iff$ agg_ok $\vee$ near_ok**；否则 **H-DATA**。
- **cross 次要**（transferability，报告不判）。
- **near-threshold 诚实规则**（08-13/08-14/08-15 纪律）：判决量（CI-low 对
  阈值）落在 $\pm 0.005$ 内一律打 flag 报原始值；**点估计-vs-margin 的贴近
  只进 summary 手工警示段，不设形式 flag**。

### Branch 字典（`RAW_BRANCH_DICTIONARY`，逐字冻结）

- **H-VAE**: raw within-split clears floor+0.02 at the aggregate off-diag OR
  the k8-near bin (the latent cleared neither, 08-15) → the per-channel DEVAE
  destroyed countable cross-channel transition info; dissect: D-scan → t_lat
  scan → joint-latent minimal variant.
- **H-DATA**: raw within clears neither → window-level cross-channel transition
  info is absent in the raw domain too; per-channel compression exonerated;
  next = band-split raw probe (4b) / horizon scan (4c).

### 阈值与常量（frozen）

| 量 | 值 |
|---|---|
| margin（CI_low 需超地板） | $0.02$（= 08-15 `BRANCH_MARGIN`） |
| near_tol（形式 flag 容差） | $0.005$（= 08-15 `NEAR_TOL`） |
| 近箱 k（`near_bin_mask`，自通道排除） | $8$（对齐 08-13 Teacher-L 约定） |
| 距离四分位箱数（off-diag 合并条目） | $4$（完备不重叠） |
| AC2 复现容差（latent 重跑 vs 08-15 归档 agg 三分量） | $1 \times 10^{-4}$ |

### 种子（frozen）

| 种子 | 值 |
|---|---|
| probe 主种子（splits / boot 母 rng / shuffle 母 rng） | $0$ |
| data_split（SubjectFoldSplitter + val carve） | $42$ |
| boot（`default_rng(seed+1)`，两域同 → 配对重采样） | $1$ |
| shuffle 母 rng（`default_rng(seed+2)`，派生地板种子） | $2$ |
| mlp_base（`seed+7`，仅 raw 域 MLP 对照） | $7$ |
| floor_within（母 rng 第 1 抽） | 1798679647 |
| floor_cross（母 rng 第 2 抽） | 561807779 |

### 规模与协议（frozen）

- **源**：08-14 的 8 个 AP 轴审计通道 $[70, 61, 56, 41, 38, 37, 26, 9]$；
  **目标**：全部 128 通道。
- **原始对重建**：逐被试 `MODMADataset(window_size=256, obs_size=128,
  group="MIX", norm_stats=train-only 缓存 fold0_modma_L128.npz)`，
  `shuffle=False`；$x_{\mathrm{obs}} = x_{\mathrm{target}}[:, :128]$、
  $x_{\mathrm{fc}} = x_{\mathrm{target}}[:, 128:]$，逐通道转置成
  $(N, 128, 128)$ float32。**仅 train+test**（train 6394 窗 / 38 被试
  sid 0–37；test 1673 窗 / 10 被试 sid 43–52 $= 38+5$，与归档 npz 编号
  一致）。被试表：`SubjectFoldSplitter(n_folds=5, seed=42).get_fold(0)` +
  `_split_train_val_subjects(train_files, 5, 42)`。
- **对齐硬门**（`verify_alignment`）：重建数组与 `fold0_pairs.npz` 非 val
  行按 train→test 块序逐位 `np.array_equal`（sid 与 split 标签）；失败
  立即中止。smoke 截断时退化为逐块前缀校验并如实报告。
- **探针**（`probe_matrix.run_probe`，08-15 机件零改动）：
  ridge（闭式批量，$\alpha$ 网格 `logspace(-2,3,12)`，10% inner carve
  选优后全量 refit）+ raw 域 MLP 非线性对照（128→256→256→128·128，
  AdamW 早停，200 epochs，仅 8 源，不是判决者）；
  $n_{\mathrm{boot}}{=}1000$（被试层聚类 bootstrap，冻结预测）；
  $n_{\mathrm{shuffle}}{=}0$（地板单独调）。
  **latent 配对参照**：同参对 `fold0_pairs.npz` 的 $\mu$ 重跑 ridge-only。
- **地板**（4 次 `shuffle_floor(..., return_rows=True)`，raw/latent ×
  within/cross）：被试内置换 $z_{\mathrm{fc}}$（`assert_no_cross_subject_shuffle`
  守卫），$n_{\mathrm{shuffle}}{=}5$（对齐 08-15 注册协议），在已选
  $\alpha$ 处 refit；masks 由 `build_splits(sid, split, seed=0)` 重建
  （与 run_probe 内部一致）；从 rows 导出 off-diag / k8 / 四分位箱地板
  （`floor_from_rows`；off-diag 语义与 08-15 聚合地板完全一致，单测钉死）。
- **AC2 复现硬门**：latent 重跑 agg_ridge（value/ci_low/ci_high 三分量）
  必须复现 08-15 归档值 within $+0.037888\ [+0.018582, +0.062929]$、
  cross $+0.049483\ [+0.014994, +0.083130]$，容差 $10^{-4}$；超差即
  配对前提破坏，**停止并报告，不写任何产物**。latent off-diag 地板与
  归档地板（$+0.014395$ / $+0.023061$）的一致性为软校验（WARN 不中止）。
- **配对差** $\Delta(\text{raw} - \text{latent})$：两域同 seed → 同
  bootstrap 重采样 → 逐 resample 配对差（off-diag 与 k8 两箱，各带 CI）。
- **$R^2$ 约定**：$1 - \mathrm{MSE}/L_{\mathrm{persist}}$，
  $L_{\mathrm{persist}}$ = `learnability._persistence_baseline`（T-002D），
  与 08-15 可直比。
- **报告量**（summary.md 头表）：两域 × 两 split 的 off-diag 聚合 [CI]
  （含地板）、k8 箱 [CI]（含地板）、对角锚 $I[a,a]$ [CI]、配对差
  $\Delta$ off/k8 [CI]、四分位剖面、Spearman(电极距离, $R^2_{\mathrm{off}}$)
  局部性、raw MLP 对照聚合。

### 判决量到 branch 的映射表（AC 级单测钉死）

| (agg_ok, near_ok) | branch |
|---|---|
| (T, *) 或 (*, T) | H-VAE |
| (F, F) | H-DATA |

## 设计与执行

- **库**：`analysis/fm_gate/probe_raw.py`（新增；判决常量与 branch 字典
  冻结于内）。`probe_matrix.shuffle_floor` 增加可选参 `return_rows`
  （默认 False，默认路径行为不变，回归测试钉死）——08-15 冻结机件的
  **唯一** 改动，其余全部 additive。
- **runner**：`scripts/run_probe_raw.py`。执行顺序：被试表 → norm 缓存 →
  原始重建 → 对齐硬门 → **latent 参照（便宜）→ AC2 硬门 → raw 探针
  （贵，ridge+MLP）** ——AC2 前置使配对前提破坏时在烧掉数小时算力前中止。
- **实现期修复**（均在注册运行前、由单测钉死）：草稿 `quartile_bin_masks`
  误用 offdiag mask 的补集（装箱了对角条目）；草稿 `verify_alignment`
  整段前缀比较在 smoke 的 train/test 块边界必假失败，改为**逐块前缀
  校验**（full 语义不变）。
- **测试**：`tests/test_probe_raw.py` 19 项（对齐块序/前缀/失配即抛、
  near-bin 自通道排除与 k 选择、四分位完备不重叠、diag_anchor、
  paired_delta、floor_from_rows 语义一致、decide_raw_probe 分支表 +
  near flag、shuffle_floor return_rows 默认路径回归等）。
- **算力**：实际 **14186 s（≈3.9 h CPU）**：latent 参照 4224 s、raw
  ridge+MLP+bootstrap 9744 s、4 域地板 172 s、原始重建 7 s、产物落盘
  ~70 s（raw pairs 压缩 0.91 GiB，写前磁盘余量 211 GiB）。
- **冒烟**：`--smoke` 端到端通过、零产物写入；冒烟数字未进本笔记。

## 结果

### 判决：H-VAE（经 k8 近箱单判据触发；AC2 复现逐位精确）

| 判据 | 数值 | 阈值 | 结果 |
|---|---|---|---|
| `within agg_ok` | $\mathrm{CI\_low} = -0.00238$ | floor $-0.01348 + 0.02 = +0.00652$ | **False** |
| `within near_ok` | $\mathrm{CI\_low} = +0.02039$ | floor $-0.01507 + 0.02 = +0.00493$ | **True** |
| H-VAE $\iff$ agg_ok $\vee$ near_ok | — | — | **True → H-VAE** |

`near_threshold = False`（两判决量距阈值分别 $0.0089$ / $0.0155$，均超
near_tol $0.005$；无形式 flag）。cross 次要读数：`agg_ok` False、
`near_ok` True（与 within 同构）。**AC2 硬门**：latent 重跑 agg 三分量
两 split 均 $\max|\mathrm{diff}| = 0$（逐位复现 08-15 归档值）；latent
地板软校验同样 $0$ 偏差（$+0.014395$ / $+0.023060$）——配对前提成立到
机器精度。

### 头表：两域 × 两 split（判决量 + 地板 + 对角锚）

| domain | split | off-diag $R^2$ [CI] | floor | k8 near [CI] | floor | diag $I[a,a]$ [CI] |
|---|---|---|---|---|---|---|
| raw | within | $+0.0261$ $[-0.0024, +0.0573]$ | $-0.0135$ | $+0.0605$ $[+0.0204, +0.1003]$ | $-0.0151$ | $+0.0753$ $[+0.0296, +0.1187]$ |
| raw | cross | $+0.0169$ $[+0.0007, +0.0250]$ | $-0.0109$ | $+0.0563$ $[+0.0258, +0.0785]$ | $-0.0120$ | $+0.0796$ $[+0.0355, +0.1114]$ |
| latent | within | $+0.0379$ $[+0.0186, +0.0629]$ | $+0.0144$ | $+0.0672$ $[+0.0446, +0.0960]$ | $+0.0240$ | $+0.0824$ $[+0.0581, +0.1135]$ |
| latent | cross | $+0.0495$ $[+0.0150, +0.0831]$ | $+0.0231$ | $+0.0790$ $[+0.0422, +0.1134]$ | $+0.0287$ | $+0.0978$ $[+0.0583, +0.1368]$ |

（latent 行 = 08-15 归档值的逐位复现，作配对参照；对角锚 $I[a,a]$ 为
本次新增的标尺——raw 自通道参考约为其 off-diag 均值的 2.9×。）

### 配对差 $\Delta(\text{raw} - \text{latent})$（同 bootstrap 重采样）

| split | off-diag [CI] | k8 near [CI] |
|---|---|---|
| within | $-0.0137$ $[-0.0373, +0.0044]$ | $-0.0091$ $[-0.0384, +0.0144]$ |
| cross | $-0.0333$ $[-0.0656, -0.0048]$ | $-0.0242$ $[-0.0607, +0.0069]$ |

### 四分位剖面（近 → 远；括号内地板）

| domain | split | q0 [CI] (floor) | q1 [CI] (floor) | q2 [CI] (floor) | q3 [CI] (floor) |
|---|---|---|---|---|---|
| raw | within | $+0.0388$ $[+0.0070, +0.0721]$ ($-0.0140$) | $+0.0149$ $[-0.0117, +0.0460]$ ($-0.0132$) | $+0.0139$ $[-0.0091, +0.0400]$ ($-0.0134$) | $+0.0367$ $[+0.0041, +0.0719]$ ($-0.0133$) |
| raw | cross | $+0.0306$ $[+0.0101, +0.0431]$ ($-0.0115$) | $+0.0026$ $[-0.0132, +0.0094]$ ($-0.0110$) | $+0.0051$ $[-0.0065, +0.0101]$ ($-0.0106$) | $+0.0292$ $[+0.0101, +0.0394]$ ($-0.0104$) |
| latent | within | $+0.0488$ $[+0.0283, +0.0754]$ ($+0.0176$) | $+0.0295$ $[+0.0111, +0.0545]$ ($+0.0125$) | $+0.0298$ $[+0.0121, +0.0533]$ ($+0.0130$) | $+0.0434$ $[+0.0231, +0.0694]$ ($+0.0145$) |
| latent | cross | $+0.0585$ $[+0.0233, +0.0919]$ ($+0.0230$) | $+0.0408$ $[+0.0059, +0.0766]$ ($+0.0225$) | $+0.0440$ $[+0.0096, +0.0786]$ ($+0.0245$) | $+0.0547$ $[+0.0179, +0.0884]$ ($+0.0222$) |

（两域同构的 **U 形**剖面：q0 最高、q3 次高、q1/q2 低——不是单调的
近强远弱。）

### 局部性与 MLP 对照

- Spearman(电极距离, $R^2_{\mathrm{off}}$)：raw within $-0.0111$
  $[-0.0473, +0.0116]$（含 0）；raw cross $+0.0147$ $[-0.0169, +0.0305]$
  （含 0）；latent within $-0.0623$ $[-0.1273, -0.0039]$（不含 0，微弱
  远距衰减）；latent cross $-0.0297$（含 0）。
- raw MLP 对照聚合：within $+0.0329$ $[+0.0032, +0.0670]$（vs ridge
  $+0.0261$，非线性增益 $+0.007$）；cross $+0.0172$（vs $+0.0169$，
  增益 $\approx 0$）——线性主导，与 08-15 同构。

## 分析与结论

- **是否达成假设**：达成（预注册判据全部落表，判决函数按冻结规则输出
  H-VAE）。判决路径：within `agg_ok` False + `near_ok` True → H-VAE。
  按 branch 字典，含义是**表征层修复优先**（D-scan → t_lat scan →
  joint-latent minimal variant），H-DATA 路线（band-split 4b / horizon
  scan 4c）降为后备。
- **但判决的机械成因是地板不对称**（见警示 a）：raw 域被试内置换地板为
  负（置换 refit 劣于持久性基线），latent 域地板为正（被试均值结构在
  $\mu$ 中存活置换——08-15 警示 b 的同一机制）。H-VAE 的触发等价于
  "raw k8 近箱的 CI 下界（$+0.0204$）高于一个负地板加 margin 的低门
  （$+0.0049$），越线 $+0.0155$"。**反事实对照**：若把 08-16 的同一
  判决规则（agg $\vee$ k8）套到 latent 域上，latent 的 k8 CI 下界
  $+0.0446$ 也**越线**其门槛 $+0.0440$（$+0.0240+0.02$）——只差
  $0.0006$ 地擦过、且必带 near_threshold flag——即两域在各自门槛下
  都由 k8 单判据"显著"，raw 越线更宽、latent 点估计反而更高。branch
  字典里"the latent cleared neither (08-15)"指的是 08-15 当时注册的
  **聚合**判据（k8 近箱读数是 08-16 新增的，08-15 不存在）。
- **不做超出预注册的解释**：H-VAE 的字面叙事（"压缩摧毁了可计数信息"）
  与配对差读数（raw 不优于 latent）之间的张力如何裁决——例如是否把
  判决量改为"配对差 CI 排除 0"或"两域同门槛对比"——留给讨论与后续
  预注册。

## 诚实警示（Honest caveats）

- **(a) 判决是地板驱动的，不是信号驱动的（最重要）**：raw 地板为负
  （within off-diag $-0.0135$ / k8 $-0.0151$；cross $-0.0109$ / $-0.0120$），
  latent 地板为正（$+0.0144$ / $+0.0240$；$+0.0231$ / $+0.0287$）。raw
  的所有点估计都**低于** latent 对应值（off-diag within $+0.0261$ vs
  $+0.0379$；k8 $+0.0605$ vs $+0.0672$；cross 差距更大）。配对差 within
  两箱 CI 均含 0，cross off-diag 显著为**负**（$-0.0333$，CI 不含 0）。
  即：H-VAE 触发靠的是"负地板 + 低门槛"，不是"raw 信号更强"。
- **(b) 聚合判据未过、branch 由单判据支撑**：`agg_ok` False
  （CI_low $-0.0024$，含 0）；只有 k8 近箱触发。k8 箱（64 条目）的抬升
  也不构成平滑距离梯度（见 d）。
- **(c) 点估计-vs-margin**（手工报告，无形式 flag）：raw within 点增益
  off-diag $+0.0395$、k8 $+0.0756$——这次点估计同样越过 $0.02$ margin
  （与 08-15 的"点过线、CI 不过"同款张力，方向相同但由 k8 箱放大）；
  cross off-diag 点增益 $+0.0278$、k8 $+0.0683$，同样过线。
- **(d) 四分位 U 形 + 局部性 ≈ 0**：q0 与 q3 抬升、q1/q2 低（两域同构，
  latent 也是 U 形）；raw Spearman(距离, R²) 含 0。k8 箱的"近"效应与
  连续距离不单调相容——近箱判据的抬升不能读成"距离衰减的局部转迁"。
- **(e) MLP 非线性增益边际**（within $+0.007$、cross $\approx 0$）：
  线性主导，与 08-15 同构；非线性救不了也不翻案。
- **(f) 覆盖范围**：8 个 AP 轴代表源 × 128 目标、fold 0、$L{=}128$
  horizon、train-only 归一化；结论在此覆盖内成立。
- **(g) 地板机制差异本身是域属性**：raw 置换地板为负说明置换后的
  refit 比"预测被试均值"还差（128 维原始向量的伪映射过拟合），latent
  地板为正说明压缩保留了被试均值结构——两者度量的是不同东西，跨域
  "清地板"对比的解释要谨慎（与 a 同源，单列以免混淆）。
- **(h) 算力**：总计 14186 s（latent 4224 s、raw 9744 s、地板 172 s），
  纯 CPU 8 线程；产物落盘含 0.91 GiB 压缩 raw 对。

## 可复现性

环境 `D:\program/anaconda/envs/npi/python.exe`，从 `D:\102-npi`
（`latent_flow/` 的父目录）运行——直接调 env 的 python，不用 `conda run`
（Windows gbk 崩溃）：

```bash
# 冒烟（零产物）
D:/program/anaconda/envs/npi/python.exe -u -m latent_flow.scripts.run_probe_raw --fold 0 --smoke

# 注册运行（写 results/de/probe_raw/；纯 CPU 实测 14186 s ≈ 3.9 h）
D:/program/anaconda/envs/npi/python.exe -u -m latent_flow.scripts.run_probe_raw --fold 0

# 全测试（本任务后：218 passed, 3 skipped = 基线 199 + 新增 19）
D:/program/anaconda/envs/npi/python.exe -m pytest latent_flow/tests/ -q
```

- **Provenance**（全量戳在 `verdict.json` `meta`）：pairs npz sha1
  `c4586b7521a6`；08-14 layerwise sha1 `10c392d29fc3`；ckpt sha1
  `fc722302e717`（hash-only，本实验不加载）；norm cache sha1
  `a0fbdfa3b210`；geometry sha1 `476c21310727`；08-15 归档 npz sha1
  `bf559465c6cd`；git rev `1d7636b`；种子全表与阈值见预注册块（与
  `meta.seeds` / `meta.thresholds` 一致）；AC2 对照值逐分量入 `meta.ac2`。
- **AC2 结果**：两 split 三分量 $\max|\mathrm{diff}| = 0.000$（逐位复现）；
  latent 地板软校验同样 $0$ 偏差。
- **$R^2$ 约定**：$1 - \mathrm{MSE}/L_{\mathrm{persist}}$（T-002D），
  与 08-15 可直比；聚合排除对角（$a=b$），对角锚单独报告。
- **新增代码全部 additive**：`analysis/fm_gate/probe_raw.py`、
  `scripts/run_probe_raw.py`、`tests/test_probe_raw.py`（19 测试）；
  `probe_matrix.py` 仅增 `return_rows` 可选参（默认路径不变，回归钉死）；
  `git diff -- models/` 为空。

## 下一步（候选方向，非承诺）

- **判读讨论**（最高优先）：H-VAE 的地板驱动成因（警示 a）如何影响修复
  排序——是否需要以"两域同门槛"或"配对差排除 0"重做一次预注册判定。
- **branch 字典的行动路径**（若 H-VAE 维持）：D-scan（D8/16/32/64 substrate
  已有）→ t_lat scan → joint-latent minimal variant，每步用本探针复检。
- **band-split raw probe（4b）/ horizon scan（4c）**：H-DATA 路线降为后备，
  但 U 形四分位剖面（警示 d）提示频带/远近结构值得单独拆。
- 逐窗去被试均值变体（08-15 下一步的共用项）：同时锐化两域地板的解释。

## 产出物索引

| 类型 | 路径 |
|---|---|
| 判决（branch + 判据原始数 + branch 字典 + provenance meta + AC2 对照） | `results/de/probe_raw/verdict.json` |
| 摘要（头表、配对差、四分位、局部性、MLP 对照、判决理由、手工警示段） | `results/de/probe_raw/summary.md` |
| 原始 $R^2$ 数组（两域矩阵、boot 样本、地板 scalars/samples/rows、masks+edges、deltas、locality、meta） | `results/de/probe_raw/probe_raw_r2.npz` |
| 原始对（仅 train+test，0.91 GiB 压缩，对齐逐位校验 full） | `results/de/probe_raw/fold0_raw_pairs.npz` |
| 运行日志（stdout 捕获，含 AC2 对照行、判决行、墙钟分段） | `results/de/probe_raw/run_registered.log` |
| 库模块 | `analysis/fm_gate/probe_raw.py` |
| 运行器 | `scripts/run_probe_raw.py` |
| 测试（19 项） | `tests/test_probe_raw.py` |
| 冻结输入（只读） | `results/fm_gate/fold0_pairs.npz`、`results/de/probe_matrix/probe_r2.npz`、`results/fm_gate/_norm_cache/fold0_modma_L128.npz`、`data/modma/geometry.pt` |

## 关联笔记

- 判决基线与协议母本：[[08-15-substrate-cross-channel-probe]]（BRANCH 3；本实验 AC2 硬门逐位复现其归档 agg 与地板）
- 审计链：[[08-14-frozen-fm-layerwise-perturbation-audit]]（off 透传 sub-noise）、[[08-13-diagonal-r-mixer-diagnosis]]（mixer 容量正常；Teacher-L k=8 的近箱约定）
- Spec：[[perturbation]]（审计家族契约）、[[feasibility-gate]]（gate 纪律与预注册判据）
