---
date: 2026-08-15
name: substrate-cross-channel-probe
aliases:
  - probe matrix
  - cross-channel probe
  - 跨通道信息探针
task: 08-15-substrate-cross-channel-probe
type: diagnosis (frozen-substrate representation information probe)
dataset: modma
fold: 0
status: "完成 (verdict branch-3 / 无米之炊)"
tags:
  - experiment
  - diagnosis
  - cross-channel
  - probe-matrix
  - modma
  - result-negative
checkpoint: checkpoints/de_flow/fold0_de_flow_best.pt   # read-only (sha1 fc722302e717)；本实验未产出新 checkpoint
variant: null
related_spec: "[[perturbation]] / [[feasibility-gate]]"
---

# Substrate cross-channel probe matrix（冻结 substrate 里到底有没有跨通道信息）

> Task `08-15-substrate-cross-channel-probe`。审计家族的第三棒：[[08-13-diagonal-r-mixer-diagnosis]]
> 证明 mixer 跨通道**容量**正常，[[08-14-frozen-fm-layerwise-perturbation-audit]]
> 证明扰动**透传链**上 off-diag 响应自 mixer 输出起 sub-noise（两级衰减）。
> 但这两棒都只审计了 FM 对扰动的**响应**，没有回答更上游的问题——**冻结表征
> （substrate）本身是否携带可读出的跨通道信息**。本实验绕开 FM，直接在
> `results/fm_gate/fold0_pairs.npz` 的 $\mu$ 上用探针回归
> $I[a,b] = R^2(z_{\mathrm{fc},b} \sim z_{\mathrm{obs},a})$（单源通道 obs latent →
> 全部 128 个目标通道的 future latent），以预注册的 branch 字典判定信息在哪一层
> 缺席。纯 CPU、无重训、无 `models/` 改动。

> [!summary] 一句话结论
> 预注册判决 **BRANCH 3 — 无米之炊**（no substrate information）：within 与 cross
> 两个 split 的 off-diag 均值 $R^2$ 的 CI 下界均未超过 floor $+0.02$ 阈值
> （within $+0.0186$ vs $+0.0344$；cross $+0.0150$ vs $+0.0431$），结构判据
> $|\rho(I_{\mathrm{cross}}, G_{\mathrm{off,mix}})| = 0.010$ 也远低于 $0.2$——在
> 8 个 AP 轴代表源 × 128 目标、128 维 pooled latent 的覆盖下，冻结 substrate
> 的**跨通道窗口级转迁信息不足以按预注册标准计数**。含义：FM / objective 级
> 修复在此 substrate 上**全部作废**，应转向表征/任务重构（$D \uparrow$、更短
> horizon、band-split latents）；跨被试墙被解释为表征问题的下游。注意：这是
> **CI 驱动**的判决——点估计增益（within $+0.0235$、cross $+0.0264$）其实都
> 越过了 $0.02$ margin，within 的 CI 下界也在 floor 之上（见诚实警示 a）。

## 实验目的与假设

08-13/08-14 把病因收窄到"扰动透传层"，但若表征里根本没有跨通道信息
（源通道的 obs latent 不含预测目标通道 fc latent 的窗口级信息），那么 mixer
学不到、FM 传不动都是**下游症状**而非病因。本实验用直接探针检验这一假设。
预注册 branch 字典（`verdict.json` `branch_dictionary`，判定顺序为 pure
function、AC5 单测钉死）：

- **branch 1** FM shortcut 确认：within + cross 均显著**且** mixer 结构化
  → 信息在、mixer 传、stack 不读 → 修复 = transmission-consistency objective，
  以 P4 反循环判据（held-out real-transition prediction gain）验收；
- **branch 2** 被试特异信息：within 显著、cross 不显著
  → 修复 = 被试泛化与透传并举；
- **branch 3** 无米之炊：within 即不显著
  → FM/objective 级修复全部作废，转向表征/任务重构（$D \uparrow$、更短
  horizon、band-split latents）；跨被试墙是表征的下游；叙事变为
  "fix the representation first, DE is downstream"；
- **branch 4** 最强 shortcut 证据：within + cross 均显著**且** mixer 均匀
  （P1 CV 小）→ objective 级修复最优先；若
  $\rho(W, I_{\mathrm{within}}) \gg \rho(W, I_{\mathrm{cross}})$ 则佐证 08-13 的
  "W 学到被试记忆结构"。

判决量（预注册）：$\mathrm{CI\_low}(R^2) > \mathrm{floor} + 0.02$（两个 split
各一），$\mathrm{structured} = |\rho(I_{\mathrm{cross}}, G_{\mathrm{off,mix}})| \geq 0.2$
且其 bootstrap CI 不含 0；任何判决量落在阈值 $\pm 0.005$ 内须标
`near_threshold` 并报原始值（08-13/08-14 的诚实纪律）。

## 设计与执行

- **数据**：`results/fm_gate/fold0_pairs.npz`（sha1 `c4586b7521a6`；
  $N{=}8881$ 窗，$C{=}128$ 通道，$d{=}128$ 维 pooled latent——$T_{\mathrm{lat}} \times D$
  展平）。split：train 6394 窗 / 38 被试，val 814 / 5，test 1673 / 10；
  val 全程未动（留给下游修复工作）。
- **探针**：$I[a,b] = R^2(z_{\mathrm{fc},b} \sim z_{\mathrm{obs},a})$，
  **row = 源 $a$，col = 目标 $b$**。主估计器为批量闭式 Ridge（一次解全部
  128 个目标，共享 Gram 矩阵；与 `sklearn.Ridge(solver='cholesky')` 的等价性
  由 AC2 单测保证，sklearn 不进热路径）；$\alpha$ 网格
  `logspace(-2, 3, 12)`，在 10% inner carve 上逐 (source, split) 选优后全量
  refit。$R^2 = 1 - \mathrm{MSE}/L_{\mathrm{persist}}$，
  $L_{\mathrm{persist}}$ 取 `learnability._persistence_baseline` 语义
  （T-002D 约定，可与既有 learnability 数字互比）。非线性对照：每源一个
  多任务 MLP（128→256→256→$128{\cdot}128$，AdamW，早停，仅 8 源）——
  只做非线性检查，**不是** branch 判定者。
- **两个 split**：within = train 被试内、逐被试 seeded 80/20 窗留出（测
  "已见被试的新窗"）；cross = train 全量拟合 → test 被试评估（encoder 从未
  见过 test 被试，探针也不在其上拟合）。
- **Shuffle floors**：$n_{\mathrm{shuffle}}{=}5$，在被试内置换 $z_{\mathrm{fc}}$
  的窗序（`assert_no_cross_subject_shuffle` 守卫，绝不跨被试），在已选
  $\alpha$ 处 refit 后在同一 eval 窗上评估。MLP 无 shuffle 臂（以 ridge floor
  为参考底，已记录）。
- **CI**：冻结预测 + 被试层聚类 bootstrap（`cluster_bootstrap_indices`，
  $n_{\mathrm{boot}}{=}1000$，重采样 eval 被试、重算 off-diag 均值 $R^2$ 与
  persistence 基线，percentile 2.5/97.5）。同一机制 CI 结构 Spearman 相关。
- **结构三角**（off-diag 公共条目，Spearman）：$FC_{\mathrm{emp}}$（从原始
  MODMA 窗新算，53 被试、逐被试 Pearson over pooled obs+fc timepoints、
  Fisher-z 平均，9.3 s，缓存 `fc_emp.npz`）；$W$ = `mixer.W`、
  $A_{\mathrm{eff}}$ = `mixer.effective_adjacency()`——两者按 orientation
  contract **转置后**比较；$G_{\mathrm{off,mix}}$ 取 08-14 审计（arm=ref，
  $r = r_{\max}$，860 窗平均）的 (source → target-channel) 模式，本身方向
  已一致。
- **源**：复用 08-14 的 8 个 AP 轴审计通道
  $[70, 61, 56, 41, 38, 37, 26, 9]$（E71, E62, E57, E42, E39, E38, E27,
  E10）；目标 = 全部 128 通道。
- **Orientation contract（最易藏 silent bug 的一处，design §2）**：探针矩阵
  row = 源 / col = 目标，而 mixer 约定 $A[i,j]$ = 通道 $j$ 驱动 $i$（row =
  目标 / col = 源，`models/oc_latent_flow.py:426`）——因此所有结构比较一律
  用 $W^{\top}$、$A_{\mathrm{eff}}^{\top}$，AC4 单测以合成有向矩阵钉死。
  check 阶段曾发现 `--full`（128 源）扩展路径的三角比较**漏了转置**，
  在注册运行**之前**捕获并修复；注册结果所走的 8 源路径自始正确。
- **$\mu$ 直接取自 npz（不 re-encode）**：纯 CPU 约束下的注册量就是均值
  探针；re-encode 采样版本留作后续。
- **算力**：ridge + mlp + floors + bootstrap 共 10528 s（约 2.9 h，CPU 8
  线程）；fc_emp 9.3 s。无 GPU 依赖。

## 结果

### 判决：BRANCH 3 — 无米之炊（no substrate information）

三个预注册判据（原始数字见 `verdict.json`）：

| 判据 | 数值 | 阈值 | 结果 |
|---|---|---|---|
| `within_ok` | $\mathrm{CI\_low} = +0.01858$ | floor $+0.01440 + 0.02 = +0.03440$ | **False** |
| `cross_ok` | $\mathrm{CI\_low} = +0.01499$ | floor $+0.02306 + 0.02 = +0.04306$ | **False** |
| `structured` | $\rho = +0.01030$，CI $[-0.0446, +0.0677]$ | $|\rho| \geq 0.2$ 且 CI 不含 0 | **False** |

`near_threshold = False`（五个形式 flag 全未触发；但见诚实警示 a 的手工
proximity 报告）。按预注册顺序 `not within_ok` → **branch 3**：表征里没有
（按此标准可计数的）跨通道转迁信息。

### 头表：off-diag 均值 $R^2$（预注册量）

| split | $R^2_{\mathrm{ridge}}$ [CI] | shuffle floor | $R^2_{\mathrm{mlp}}$ [CI] |
|---|---|---|---|
| within | $+0.0379$ $[+0.0186, +0.0629]$ | $+0.0144$ | $+0.0487$ $[+0.0279, +0.0753]$ |
| cross | $+0.0495$ $[+0.0150, +0.0831]$ | $+0.0231$ | $+0.0570$ $[+0.0224, +0.0910]$ |

（floor 为 5 次 shuffle 的均值；逐次 rep：within
$0.0135/0.0148/0.0154/0.0135/0.0148$，cross
$0.0238/0.0233/0.0220/0.0231/0.0231$。对角元 $a{=}b$ 是 self-history 参考
行，聚合只取 off-diagonal。）

### 结构 Spearman 三角（off-diag 公共条目；$W$/$A_{\mathrm{eff}}$ 已转置）

| $I$ | $FC_{\mathrm{emp}}$ | $W^{\top}$ | $A_{\mathrm{eff}}^{\top}$ | $G_{\mathrm{off,mix}}$ |
|---|---|---|---|---|
| $I_{\mathrm{within}}$ | $+0.061$ $[+0.019, +0.119]$ | $-0.240$ $[-0.288, -0.151]$ | $+0.466$ $[+0.310, +0.533]$ | $+0.080$ $[+0.020, +0.140]$ |
| $I_{\mathrm{cross}}$ | $+0.079$ $[+0.024, +0.126]$ | $-0.056$ $[-0.109, +0.011]$ | $+0.371$ $[+0.236, +0.418]$ | $+0.010$ $[-0.045, +0.068]$ |

- $G_{\mathrm{off,mix}}$ 逐源 CV（"P1 CV"，均匀性描述量）：均值 $0.5805$；
  逐源 70:0.503, 61:0.497, 56:0.565, 41:0.569, 38:0.547, 37:0.653,
  26:0.643, 9:0.667。
- branch-4 佐证对（此处按 branch 3 只作记录）：
  $\rho(W, I_{\mathrm{within}}) = -0.2403$ vs
  $\rho(W, I_{\mathrm{cross}}) = -0.0561$。

> [!example] 关键数值
> 三个锚点：(1) within 点增益 $0.0379 - 0.0144 = +0.0235$、cross 点增益
> $0.0495 - 0.0231 = +0.0264$——**点估计都越过 $0.02$ margin**，但 CI 太宽，
> 下界不够；(2) within $\mathrm{CI\_low} = +0.0186 > \mathrm{floor} = +0.0144$
> （被试内信息在统计上可检出、高于 floor 本身），cross $\mathrm{CI\_low} =
> +0.0150 < \mathrm{floor} = +0.0231$（跨被试连 floor 都未在 CI 下界越过）；
> (3) 最强结构对齐是 $A_{\mathrm{eff}}$ 与 $I$ 的 $+0.466$（within）/
> $+0.371$（cross）——但它对齐的是一个量级 sub-margin 的信息矩阵。

## 分析与结论

- **是否达成假设**：达成（预注册判据全部落表，branch 3 触发）。审计链闭环：
  08-13（mixer 容量正常）→ 08-14（透射链 off sub-noise）→ 08-15（上游表征
  本身缺跨通道信息）。三者合起来的因果叙事：**表征缺信息在先**，mixer 的
  稠密 $W$ 与透传衰减是对一个信息贫乏输入的响应，而非独立病因。
- **branch 3 的含义**：FM / objective 级修复（transmission-consistency
  objective、paired mean-shift augmentation 等 branch-1/4 药方）在当前
  substrate 上**无的放矢**——没有信息可教 FM 去传。修复方向上移到
  表征/任务层：$D \uparrow$、更短 horizon、band-split latents。跨被试墙
  被解释为表征问题的下游症状，而非独立的泛化故障。
- **$W$ 的双面孔**：$\rho(W, I_{\mathrm{within}}) = -0.24$（CI 不含 0）vs
  $\rho(W, I_{\mathrm{cross}}) = -0.06$（CI 含 0）——与 08-13 "W 学到被试
  记忆结构"的读法一致（$W$ 的模式追踪被试特异、不可迁移的信息）；负号是
  模式**反**对齐，如实报告（见警示 c）。
- **MLP 对照**：非线性只带来 $+0.011$（within）/ $+0.008$（cross）的边际
  增益——非线性救不了，与"信号以线性为主（或根本缺席）"一致。

## 诚实警示（Honest caveats）

- **(a) 判决是 CI 驱动的，不是点估计驱动的（最重要）**：within 点增益
  $+0.0235$、cross 点增益 $+0.0264$ **都超过** $0.02$ margin；within 的
  $\mathrm{CI\_low}$（$+0.0186$）高于 floor 本身（$+0.0144$）——即被试内
  信息在统计上可检出、高于 floor，只是效应量贴着 margin 且 CI 对预注册
  判据太宽。**branch 3 在这里的含义是"效应太小/太异质，不足以计数"，不是
  "精确为零"**。形式 `near_threshold` flag 未触发，因为预注册判决量是
  CI-low-vs-threshold（within 差 $0.0158$、cross 差 $0.0281$，均超
  `near_tol` $0.005$）；点增益-vs-margin 的贴近是另一个（未预注册的）量，
  此处按 08-13/08-14 诚实纪律手工报告。
- **(b) floors 非零的含义**：被试内置换保留被试层均值结构，故 floors
  （$+0.0144$ / $+0.0231$）度量的是**被试均值信息**而非窗口级转迁信息；
  真臂必须在 CI 上胜过 floor $+ 0.02$，未做到。
- **(c) $W$ 的负相关是"结构对齐"证据，不是"信息通路"证据**：
  $\rho(W, I_{\mathrm{within}}) = -0.24$（CI 不含 0）vs
  $\rho(W, I_{\mathrm{cross}}) = -0.06$（CI 含 0）与 08-13 "W 学到被试记忆
  结构"相容（$W$ 模式追踪被试特异、不可迁移的信息）；符号为负，是模式
  **反**对齐，按此报告。
- **(d) $A_{\mathrm{eff}}$ 与 $I$ 的正相关（$+0.47$ / $+0.37$）是本次找到的
  最强结构对齐——但在 branch 3 之下，它对齐的是一个量级 sub-margin 的
  信息矩阵；不要过度推销。**
- **(e) `--full`（128 源全图）未运行**：branch 选择用不到它；留作表征修复
  任务规划的可选后续。**冒烟模式的 full 数字不得被引用。**
- **(f) 覆盖范围**：8 个 AP 轴代表源、128 维 pooled latent
  （$T_{\mathrm{lat}} \times D$ 展平）——空间与维度覆盖均有限，branch 3 的
  结论在此覆盖内成立。
- **(g) MLP 相对 ridge 增益很小**（within $+0.011$、cross $+0.008$）：
  非线性不救援，与线性主导（或缺席）的信号一致。
- **(h) $FC_{\mathrm{emp}}$ 是新算的**（53 被试、9.3 s、缓存
  `fc_emp.npz`）；重叠窗会重复计数时间点——只作 anchor 用。
- **(i) 算力**：ridge+mlp+floors+bootstrap 10528 s（约 2.9 h CPU）；
  fc_emp 9.3 s。

## 可复现性

环境 `D:\program\anaconda\envs\npi\python.exe`，从 `D:\102-npi`
（`latent_flow/` 的父目录）运行——直接调 env 的 python，不用 `conda run`
（Windows gbk 崩溃）：

```bash
# 冒烟
python -u -m latent_flow.scripts.run_probe_matrix --fold 0 --smoke

# 注册运行（写 results/de/probe_matrix/；纯 CPU 约 2.9 h）
python -u -m latent_flow.scripts.run_probe_matrix --fold 0

# 全测试
python -m pytest latent_flow/tests/ -q     # 199 passed, 3 skipped
```

- **Provenance**（全量戳在 `verdict.json` `meta`）：pairs npz sha1
  `c4586b7521a6`；08-14 layerwise npz sha1 `10c392d29fc3`；ckpt sha1
  `fc722302e717`（只读）；git rev `8117252`；fc_emp 输入 sha
  `1a60c7622764`（53 被试，本次新算非缓存）。pairs 的 $\mu$ 由 encoder
  `checkpoints/de_vae/fold0_de_vae_D8_b0.01_mkl_variantA_best.pt`
  提取（`verdict.json` `meta.pairs_meta_ckpt` 原样记录）。
- **种子**：probe=0、data_split=42、boot=1、shuffle=2、mlp_base=7；
  $\alpha$ 网格 `logspace(-2, 3, 12)`（10% inner carve 选优）；阈值
  margin $0.02$ / $\rho_{\mathrm{thresh}} 0.2$ / near_tol $0.005$。
- **$R^2$ 约定**：$1 - \mathrm{MSE}/L_{\mathrm{persist}}$，
  $L_{\mathrm{persist}}$ = `learnability._persistence_baseline`
  （T-002D 约定）。
- **P1/P4 标签映射**（用户外部标签，不在 repo 内）：P1 = 08-14
  $G_{\mathrm{off,mix}}$ 读出；P4 = 反循环验收（held-out real-transition
  prediction gain）。
- **新增代码全部 additive**：`analysis/fm_gate/probe_matrix.py`、
  `scripts/run_probe_matrix.py`、`tests/test_probe_matrix.py`（15 个新
  测试，含 AC2 ridge 等价、AC4 orientation contract、AC5 branch 表）；
  `git diff -- models/` 为空（无重训、无模型改动）。

## 下一步（候选方向，非承诺）

- **branch 3 的表征修复方向**：$D \uparrow$（d_ablation 结果已有
  D8/16/32/64——在重训后的 $D \uparrow$ substrate 上重跑本探针）、更短
  horizon、band-split latents。
- **可选 `--full` 128 源全图**：为修复规划提供完整 $I$ 图（仅 ridge 路径）。
- **同探针的逐窗去被试均值变体**：在 subject-mean-removed latents 上重跑，
  把"被试均值信息"与"窗口级转迁信息"分开——同时锐化 floor 的解释
  （针对警示 b）。

## 产出物索引

| 类型 | 路径 |
|---|---|
| 判决（机器可读：branch + 判据原始数 + branch 字典 + provenance meta） | `results/de/probe_matrix/verdict.json` |
| 摘要（人类可读：头表、三角、判决理由） | `results/de/probe_matrix/summary.md` |
| 原始 $R^2$ 数组（ridge/MLP 矩阵、floors、选中 $\alpha$、CI、三角、CV、meta） | `results/de/probe_matrix/probe_r2.npz` |
| $FC_{\mathrm{emp}}$ 缓存（被试列表 + 输入 sha） | `results/de/probe_matrix/fc_emp.npz` |
| 运行日志（stdout 捕获，含 shuffle reps、10528s、判决行） | `C:\Users\10618\.zcode\cli\exec\sess_5399ff2a-7a44-49c8-b1d1-a2d50aaef72d\call_ecffa34f8a954d3a9fde89e4-stdout.log` |
| 库模块（估计器 / splits / floors / 结构统计 / 判决） | `analysis/fm_gate/probe_matrix.py` |
| 运行器 | `scripts/run_probe_matrix.py` |
| 测试（15 个新测试） | `tests/test_probe_matrix.py` |
| 冻结输入（只读） | `results/fm_gate/fold0_pairs.npz`、`results/de/layerwise_audit/layerwise_g.npz`、`checkpoints/de_flow/fold0_de_flow_best.pt` |

## 关联笔记

- 前序 / 审计链：[[08-14-frozen-fm-layerwise-perturbation-audit]]（off-diag 自 mixer 输出起 sub-noise——本实验将其上溯到表征层）
- 容量排除：[[08-13-diagonal-r-mixer-diagnosis]]（mixer 容量正常；"W 学到被试记忆结构"本次被 $\rho(W, I)$ 对佐证）
- 动机 / 被解释的发现：[[08-10-de-inference-prep-ijkl]]（diagonal-R、off 响应低于 Null 底）
- Spec：[[perturbation]]（审计家族契约）、[[feasibility-gate]]（gate 纪律与判据预注册）
- 关联 checkpoint：见 frontmatter `checkpoint`（只读）
