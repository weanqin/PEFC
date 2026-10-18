---
date: 2026-08-14
name: frozen-fm-layerwise-perturbation-audit
aliases:
  - layerwise audit
  - Experiment 2
  - 层级扰动审计
task: 08-14-frozen-fm-layerwise-perturbation-audit
type: diagnosis (frozen-model layerwise perturbation-transmission audit)
dataset: modma
fold: 0
status: "完成 (verdict case-1-via-null-floor / 两级衰减结构)"
tags:
  - experiment
  - diagnosis
  - layerwise-audit
  - cross-channel
  - modma
  - near-threshold
checkpoint: checkpoints/de_flow/fold0_de_flow_best.pt   # read-only (sha1 fc722302e717)；本实验未产出新 checkpoint
variant: null
related_spec: "[[perturbation]]"
---

# Frozen-FM layerwise perturbation audit（Experiment 2 — 扰动在哪一层死掉）

> Task `08-14-frozen-fm-layerwise-perturbation-audit`。**Experiment 2**，直接门控于
> [[08-13-diagonal-r-mixer-diagnosis]] 的判决（mixer 跨通道容量基本正常，diagonal-R
> 病因被定位到**扰动透传层**）与 [[08-10-de-inference-prep-ijkl]] 的发现
> （$R_{a,b}(r_{\max})$ 有效对角、off 响应在 $r_{\max}$ 处低于 Null 底）。本实验
> 对**冻结的** `DEFlow`（无重训、无 `models/` 改动）注入 Mahalanobis 匹配的成对
> 扰动，沿四个内部层逐层捕获响应，回答"$a$ 通道扰动到底在哪一层传不到 $b$ 通道"。

> [!summary] 一句话结论
> 形式判决 4/4 格（arm × r）均为 case 1 "dies at mixer output"——但**全部经 null-floor 分支触发而非级联骤降分支**：off-diag 响应**从 mixer 输出起就低于后验噪声底**（这解释了 08-10 的 "R below null"——它在 mixer 输出处已经成立，不是解码伪影）；结构上没有单层湮灭，是 **velocity stack 与 decoder 两级相继的约 $14\times$ 衰减**把 off 份额从 $6.6\%$ 压到 $0.035\%$（解码端 diag/off 抑制 $2848\times$）；mixer 本身**确实透传**跨通道扰动（off/diag $= 6.6\times10^{-2}$，与 08-13 的稠密 $W$ 一致），"mixer 什么也不传"为假；扰动几何失配假说（Arm B 救援）被**否证**（mixer 处 data/ref $= 1.04$）。

## 实验目的与假设

08-10 在冻结 substrate 上测得：$a$ 通道的 $q{=}0.99$ 尺度 latent 均值位移只动 $a$
自己的预测（diag/off $\approx 3000{-}8000\times$），且 off 响应低于后验噪声底。
08-13 排除了 mixer 容量问题。本实验沿信号通路逐层定位衰减点。预注册的四个 case
（见 `verdict.json` `case_labels`）：

- **case 0** `no_collapse_through_decode` — 全链路无坍缩；
- **case 1** `dies_at_mixer_output` — 扰动在 mixer 输出处即消失；
- **case 2** `velocity_ignores_cross_channel` — velocity 忽略跨通道输入；
- **case 3** `ode_contraction` — ODE rollout 收缩掉扰动；
- **case 4** `decoder_absorbs_latent_shift` — 解码器吸收 latent 位移。

坍缩规则（预注册）：$G_{\mathrm{off},\ell} < \max(0.05\,G_{\mathrm{off},\ell-1},\; 1.1\times\text{null\_floor}_\ell)$；
首层（mix0）只看 $1.1\times$ null floor。**几何子假设**：若 mixer 处
data/ref $> 3$ 则判"扰动几何失配"（Arm B 救援成立）。

## 设计与执行

- **冻结 substrate**：`checkpoints/de_flow/fold0_de_flow_best.pt`（sha1
  `fc722302e717`，git `876ce87`），**无重训**；复用归档 ref stats
  `results/de/fold0/ref_stats.npz`（sha1 `1e67603dc722`，$r_{\max}$ 钉在
  **1.3930**）。
- **两条扰动臂**（逐方向 Mahalanobis 匹配）：
  - **Arm A（ref）**：$\delta_{\mathrm{ref}} = r \cdot L_{\mathrm{ref}} \xi_k$，**逐方向**注入——**不是** 08-10 的 K-mean 塌缩协议；
  - **Arm B（data）**：$\delta_{\mathrm{data}} = \alpha_k u_k$，$u_k$ 取逐源训练 $\mu_{\mathrm{obs},a}$ 协方差（$N_{\mathrm{full}}{=}7208$）的 top-8 主成分，$\alpha_k$ 匹配使 $\|\delta_{\mathrm{data}}\|_{S_{\mathrm{ref}}^{-1}} = r\|\xi_k\|$。
- **层级捕获**（逐步 Euler，$n_{\mathrm{steps}}{=}10$；主读出在 step 0）：mixer
  输出 $h$（hook 在 `graph_mixer`）→ 速度场 $v$ → $z$ 轨迹 → 最终 latent → 解码
  $R$。$G$ 定义：逐通道 L2 差 $/\,(2\|\delta_k\|_2)$，逐方向归一后再对 $K$ 平均。
- **共享随机性契约**（decision 13）：base/$\pm$/两臂共用同一 $\varepsilon$；
  $\xi_k$ 跨 r、跨源同一组；Null 的 $\varepsilon'$ 取 seed+2。
- **源**（region 模式，AP 轴 PC1 分位）：E71, E62, E57, E42, E39, E38, E27,
  E10（$K{=}8$ 方向，$P{=}2$ 后验采样，bootstrap 1000 次）。
- **窗口**：$N_{\mathrm{full}}{=}7208 \rightarrow N'{=}860$ rollout 窗（wps=20,
  seed 0，43 subjects）；$r \in \{0,\ 0.5r_{\max},\ r_{\max}\}$。
- **算力**：主循环 11925 s（约 3.3 h，共享 12 GB GPU，含编码）。
- **不变量验证**（run log）：$r{=}0$ 时内部各层 $\max|G| = 0.0$（**EXACT
  ZERO**），$\max|R| = 0$ —— decision-13 契约成立。

## 结果

### 判决：4/4 格 case 1，但坍缩全部经 null-floor 分支触发

四个（arm × r）格的形式判决均为 **case 1**（首坍层 mix0）。但逐层看触发分支：
**每一层**的 $G_{\mathrm{off}}$ 都低于自身 null floor 的 $1.1\times$（以 ref
$r_{\max}$ 为例：mix0 的 $G_{\mathrm{off}}{=}0.0794$ vs floor $2.93$；R 的
$3.9\times10^{-4}$ vs floor $0.818$）。**正确读法**：off-diag 的 $\delta$ 响应
**从 mixer 输出起就淹没在后验噪声之下**——$q{=}0.99$ 尺度的 latent 均值位移对
内部状态的扰动**小于一次新的后验噪声重采样**。这镜像并解释了 08-10 的
"R below null"：它在 mixer 输出处**已经成立**，不是解码伪影。注意对角通量在解码
端**高于** R null floor（$R_{\mathrm{diag}} = 1.107 > 0.818$）——**只有
off-diag 低于噪声**。另：null floor 随 $r$ 反比缩放（mix0 floor 从 $5.86$
降到 $2.93$，恰为 r 之比），故 $r_{\max}$ 处的 sub-noise 判决是最强的。

### 四层级联（arm=ref，$r = r_{\max} = 1.3930$；主表）

| layer | $G_{\mathrm{diag}}$ | $G_{\mathrm{off}}$ | off/diag | $G_{\mathrm{off}}$ CI (bootstrap) | null floor |
|---|---|---|---|---|---|
| mix0 | 1.2058 | 7.9401e-2 | 6.585e-2 | [7.933e-2, 7.948e-2] | 2.932 |
| vel0 | 0.6520 | 5.869e-3 | 9.003e-3 | [5.611e-3, 6.163e-3] | 0.1440 |
| roll | 0.5091 | 5.411e-3 | 1.063e-2 | [5.193e-3, 6.660e-3] | 0.2136 |
| R | 1.1068 | 3.886e-4 | 3.511e-4 | [2.549e-4, 5.736e-4] | 0.8179 |

（CI 为 subject 层聚类 bootstrap。其余三格完整表见
`results/de/layerwise_audit/summary.md`；四格的 $G_{\mathrm{off}}$ 级联紧凑版 —
ref 0.5r_max：$0.0795 \rightarrow 0.00587 \rightarrow 0.00542 \rightarrow 8.93\times10^{-5}$；
data 0.5r_max：$0.0828 \rightarrow 0.00638 \rightarrow 0.00564 \rightarrow 6.86\times10^{-4}$；
data r_max：$0.0823 \rightarrow 0.00637 \rightarrow 0.00601 \rightarrow 3.47\times10^{-3}$。）

### 级联 $\tau$ 与两级衰减（ref，$r_{\max}$）

| 跃迁 | $\tau_{\mathrm{off}}$ | 含义 |
|---|---|---|
| mix0 → vel0 | **0.0739**（$13.5\times$ 衰减；diag 仅 $1.85\times$） | velocity stack 是第一个大衰减器；off/diag $6.6\times10^{-2} \rightarrow 9.0\times10^{-3}$（相对抑制 $7\times$） |
| vel0 → roll | **0.922** | ODE rollout **不收缩**（case 3 被排除） |
| roll → R | **0.0718**（off/diag $1.06\times10^{-2} \rightarrow 3.5\times10^{-4}$，再压 $30\times$） | decoder 吸收真实存在；但 $\tau$ 混单位（见 caveat c） |

结构结论：**没有单层湮灭**——两级相继的约 $14\times$ 衰减（velocity stack、
然后 decoder）把 off 份额从 $6.6\%$ 压到 $0.035\%$。注意
$\tau_{\mathrm{off}}(\mathrm{vel0/mix0}) = 0.074$ 与
$\tau_{\mathrm{R/roll}} = 0.072$ 都**恰好悬在预注册 $0.05$ 阈值之上**——级联骤降分支在 $\tau$ 上并不
单独触发；这是继 08-13 差 $0.009$ 未达阈值之后的又一个 near-threshold 结果，
报告原始比值，不四舍五入成单一凶手。（ref $0.5r_{\max}$ 处 R/roll
$\tau = 0.0165$，小 r 下 decoder 相对吸收更强。）

### 逐步步轨迹：stack 在持续淬火一个增长的 mixer 响应（ref，$r_{\max}$）

| step | $G_{\mathrm{off}}$ mix | $G_{\mathrm{off}}$ vel | $G_{\mathrm{off}}$ ztraj |
|---|---|---|---|
| 0 | 7.940e-2 | 5.869e-3 | 0.0 |
| 1 | 8.206e-2 | 5.266e-3 | 5.870e-4 |
| 2 | 1.059e-1 | 5.274e-3 | 1.045e-3 |
| 3 | 1.391e-1 | 5.497e-3 | 1.502e-3 |
| 5 | 2.225e-1 | 5.904e-3 | 2.517e-3 |
| 7 | 3.119e-1 | 6.149e-3 | 3.635e-3 |
| 9 | 3.992e-1 | 6.306e-3 | 4.810e-3 |

（完整 10 步见 summary.md。）mixer 的 off 响应沿 rollout 从 $0.079$ **增长**到
$0.399$，而 vel/ztraj 的 off 响应几乎平直（$5{-}6\times10^{-3}$）——每一步，
velocity stack 都把一个持续增长的 mixer 响应淬回去。case 3（ODE 收缩）不仅被
$\tau = 0.922$ 排除，逐步表还给出了**反向**画面。

### 几何子判决：扰动几何失配**否证**

| 量 | $r = 0.5r_{\max}$ | $r = r_{\max}$ |
|---|---|---|
| mixer 处 data/ref（$G_{\mathrm{off}}$） | 1.041 | 1.037 |
| 判决 | 失配标志 **false** | 失配标志 **false** |

FM 透传 data-manifold 方向**不比** $S_{\mathrm{ref}}$ 形方向更好——各 G 层基本
同幅（vel0 1.09，roll 1.04–1.11）。次级观察：**R 层** data/ref 达
$7.7{-}8.9\times$（decoder 对 PC 形状的 latent 位移吸收较少，data 臂 R off 在 $r_{\max}$
处 $3.47\times10^{-3}$）——但仍低于 null floor $236\times$，记为二阶观察，
不构成救援。

### AC5 锚点

> [!example] 关键数值
> arm=ref 解码端 $r_{\max}$：diag $= 1.1068$，off $= 3.886\times10^{-4}$ →
> **diag/off 抑制 $2848\times$**——在逐方向协议下复现了 08-10 K-mean 协议的
> $3000{-}8000\times$ 定性结论（协议与窗口子采样差异，不必精确匹配）。
> 另两个锚点数：$G_{\mathrm{off,mix0}} = 0.079$ vs null $2.93$（mixer 输出已
> sub-noise）；off 份额 $6.6\% \rightarrow 0.035\%$（两级各约 $14\times$）。

## 分析与结论

- **是否达成假设**：部分。形式判决落在预注册的 case 1，但触发分支是
  null-floor 而非级联骤降——即"case 1"的**字面读法（mixer 处死亡）是误导的**：
  mixer 确实透传 off-diag（$G_{\mathrm{off,mix0}} = 0.079$，off/diag
  $6.6\times10^{-2}$，与 08-13 测得的稠密 $W$ 定量一致），透传量只是**低于后验
  噪声底**。
- **对 08-10 的解释**："R below null" 不是解码端伪影——sub-noise 状态在 mixer
  输出处已成立；且对角高于 R null floor，说明信号链对角方向健康、唯 off 方向
  低于噪声。
- **case 排除与归位**：case 3 被排除（$\tau_{\mathrm{roll/vel0}} = 0.922$，
  且逐步表显示 stack 在主动淬火增长的 mixer 响应）；case 2 弱化——velocity
  stack 并非"忽略"（$G_{\mathrm{off,vel0}} = 5.9\times10^{-3}$ 非零）但确是
  第一个 $13.5\times$ 大衰减器；case 4（decoder 吸收）真实存在（off/diag 再压
  $30\times$）但携带单位混合 caveat。
- **几何假说**：扰动几何失配否证（mixer 处 data/ref $= 1.04$）；R 层
  $8\times$ 的 data/ref 是二阶现象。

## 诚实警示（Honest caveats）

- **(a) case 标签 vs null-floor 分支**：形式标签 `case1_dies_at_mixer_output`
  按预注册规则是对的（mix0 低于 $1.1\times$ floor），但**不得**读作"mixer 什么
  也不传"——mixer 透传量为真（off/diag $6.6\times10^{-2}$），且后续每一层同样
  低于各自 floor。正确表述：**off-diag 响应自 mixer 输出起 sub-noise**。
- **(b) 两个 near-threshold $\tau$**：$\tau_{\mathrm{off}}$(vel0/mix0)
  $= 0.0739$、R/roll $= 0.0718$，都悬在 $0.05$ 阈值之上——不把任何一个单独
  归为凶手；这是 08-13 差 $0.009$ 之后的第二个 near-threshold 判决。
- **(c) R/roll 单位混合**：latent 层的 $G$ 是 latent-L2 单位，R 是解码
  MSE/MAD$^2$ 单位，$\tau_{\mathrm{R/roll}}$ 混合两者；无单位的 off/diag 对比
  （$1.06\times10^{-2} \rightarrow 3.5\times10^{-4}$，$30\times$）才是更干净的
  表述。
- **(d) data 臂 R 层 $8\times$ 是绝对 sub-noise 的**：$3.47\times10^{-3}$ 仍
  低于 null floor $236\times$——二阶观察，不是 Arm-B 救援。
- **(e) 共享 GPU 墙钟**：主循环 11925 s（约 3.3 h，含编码）；rollout batch
  128 峰值约 5.4 GB（12 GB 共享卡，余量充足、稳定无 CUDA 故障——对比 08-13
  远臂 batch-128 的 4/4 故障）。
- **(f) 测试夹具 D=4 约束**：`tests/test_layerwise_audit.py` 的迷你模型用
  $D{=}4$（`SinusoidalTimeEmbedding` 在 $D \leq 2$ 时退化）。
- **(g) npz schema 是 design §8 的务实变体**：step-0 主读出独立成
  `g_mix0/g_vel0 (N', n_src, C, n_arm, M)`（对 K 平均）；逐步级联以通道归并的
  (diag, off) 对存储 `g_mix_step (N', n_src, n_steps, 2, n_arm, M)`（非 design
  的全维 `g_mix (N', n_src, C, n_steps, K, n_arm, M)`）；逐方向明细另存
  `g_dir_diag/g_dir_off (N', n_src, K, n_arm, M)`。形状已实载核验；这是数十 MB
  预算（design §7）下的裁剪。

## 可复现性

环境 `D:\program\anaconda\envs\npi\python.exe`，从 `D:\102-npi`（`latent_flow/`
的父目录）运行——直接调 env 的 python，不用 `conda run`（Windows gbk 崩溃）：

```bash
# Gate B（冒烟）
python -m latent_flow.scripts.run_layerwise_audit --fold 0 --smoke

# 注册运行（写 results/de/layerwise_audit/；共享 GPU 约 3.5 h 含编码）
python -m latent_flow.scripts.run_layerwise_audit --fold 0

# 全测试
python -m pytest latent_flow/tests/ -q     # 184 passed, 3 skipped
```

- 种子：de_seed=0（$\xi_k$、$\varepsilon$）、subsample seed=0、Null
  $\varepsilon'$ 取 seed+2；r_max 钉 1.3930（归档 ref stats）。
- **新增代码全部 additive**：`analysis/de/layerwise.py`、
  `scripts/run_layerwise_audit.py`、`tests/test_layerwise_audit.py`（13 个新
  测试）；`git diff -- models/` 为空（无重训、无模型改动）。
- Provenance 全量戳在 `verdict.json` `meta`（ckpt sha1 `fc722302e717`、git
  `876ce87`、ref_stats sha1 `1e67603dc722`、源/AP 坐标、bootstrap 参数）。

## 下一步（候选方向，非承诺）

- **velocity stack 解剖**：为什么逐通道 dilated-conv stack 对跨通道输入做
  $7\times$ 的相对抑制（off/diag $6.6\times10^{-2} \rightarrow 9.0\times10^{-3}$）
  ——逐 conv 层 / 逐通道谱位的衰减定位。
- **decoder off 份额吸收探针**：解码端 off/diag 再压 $30\times$ 的机理
  （per-channel FC 头是否天然吸收跨通道 latent 位移）。
- **把扰动 SNR 抬到 $\varepsilon$ 底之上**：更大的 $r$（超出 $q{=}0.99$ 支持
  域需谨慎）或 $\varepsilon$-冻结读出（固定后验噪声、纯 $\delta$ 因果效应的
  确定性下限），以判断 sub-noise 是"真无跨通道因果"还是"SNR 不足"。
- 已沉淀：本实验的 "case-1-via-null-floor" 判读契约（null-floor 分支 vs 级联骤降
  分支的语义区分）已写入 [[perturbation]] spec（§ Layerwise
  perturbation-transmission audit）。无新 checkpoint，无需 registry 登记。

## 产出物索引

| 类型 | 路径 |
|---|---|
| 判决（机器可读：级联 + 坍缩 + 几何子判决 + meta） | `results/de/layerwise_audit/verdict.json` |
| 摘要（人类可读：四层表 × 4 格、臂比表、逐步级联、AC5） | `results/de/layerwise_audit/summary.md` |
| 原始 G 数组（per-window G、逐步级联、null、逐方向明细、$\delta$ 范数/Maha、PC、$\xi$） | `results/de/layerwise_audit/layerwise_g.npz` |
| 运行日志（stdout 捕获，含 [layerwise] done in 11925.0s、r=0 EXACT ZERO、AC5 行） | `C:\Users\10618\.zcode\cli\exec\sess_20dcfcba-907f-44aa-802d-0250e914384d\call_623e0fdb807842ccbb4e7fcb-stdout.log` |
| 库模块（捕获 / G 计算 / Arm-B 构造 / 判决） | `analysis/de/layerwise.py` |
| 运行器 | `scripts/run_layerwise_audit.py` |
| 测试（13 个新测试） | `tests/test_layerwise_audit.py` |
| 冻结 substrate（只读） | `checkpoints/de_flow/fold0_de_flow_best.pt`、`results/de/fold0/ref_stats.npz` |

## 关联笔记

- 前序 / 门控依据：[[08-13-diagonal-r-mixer-diagnosis]]（mixer 容量正常 → 病因定位到透传层）
- 动机 / 被解释的发现：[[08-10-de-inference-prep-ijkl]]（diagonal-R、R below null at $r_{\max}$）
- Spec：[[perturbation]]（decision-13 共享随机性契约、本次建议增补 null-floor 判读）
- 关联 checkpoint：见 frontmatter `checkpoint`（只读）
