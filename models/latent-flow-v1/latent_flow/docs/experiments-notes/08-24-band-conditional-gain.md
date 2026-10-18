---
date: 2026-08-25
name: band-conditional-gain
aliases:
  - Exp 5b-lean
  - 实验五b-lean
  - alpha conditional gain
  - common-mode control
task: 08-24-band-conditional-gain
type: diagnosis (band-conditional gain + re-reference common-mode control)
dataset: modma
fold: 0
status: "完成 (verdict CB-REDUNDANCY-DEEP: alpha 判决格两臂双线均不过 -> band-split 路线冻结)"
tags:
  - experiment
  - diagnosis
  - cross-channel
  - conditional-gain
  - common-mode
  - re-reference
  - probe-matrix
  - modma
  - raw-domain
  - band-filter
  - result-negative
checkpoint: null   # 本实验不加载任何模型权重；冻结输入（raw pairs npz / FC_emp / 08-16 归档读数 / geometry）只读，sha1 戳于正文与 verdict.json
variant: null
related_spec: "[[feasibility-gate]]"
---

# Exp 5b-lean：alpha 条件增益 + 共模对照（band-conditional gain）

> Task `08-24-band-conditional-gain`。Exp 5（[[08-23-task-snr-scan]]）判
> BR-BAND，但其读数全为单源探针——「a 的历史 beyond b 自身历史」的
> **条件增益**从未在带格上测过；k8/对角恒定 0.74–0.80、U 形四分位、
> 弱 FC_emp 相关三条旁证提示共模冗余。本实验只在决策格（alpha@fc128）
> 与锚定格（bb@fc128）上补这一读出，并加 Hjorth 式邻均再参考对照臂，
> 为 band-split VAE（alpha@fc128）的 GPU 投入做最后资格判定。**本轮
> 明确不做 band×horizon 扫描：horizon 轴完全不涉及（固定 fc128），
> 除判决格与锚定格外不跑任何其他带格**（theta/beta 佐证格默认关、
> 实际未开；delta 不进任何判决）。纯 CPU、无训练、无扰动、`models/`
> 零 diff、既有注册产物只读、不 commit。判读留给讨论。

> [!summary] 一句话结论
> 预注册判决 **CB-REDUNDANCY-DEEP**——判决格 alpha@fc128 两臂双线均
> 不过：原信号臂聚合 CondGain **−0.281** [−0.336, −0.245]（CI_low 远
> 低于 +0.02 线）、PairGain 中位 +0.0095 [+0.0067, +0.0115]（小正但
> 单线不足以过）；再参考臂更弱（CondGain −0.469、PairGain +0.0005
> [−0.0013, +0.0021]，CI 含 0，唯一 near flag）→ 按冻结分支表
> **band-split 路线冻结；相位类读出（PLV/相位传递熵）升为决定性实验，
> 论文末章转「读出族边界」叙事**。三个支撑读数：(i) Exp 5 的绝对增益
> 主体确认**不是条件信息**（alpha 绝对 k8 +0.6536 逐位复现，但逐对
> 条件增益中位仅 +0.0095 ≈ 1% R²）；(ii) bb 格再参考后绝对单源读数
> 崩塌（off-diag/k8 → −0.007、U 形四分位打平）而宽带条件增益存活
> （CondGain +0.026→+0.045）——真跨通道条件信息在宽带、不在带格；
> (iii) alpha 格去共模后 PairGain 从 +0.0095 → ≈0——判决格仅有的那点
> 条件信息也主要是共模。AC 硬门 6 量（含对角锚，5b 新增）全部
> max|diff| = 0.000e+00 逐位复现；地板软校验 4 项 0 偏差。

## 实验目的与假设

Exp 5 判 BR-BAND 后 substrate 路线定为逐通道 band-split VAE（alpha
优先），但其全部读数是绝对单源 R²(x_obs,a → x_fc,b)：若带格读数的
增益主体来自「窄带载体可预测 + 容积传导共模」，而非「a 的历史对
b 的未来**在 b 自身历史之外**携带信息」，band-split VAE 的资格就不
成立。三条旁证（k8/对角比恒定 0.74–0.80、U 形四分位、|ρ(I_off,
FC_emp)| ≤ 0.066）指向共模冗余。5b-lean 用两个新读出（聚合条件增益
PCA-256、k8 逐对条件增益）× 两臂（原信号 / Hjorth 4 近邻再参考）在
2 格（bb、alpha）上裁决。

## 预注册块（frozen before the registered run；全在
`.trellis/tasks/08-24-band-conditional-gain/prd.md`）

### 单元格与臂（冻结）

- **bb@fc128**（恒等直通）：AC 锚定格 + 宽带参照（条件增益基线水平）；
- **alpha@fc128（8–13 Hz）**：判决格（Exp 5 中 off-diag 增益最大
  +0.27~+0.29、带内能量占比 41%、泄漏 3.8%、无边缘警示）；
- `--bands theta,beta` 佐证格默认关（本轮未开）；delta 不进任何判决；
- 再参考臂（--reref 默认开）：`x'_c = x_c − mean(4 个几何最近邻)`
  （`load_geometry("modma")` 坐标欧氏距离、自通道排除、稳定排序），
  **滤波前**作用于整窗（顺序冻结；空间常数线性组合与逐通道 LTI
> sosfiltfilt 数学上可交换，单测钉死，不依赖）；滤波规格逐字复用
  Exp 5（`task_snr` 直调，零复制）。

### 条件增益读出（每格每臂；within 主判 / cross 次要；被试层
bootstrap 1000；±0.005 纪律）

- a. 聚合：`CondGain_b = R²(PCA-256(全通道 obs) → x_fc,b) −
  R²(x_obs,b → x_fc,b)`（PCA fit 于 fit 集、确定性样本 Gram+eigh；
  α 网格 logspace(−2,5,14)；自通道 ridge 逐 b、128×128 Gram 批量解）；
- b. 逐对（k8 近邻对，最贴近 DE 扰动语义）：`PairGain_{a→b} =
  R²([x_obs,a; x_obs,b] → x_fc,b) − R²(x_obs,b → x_fc,b)`，8 审计源
  × 各自 k8 近目标 = 64 对，逐对选 α，聚合 = **中位** [CI]，报分布；
- c. 伴随（描述性）：两臂单源 I 矩阵（与 run_probe 同规范）的
  q0–q3 四分位剖面、Spearman(距离, R²_off)、Spearman(I_off, FC_emp)
  （FC_emp 用注册归档、**未再参考**，两臂同锚）。

### 分支（逐字冻结；判决格 = alpha@fc128；臂级过线 = CondGain
CI_low > +0.02 **且** PairGain 中位 CI_low > 0，双线同时）

- **CB-PASS** → band-split VAE（alpha@fc128）资格门通过，派发训练；
- **CB-REDUNDANCY-REVIVED** → 共模混淆坐实且可预处理修复 →
  再参考/Laplacian 进 pipeline，substrate 路线不变；
- **CB-REDUNDANCY-DEEP** → 带格跨通道结构为纯共模冗余 → band-split
  路线冻结；相位类读出（PLV/相位传递熵）升为决定性实验，论文末章转
  「读出族边界」叙事。

### 种子与常量（frozen）

probe 主种子 0（build_splits/boot 母 `default_rng(1)`/shuffle 母
`default_rng(2)`，within→cross 顺序消费，与 run_probe 一致 → 电池内
与臂间同重采样、配对）；floor 种子推导 1798679647/561807779（与
08-16/08-23 逐位同推导，日志确认）；条件增益 α 网格
logspace(−2,5,14)；I 矩阵探针 α 网格 DEFAULT_ALPHAS
logspace(−2,3,12)（可比性）；PCA 256；reref 4 近邻；near_tol 0.005；
AC 容差 1e-4；n_boot 1000 / n_shuffle 5（地板仅锚定格臂）；无 MLP。

### AC 硬门（先于其余格臂）

bb@fc128 **原信号臂**走**字面** `run_probe`（n_boot=1000）+
`shuffle_floor`（08-16 种子推导）复现 08-16 归档 6 量三分量（5b 新增
对角锚）：within off-diag +0.0261 [−0.0024,+0.0573]、k8 +0.0605
[+0.0204,+0.1003]、diag +0.0753 [+0.0296,+0.1187]；cross off-diag
+0.0169、k8 +0.0563、diag +0.0796。**实测 6 量全部 max|diff| =
0.000e+00**；地板软校验 4 项 0 偏差（无 WARN）。

## 设计与执行

- **库**：`analysis/fm_gate/band_cond_gain.py`（冻结常量/分支字典/
  再参考/快速 bootstrap/PCA-256/自通道批量 Gram/逐对模型/判决全为
  新增）；**runner**：`scripts/run_band_cond_gain.py`；机件复用
  import（`task_snr` 滤波与能量、`probe_matrix` ridge/splits/
  persistence/bootstrap、`probe_raw` 距离箱），零算法复制；
  `probe_matrix.py`/`task_snr.py`/`probe_raw.py` **零改动**。
- **快速 bootstrap（行加权和）**：字面 `r2_bootstrap_rows` 每
  resample 重取全 (n,T,D) 数组（内存带宽型）；新路径预算
  `row_p2/row_py/row_y2/row_y2mu` 后每 resample 只做权向量点积——
  代数等价（测试对齐 ≤1e-8，实测 ~1e-12），**AC 门量仍只走字面路径**
  （锚定臂 4672 s 字面 run_probe；其余臂探针 + 条件电池各 ~4 min）。
  附带交叉验证：alpha/orig 臂快速路径的单源读数与 08-23 注册值
  **逐位一致**（k8 +0.6536 [+0.6110,+0.6753]、off +0.3007
  [+0.2322,+0.3517]、diag +0.8160）——非门控但强一致。
- **测试**：`tests/test_band_cond_gain.py` 15 项（Hjorth 合成拓扑
  算例、近邻矩阵不变量、再参考-滤波交换律、自通道批量解==逐 b 路径、
  PCA 正交/降序/得分、CondGain/PairGain 手工算例（rank-3 因子：源
  独占因子 → CondGain>0 且源对>0/惰性对≈0；无源因子 → 均≈0）、
  快速 bootstrap==字面、分支全分支+双线+cross 不进分支、AC 门 6 量、
  臂隔离（orig 逐位==task_snr + 共模抑制算例）、格臂契约（delta
  拒绝/缺 bb 拒绝/chunk 无关）、run_cond_gain→判决端到端形状契约）。
  全量 pytest **299 passed, 3 skipped**（基线 284 + 新增 15，零回归）。
- **执行序列**：PRD 冻结 → 冒烟 ×2（alpha/orig、bb/reref，各 34 s，
  零产物）→ 注册运行 4 格臂。**过程警示（如实记录）**：首次注册运行
  在全部 4 格臂计算完成后、verdict 组装处因接口形状 bug 崩溃（runner
  把 run_cond_gain 的 split-major 输出直传判决函数；修复 =
  `decision_arm_from_cond` 适配 + 回归测试 #15）；确定性重跑全部数值
  逐位一致，注册产物来自完整的第二次运行（5701 s ≈ 95 min，锚定臂
  4672 s 占 82%——任务简报 20–30 min 预估被 PRD 修正为 ~1.5–2.5 h，
  实测落在此区间）。

## 结果

### 判决：CB-REDUNDANCY-DEEP（alpha 两臂双线均不过；AC 门逐位复现）

| 判据组 | 结果 |
|---|---|
| 原信号臂（alpha） | CondGain −0.2809 [−0.3360, −0.2446]（CI_low ≪ +0.02 ✗）；PairGain 中位 +0.0095 [+0.0067, +0.0115]（✓ 但需双线）→ **不过线** |
| 再参考臂（alpha） | CondGain −0.4692 [−0.5268, −0.4321] ✗；PairGain 中位 +0.0005 [−0.0013, +0.0021]（CI 含 0 ✗）→ **不过线** |
| near flag | `reref:pair_median_ci_low_vs_zero`（CI_low −0.0013 距 0 在 ±0.005 内，如实报原始值） |
| AC 硬门 | 6 量（含对角锚）max\|diff\| = **0.000e+00**；地板软校验 4 项 0 偏差 |
| cross 次要 | 全部同构偏弱（alpha/orig pair cross +0.0078 [+0.0064,+0.0096]；reref pair cross −0.0010），不进分支 |

### 头表：4 格臂全读数（within 主判；括号 cross）

| 格臂 | CondGain agg [CI] | PairGain 中位 [CI] (正对) | r²_pca / r²_self（均值） | 单源 off-diag / k8 / diag [CI] | q0–q3 |
|---|---|---|---|---|---|
| bb/orig | +0.0260 [+0.0177, +0.0338] (−0.008) | +0.0329 [+0.0279, +0.0403] 64/64 (+0.019) | +0.070 / +0.044 | +0.0261 / +0.0605 / +0.0753（=AC 锚） | +0.039/+0.015/+0.014/+0.037 |
| bb/reref | +0.0446 [+0.0302, +0.0573] (−0.043) | +0.0446 [+0.0369, +0.0534] 64/64 (+0.011, CI 含 0) | +0.036 / **−0.009** | **−0.0071 / −0.0070** [+0.0150..,−0.0024] / +0.0457 | −0.007 均匀（U 形打平） |
| alpha/orig | **−0.2809** [−0.3360, −0.2446] (−0.383) | +0.0095 [+0.0067, +0.0115] 64/64 (+0.0078) | +0.540 / +0.821 | +0.3007 / +0.6536 / +0.8160（=08-23 注册值逐位） | +0.432/+0.187/+0.189/+0.396 |
| alpha/reref | **−0.4692** [−0.5268, −0.4321] (−0.680) | +0.0005 [−0.0013, +0.0021] 42/64 (−0.0010) | +0.320 / +0.790 | +0.0420 / +0.0740 [+0.0361,+0.0974] / +0.7445 | +0.056/+0.039/+0.037/+0.036 |

（PCA 解释方差：bb/orig 0.906、bb/reref 0.695、alpha/orig 0.997、
alpha/reref 0.966；PCA 模型选中 α：bb 两臂 2.4e3、alpha 203/702——
单 α 全输出的重收缩是聚合 CondGain 负值的主因，见警示 (a)。）

### 伴随读数（描述性）

- **再参考臂的裁决性对比（bb 格）**：reref 后绝对单源读数全面崩塌
  （off-diag −0.0071 [−0.0150,−0.0024]、k8 −0.0070、四分位 U 形
  打平至 −0.007 均匀、对角锚 0.075→0.046）——**宽带的绝对跨通道
  读数几乎全部是共模**；但条件增益存活并升高（CondGain +0.026→
  +0.045、PairGain +0.033→+0.045，均 64/64 对为正；r²_self 均值
  +0.044→−0.009：自通道自身历史在去共模后失去预测力，邻域历史的
  条件信息占比反而上升）。
- **alpha 格去共模**：绝对读数同样崩塌（k8 +0.654→+0.074）；且
  PairGain +0.0095→+0.0005（CI 含 0）——**判决格仅有的条件信息也
  主要是共模**，带格条件层无幸存结构。
- Spearman(I_off, FC_emp) within：alpha/reref **+0.1226**
  [见 verdict]（cross +0.1027）——去共模后 I 模式与经验 FC 的相关
  升到 08-23 全格（|ρ|≤0.066）之上；bb/orig −0.017、bb/reref
  −0.048。Spearman(距离, R²)：alpha/reref −0.1433（距离梯度出现）。

## 分析与结论（事实性读法；判读留给讨论）

1. **判决函数按冻结规则输出 CB-REDUNDANCY-DEEP**：判决格两臂的
   双线判据均未满足（原信号臂败于 CondGain 线、再参考臂双败且
   PairGain CI 含 0）——分支表层级清晰，无 near 判定翻转（唯一
   near flag 是 reref PairGain CI_low −0.0013，其点值 +0.0005 同侧
   于 0 之下不远，无点-vs-margin 张力）。
2. **条件增益与绝对增益彻底解耦（本实验的核心读法）**：alpha 格
   绝对单源 k8 +0.6536（与 08-23 注册值逐位一致）对 PairGain 中位
   +0.0095——Exp 5 BR-BAND 的增益主体**不是**「a 的历史 beyond b
   自身历史」的条件信息，而是（i）带限让 b 自身可预测（对角锚
   0.816）+（ii）共模（去共模后绝对与逐对条件读数同灭）。5b 的
   三条旁证假设（k8/对角恒定、U 形、弱 FC 相关 ⇒ 共模）被再参考
   臂直接证实。
3. **真跨通道条件信息在宽带、不在带格**：bb 格（非判决格、仅作
   基线）CondGain +0.026 [+0.0177,+0.0338]、PairGain +0.033，reref
   后升至 +0.045/+0.045（64/64 对为正）——宽带全通道历史确实携带
   自通道之外的可计数信息，且它不是共模（reref 后仍在）。带限
   （alpha）没有放大这份条件信息，反而把它抹掉（PairGain →0）。
   注意 bb/orig 的 CondGain CI_low +0.0177 差 +0.0023 未过 +0.02
   形式线（若 bb 是判决格会被 near flag 标记）——bb 不是判决格，
   此读数只作基线水平陈述。
4. **对终止语境的含义（按冻结分支字典）**：band-split 路线冻结
   （不派发训练）；相位类读出（PLV/相位传递熵）升为决定性实验——
   线性幅值域的条件信息已在此读出族边界内测尽（bb 宽带有 ~0.03–
   0.05、带格无），相位域是唯一未测读出族；论文末章转「读出族
   边界」叙事。宽带 bb 格的条件增益存活（分析 3）为该叙事提供了
   正面锚点：「信息存在，但在幅值-线性-窗口族的可读边界内只有
   宽带级 ~0.03–0.05」。
5. **AC 逐位复现 + alpha/orig 单源读数与 08-23 注册值逐位一致**把
   本实验钉死在 08-16/08-23 冻结管线上；快速 bootstrap 路径与字面
   路径的一致性有单测（≤1e-8）与该实测双重背书。

## 诚实警示（Honest caveats）

- **(a) 聚合 CondGain 的构造敏感性（最重要）**：PCA-256 + 单 α
  服务 16384 输出，在自外推近乎完全的窄带格上结构性吃亏（alpha 格
  r²_self 均值 0.821 vs r²_pca 0.540；PCA 选中的 α 高达 203–702）。
  **CondGain < 0 不能读成「无跨通道信息」**——它是「压缩全通道历史
  是否胜过 b 自身历史」的读数；条件信息的判决重量在 PairGain（逐对、
  逐对选 α、无压缩），而 PairGain 在判决格两臂同样未过线（+0.0095 /
  +0.0005），这是 CB-REDUNDANCY-DEEP 的实质依据。两个读数的解耦
  必须并排陈述。
- **(b) PCA 基含 inner carve**（冻结于 brief，镜像 run_probe 的全集
  refit）：α 选择对 PCA 基轻微乐观；方向上只会高估 CondGain，不影响
  负值判决。
- **(c) FC_emp 锚未再参考**（两臂共用注册归档）：alpha/reref 的
  Spearman(I_off, FC_emp) +0.123 在混参考方案下只作方向性读数。
- **(d) reref 是简化 Hjorth**（4 等权几何近邻），非球样条
  Laplacian；若 REVIVED 触发本应再做更强预处理验证——本轮未触发，
  该限制只影响对 bb 格条件增益幅值（+0.045）的解读精度。
- **(e) 覆盖范围**：8 个 AP 轴代表源 × 128 目标、fold 0、固定
  fc128、恒定 obs 窗 128 样点、Butterworth-4、一次再参考实现；
  theta/beta 佐证格未开、delta 未跑、horizon 轴与全网格扫描**明确
  为本轮范围外**（需要时另立任务）。结论在此覆盖内成立。
- **(f) 过程警示**：首次注册运行在计算全部完成后于 verdict 组装处
  崩溃（接口形状 bug；修复 + 回归测试）；确定性重跑逐位一致，产物
  来自第二次完整运行。算力 5701 s ≈ 95 min（锚定臂字面复现 4672 s
  占 82%；任务简报 20–30 min 预估低估 ~3 倍，PRD 已在运行前修正为
  ~1.5–2.5 h）。
- **(g) PairGain 的量级语境**：+0.0095 R² 是「64/64 对一致为正、
  CI_low > 0」的稳健小效应——统计上真、效应量小；band-split VAE
  若以此为练料，预期可读信息上限即此量级，这正是资格门判「不过」
  的原因。

## 可复现性

环境 `D:\program\anaconda\envs\npi\python.exe`，从 `D:\102-npi`
（仓库父目录）运行——直接调 env python，禁 `conda run`（Windows gbk
崩溃），打印全 ASCII：

```bash
# 冒烟 ×2（单格单臂 × 少量被试，零产物）
D:/program/anaconda/envs/npi/python.exe -u -m latent_flow.scripts.run_band_cond_gain --fold 0 --smoke
D:/program/anaconda/envs/npi/python.exe -u -m latent_flow.scripts.run_band_cond_gain --fold 0 --smoke --smoke-cell bb --smoke-arm reref

# 注册运行（默认 bb,alpha × orig,reref = 4 格臂；实测 5701 s；写 results/de/band_cond_gain/）
D:/program/anaconda/envs/npi/python.exe -u -m latent_flow.scripts.run_band_cond_gain --fold 0

# 本任务测试（15 项）与全量（299 passed, 3 skipped = 基线 284 + 15）
D:/program/anaconda/envs/npi/python.exe -m pytest latent_flow/tests/test_band_cond_gain.py -q
D:/program/anaconda/envs/npi/python.exe -m pytest latent_flow/tests/ -q
```

- **Provenance**（全量戳在 `verdict.json` `meta`）：raw pairs sha1
  `a035e14c7923`（只读）；FC_emp `864777b6d6bc`；08-16 归档读数
  `d0be28f423e0`；geometry `476c21310727`；git rev `9efe1e3`；fs=250 Hz
  （注册表读取）；种子全表与阈值见预注册块（与 `meta.seeds` /
  `meta.thresholds` 一致）；AC 对照逐分量入 `ac_gate`。
- **新增代码全部 additive**：`analysis/fm_gate/band_cond_gain.py`、
  `scripts/run_band_cond_gain.py`、`tests/test_band_cond_gain.py`
  （15 测试）；既有模块零改动；本任务对 `models/` 零 diff；既有注册
  产物零改动；未 commit。

## 下一步（候选方向，非承诺）

- **判读讨论（最高优先）**：(a) CB-REDUNDANCY-DEEP 下相位类读出
  （PLV/相位传递熵）实验的预注册设计——它是冻结分支表指定的下一个
  决定性实验；(b) bb 宽带格条件增益的存活（+0.045，reref 后）是否
  值得单独立任务追读（非 band-split 语境的「宽带条件信道」）。
- **读出族边界叙事**：论文末章按分支字典转写；5b 的两臂对照表
  （绝对崩塌/条件存活）是该章的主证据图。
- 相位类读出若也 flat → 数据集/范式变更为唯一上游选项（BR-FLAT
  语境遗留项的最终检验）。

## 产出物索引

| 类型 | 路径 |
|---|---|
| 判决（branch + 两臂判据 + 6 量 AC 门 + 逐格臂全读数 + provenance） | `results/de/band_cond_gain/verdict.json` |
| 摘要（头表、两臂对照、伴随读数、判决、警示） | `results/de/band_cond_gain/summary.md` |
| 原始数组（逐格臂 R²/boot/α/L_persist/CondGain/PairGain/PCA/energy/masks/meta） | `results/de/band_cond_gain/band_cond_gain.npz` |
| 运行日志（stdout 捕获，含 AC 门逐分量行、逐格臂墙钟、判决行） | `results/de/band_cond_gain/run_registered.log` |
| 库模块 | `analysis/fm_gate/band_cond_gain.py` |
| 运行器 | `scripts/run_band_cond_gain.py` |
| 测试（15 项） | `tests/test_band_cond_gain.py` |
| 冻结输入（只读） | `results/de/probe_raw/fold0_raw_pairs.npz`、`results/de/probe_matrix/fc_emp.npz`、`results/de/probe_raw/probe_raw_r2.npz`、`data/modma/geometry.pt` |

## 关联笔记

- 协议母本与锚定来源：[[08-16-raw-domain-probe]]（AC 硬门逐位复现
  其归档读数与地板；探针机件原封复用）
- 上游判决：[[08-23-task-snr-scan]]（BR-BAND → 本实验为其资格门；
  alpha 绝对读数逐位复现其注册值）
- 审计链上游：[[08-15-substrate-cross-channel-probe]]（探针机件）、
  [[08-14-frozen-fm-layerwise-perturbation-audit]]（8 个 AP 轴审计源）
- Spec：[[feasibility-gate]]（gate 纪律与预注册判据）
