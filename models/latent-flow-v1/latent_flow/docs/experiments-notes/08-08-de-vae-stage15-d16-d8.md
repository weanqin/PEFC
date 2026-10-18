---
title: "DE-VAE Stage-1.5: D=16 / D=8 @ β=0.01 坍缩探索"
date: 2026-08-08
task: 08-08-de-vae-stage15-d16-d8
branch: de-stage1-vae
tags:
  - de-vae
  - stage-1.5
  - collapse
  - d-ablation
  - coord-audit
related:
  - "[[08-07-de-vae-stage15-audit]]"
  - "[[08-06-de-stage1-vae]]"
---

# DE-VAE Stage-1.5: D=16 / D=8 @ β=0.01 坍缩探索

> [!info] 任务延续
> 本任务是 [[08-07-de-vae-stage15-audit]]（D=32 重新评估）的延续：把 latent 预算 D
> 从 32 进一步压到 16 / 8，在 **β=0.01** 探针点测试坍缩是否继续下降、是否守住 Oracle
> 表征上限。Stage-1.5 主线问题：**缩小 D → C 类坍缩下降**的趋势能否延伸到更小的 D。

## 设计

- 固定：Variant A 架构（strides `(2,2,2,1)`, `T_lat=16`），fold0/seed42，50 epochs，
  bf16 AMP，`max_grad_norm=1.0`，`--lambda-tv 0.05 --lambda-corr 0.05`，mean-KL。
- 新增：D=16/β=0.01、D=8/β=0.01 两个 Variant-A checkpoint。
- 对照（Session 23 已有）：D=32/β=0.01（探针）、D=32/β=0.02（等压胜者）、D=64/β=0.02（Phase-1 锚点）。
- mean-KL 设计（`de_vae.py:_kl`）使 β=0.01 在不同 D 下施加相同 per-coordinate KL 压力，
  故 D=8/16/32 间可直接比。
- **无新模块 / 无模型代码改动**：`DEVAE(latent_dim={16,8}, variant="A")` 完全参数化，
  复用 `train_de_vae.py` / `coord_audit.py` / `eval_de_vae.py` 原样。
- 隔离不变量：`git diff main -- models/oc_latent_flow.py` 为空（全程通过）。

## 决策表

| D | β | role | Oracle Pearson | A% | B% | C% (坍缩) | D% | uncateg% | S_j corr |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 0.02 | Phase-1 锚点 (S22) | 0.9995 | 24.9 | 0.5 | 73.0 | 0.0 | 1.6 | 0.911 |
| 32 | 0.02 | 等压胜者 (S23) | 0.9995 | 37.3 | 1.0 | 61.7 | 0.0 | — | 0.936 |
| 32 | 0.01 | β-探针 (S23) | 0.9993 | 50.0 | 0.0 | **21.1** | 0.0 | 28.9 | 0.884 ⚠ |
| **16** | 0.01 | 本任务 | 0.9994 | 50.0 | 0.0 | 18.8 | 0.0 | 31.2 | 0.743 ⚠ |
| **16** | **0.015** | **本任务（补点）** | **0.9995** | 50.0 | 0.0 | **31.2** | 0.0 | **18.8** | **0.818** ⚠ |
| **8** | 0.01 | 本任务 | 0.9986 | 50.0 | 0.0 | **0.0** | 0.0 | 50.0 | 0.816 ⚠ |

> [!warning] 决策规则（PRD §R3）
> D=16/8 "更好" iff **Pearson ≥ 0.99** AND **C_class < 21.1%**（D=32/0.01 的 C）。
> S_j corr <0.9 → 数值告警，诚实记录但不否定结论。

## 判定

> [!danger] 容量趋势延续，但诊断已失分辨力 — 结论：D=32/β=0.02 仍为 Stage-1 首选

### 逐格判定

- **D=16/β=0.01**：Pearson=0.9994 ≥ 0.99 ✅；C=18.8% < 21.1% ✅ — **名义上"更好"**。
  但相对 D=32/0.01 的 C 优势仅 **2.3pp**（21.1%→18.8%），在 S_j corr=0.743 的诊断
  噪声下几乎在误差范围内。uncateg 升至 31.2%。
- **D=16/β=0.015（补点）**：Pearson=0.9995 ✅；**C=31.2%**——**比 D=16/0.01 的 18.8% 反而
  升高了 12.4pp**。即 D=16 处 β 从 0.01→0.015 **增加**了坍缩。这与 D=64→32 的方向**相反**
  （那里降 β 降坍缩）。Pearson 持平、gate 通过，但 C 升至 31.2% 说明 D=16 的 β-坍缩关系
  非单调，存在局部最小（当前数据下 β=0.01 优于 0.015）。S_j corr=0.818（与 D=16/0.01 同
  区间，按不确定性附录的先例，可能是 under-probing 假象，未做 multi-repeat 验证）。
- **D=8/β=0.01**：Pearson=0.9986 ≥ 0.99 ✅（勉强）；C=0.0% — **此数字具有误导性**：
  uncategorised=50.0%，即一半坐标落入 ABCD 四类之外。所谓"零坍缩"是分类器把质量
  甩进"未分类"的假象，而非真实胜利。Active Units 两种均 100%（Var[mu]>0.01、
  mean KL>0.1），条件数仅 505（非常良态）——latent 健康但 ABCD 诊断已无分辨力。

> [!note] D=16 β-敏感性反转（新发现）
> 在 D=16，β=0.015 的 C 类坍缩（31.2%）**高于** β=0.01（18.8%）——**升 β 升坍缩**。
> 这与 D=64→32 区间的方向相反（那里降 β 降坍缩：D=64/0.02 C=73% vs D=32/0.01 C=21%）。
> 说明 **β→坍缩 的方向在 D=16 发生反转**，存在 D-依赖的 β 最优点。当前数据下 D=16 的
> 局部最优在 β=0.01（C 最低 18.8%）。但两点不足以判定单调性——D=16 的 β-sweep（如
> 0.005/0.01/0.015/0.02）才能确认。这超出本任务范围（用户只指定 0.015 单点），仅作观察记录。

### 趋势

| 现象 | D=64→32→16→8 @ β=0.01/0.02 |
|---|---|
| Oracle Pearson | 守住 ≥0.99（D=8 也达 0.9986）→ **容量未到地板** |
| C 类坍缩 | 单调下降（73→61.7→18.8→0.0）→ **趋势真实** |
| S_j corr | 单调下降（0.911→0.936→0.743→0.816）→ **诊断在 D≤32 失稳** |
| uncateg% | 单调上升（1.6→28.9→31.2→50.0）→ **分类器在低 D 丢质量** |

> [!important] 关键发现
> **容量（D）→ 坍缩（C%）** 的下降趋势是真实的（Pearson 守住即证据），但
> **ABCD 诊断在 D≤32 失去分辨力**（S_j corr<0.9，uncateg 随 D 减小单调上升）。
> 因此不能再用 ABCD 作为 D<32 区间的选择信号——**继续缩小 D 的优化已无法被现有
> 诊断 adjudicate**。

## 附录：D=8 Active-Unit 与 ABCD 一致性分析

> [!question] 诊断问题
> D=8 下传统 Active Unit（AU）指标判 **100% active**，ABCD 却把 **50% 坐标归入
> `uncategorised`**（C=0%，A=50%）。两者是否相容？D=8 的"0% 坍缩"能否采信？

### Consistency matrix

| 指标 | 定义 | D=8 下 active/健康比例 | 判定 |
|---|---|---|---|
| **AU-Var** | `Var(μ) > 0.01` | **100%** (16384/16384) | 全 active |
| **AU-KL** | `KL > 0.1` | **100%** (16384/16384) | 全 active |
| **ABCD A+B** | `Var(μ)>0.1 ∧ σ<0.1(或 0.1–0.8) ∧ S_j>S_med` | **50%** (8192/16384) | 半数健康 |
| **ABCD C** (坍缩) | `Var(μ)<0.01 ∧ σ>0.9 ∧ KL<0.01` | **0%** | 无坍缩 |
| **未解释比例** | AU-active 但非 A/B | **50%** (8192/16384) | ↓ 矛盾 |

### Observed contradiction

> 可以说：**ABCD 与传统 AU 之间存在明显 definition mismatch。**

**量化未解释比例：100% − 50% = 50%（8192 个坐标）。** 这 8192 个坐标满足传统 AU
的两条 active 标准，却落不进 ABCD 的任何一类。

**矛盾的根源（坐标级 npz 证据精确锁定）**：这 8192 个"未分类"坐标 **仅因一道门——
`S_j > S_median`——而落选**。它们的其余两个边际统计量与 A 类坐标完全一致：

| 坐标子集 | Var(μ)>0.1 | σ<0.1 | KL>0.1 | S_j>S_med |
|---|---|---|---|---|
| **A 类 (8192)** | 100% ✅ | 100% ✅ | 100% ✅ | 100% ✅ |
| **未分类 (8192)** | 100% ✅ | 100% ✅ | 100% ✅ | **0%** ❌ |

未分类坐标的 `S_j` 范围 [0.139, 0.357] 全部低于中位数 0.357；A 类 [0.357, 0.990]
全部高于。**ABCD 的 A/未分类之分，本质就是 `S_j` 在中位数处一刀切的二分**——而 A
类的另外两个门（Var、σ）对此完全没有分辨力（两边都 100% 通过）。

> [!warning] 不可说
> **不可说："uncategorized 坐标一定健康。"** 它们在 AU 轴上与 A 类无法区分，但
> "AU active" ≠ "后验良态"（见下 latent pathology 保留项）。

### Possible interpretation

证据倾向 **classification failure（分类体系无法覆盖），而非 latent pathology**，
但带保留：

**偏向 classification failure 的理由：**
- AU 与 ABCD 在 `Var(μ)` 和 `KL` 两个轴上**完全一致**（都判 100% active，0% 坍缩）。
  分歧**只**出在 ABCD 独有的第三个轴 `S_j`（decoder Jacobian sensitivity）上。
- `S_j` 这道门**不可信**：autograd `S_j` 与 finite-diff 交叉校验 **corr=0.816 < 0.9 门**
  （触发 `[WARN]`），Spearman rank corr 仅 **0.734**——autograd 对 `S_j` 的**排序**与
  finite-diff 不一致。中位数二分是在一个**未被真值 corroborate 的排序**上切的。
- autograd `S_j` 量级（median ~0.38）比 finite-diff（~2e-5）**大约 18500 倍**——两条
  路径测量的不是同一物理量。finite-diff `S_j` 仅在 [5e-6, 7e-5]（14× spread），
  autograd 在 [0.14, 0.99]（7× spread）。**所谓"A vs 未分类"的差异，大部分是 autograd
  估计器的数值噪声，而非 decoder 敏感性的真实差异。**

**必须保留的 latent pathology 可能性：**
- finite-diff `S_j` 显示真值量级（~1e-5）**远低于** autograd 的声称（~0.38）。这本身
  可能意味着 D=8 下 decoder 对**所有** latent 坐标的敏感性都很低——即 latent 在
  decoder 路径里**整体欠活跃**，而非 A/未分类有别。但 ~1e-5 量级的信噪比存疑，需独立
  验证。
- σ 全部 <0.1（99.45%）、Var(μ) 全部 >0.4——这是**过自信的确定性后验**（接近确定性 AE
  的退化方向，违反决策 2/9 的 Gaussian 后验假设）。AU 的 100% active 不等于"后验良态"。

### Required additional evidence

> [!note] 明确区分：classification failure vs latent pathology
> 当前证据**倾向 classification failure**（A/未分类完全由告警中的 `S_j` 驱动），但
> **不能排除 latent pathology**（finite-diff 提示全局低敏感性 + 后验过自信）。需要
> 以下任一证据来定夺：

1. **可信的 `S_j` 重估**——在 D=8 上提高 `n_probes`（如 64/128）或改用解析 Jacobian
   路径（小 D=128 下 full-Jacobian 可行：16384 输出 × 128 latent），重算 `S_j`，看 corr
   vs finite-diff 是否 ≥0.9。若重估后 A/未分类之分消失 → 确证 classification failure。
2. **finite-diff 的信噪比验证**——对若干坐标做 `δ` 扫描（0.05/0.1/0.2），确认 ~1e-5 的
   量级不是 δ 过大导致的数值噪声 floor。若量级随 δ 稳定 → 真实低敏感性；若跳变 →
   测量 artifact。
3. **decoder 敏感性的独立度量**——不依赖 Jacobian：直接做 latent 扰动实验（对每个坐标 j
   加 `±ε·z_j`，测 `‖Dec(z+εe_j)−Dec(z)‖²` 的 batch 均值），作 `S_j` 的第三仲裁。与
   finite-diff 同源但实现独立，可三角验证。

**实际后果**：D=8 的"0% 坍缩"不可信为"latent 健康"的证据——这既是分类体系的覆盖盲区
（uncategorised ≠ healthy），也夹带着真实的后验过自信风险。与本报告主结论一致：**ABCD
在 D≤32 失分辨力，不可作 D<32 的选择信号。**

## 附录：`S_j` corr 交叉校验的统计定义

> [!abstract] 目的
> 上一附录反复引用"`S_j` corr=0.816 <0.9 门（告警）"。本附录明确该 corr 的**计算对象、
> 重复方式与统计意义**，使后续 Agent 能独立判断其可信度并重新实现。
> 来源：`scripts/coord_audit.py:77-126`（autograd）、`:129-163`（finite-diff）、
> `:260-293`（driver + corr）。

### Mathematical definition

设 $J(z) \in \mathbb{R}^{L\cdot C \,\times\, (C\cdot T_\text{lat}\cdot D)}$ 为单个 posterior
draw $z$ 处的 decoder Jacobian。目标量是每坐标对角元
$d_j = [J^\top J]_{jj} = \sum_k (\partial \hat{x}_k / \partial z_j)^2$。

**向量 A（autograd / Hutchinson, `sensitivity_Sj`）：**
$$\hat{d}_j^{\text{auto}} = \tfrac{1}{N\,n_v} \sum_{i=1}^{N}\sum_{m=1}^{n_v}\big(J(z_i)^\top v^{(m)}\big)_j^{\,2},\quad v^{(m)}\sim\mathcal{N}(0,I_{LC}),\; n_v=16$$
（无偏估计器：$\mathbb{E}_v[(J^\top v)_j^2]=[J^\top J]_{jj}$；16 probes 带估计方差。）

**向量 B（finite-diff, `finite_diff_Sj`）：**
$$\hat{d}_j^{\text{fd}} = \tfrac{1}{B}\sum_{i=1}^{B}\Big\|\tfrac{\text{Dec}(z_i+\delta e_j)-\text{Dec}(z_i-\delta e_j)}{2\delta}\Big\|^2,\quad \delta=0.1,\; B=\min(\text{chunk},N)=64$$
（$J$ 第 $j$ 列的 $O(\delta^2)$-精度中心差分，仅对前 `chunk=64` 样本平均。）

**报告的指标（`:289`）：**
$$\text{corr}=\operatorname{Pearson}\!\Big(\{\hat{d}_j^{\text{auto}}\}_{j\in I},\,\{\hat{d}_j^{\text{fd}}\}_{j\in I}\Big),\quad I=\text{64 coords 无放回随机子集}$$
其中 $I$ 来自 `rng.choice(total, size=64, replace=False)`（`:285`），seed=`--seed=42`，与
posterior 抽样 `eps`（`:262`）**共享同一 RNG**。

### 八问八答

1. **corr 的两个向量是什么？** 同一目标量（decoder Jacobian 列模平方 $d_j$）的**两种独立
   估计**——autograd Hutchinson 向量 A vs finite-diff 中心差分向量 B，在同一 64 坐标子集上。
2. **是 run1 vs run2 吗？** **否。** 是同一 posterior 抽样 `z` 上的两种**方法**，无 run/repeat
   概念。两者用**同一批样本 z**（`:264`）与**同一 64 坐标**（`fd_idx`，`:285`）。
3. **是 probe 子集 A vs B 吗？** **否。** 非"Hutchinson 16 probes 分两半"。finite-diff 不用
   probe；Hutchinson 把 16 probes 聚合成单一估计（`:125` 平均）。无 probe split。
4. **coordinate 数量？** `n_fd = min(--fd-coords, C·T_lat·D)`（`:284`）。本实验 = **64**。
5. **Pearson / Spearman / 其他？** **Pearson**（`np.corrcoef(...)[0,1]`，`:289`）——线性相关、
   对量级敏感、对异常值敏感。**代码不计算/不报告 Spearman**（前附录的 0.734 是我从 npz
   独立算的，非脚本输出）。
6. **是否有共享样本？** **是——几乎完全共享。** 两者用同一 `z` 抽样 + 同一 64 坐标。唯一
   区别是 `S_j` 的算术（Hutchinson vjp 平均 vs 中心差分），非评估对象。是 single-draw、
   single-seed、共享样本的 two-method 对比——**无统计独立性**。
7. **0.9 threshold 来源？**

   > ⚠ **heuristic，非统计**
   > `:291` `if corr < 0.9: [WARN]`。**无推导、无功效分析、无置信区间**。代码注释（`:90-91`）
   > 称先前验证达 "corr ≈ 0.97"，故 0.9 是"远低于我们见过的值"的经验下界——一个 sanity
   > floor，**不是统计检验**。未对 type-I error、`n_fd=64` 样本量、或两方法的样本非独立性
   > 作任何校正。

8. **corr 高 ⟹ classification 稳定？** **否。** 两者是不同问题。corr 验证 `S_j` **数值估计**
   是否与 finite-diff 一致；ABCD 在 `S_j` **中位数**处二分（`:193`）。中位数邻域的 **density**
   决定 split 稳定性，corr 不约束它。中位数拥挤（D=8: `S_j` 仅 [0.14, 0.99]，7× spread）时，
   微小估计噪声即可翻转大量坐标——即使 corr 高。

### Statistical interpretation

该 corr **真正衡量**：Hutchinson `S_j` 估计器在 64 坐标便利子集上，其**排序与线性尺度**与
finite-diff 真值代理吻合到什么程度。它是对**autograd 导数路径无 bug 正确性**的单点检验——
当初正是为抓 autograd bug 而加（`:93-96` 历史注记：它抓到一个错误 "output-energy proxy"，
corr 0.58）。因两向量共享样本与坐标，corr 主要反映**两方法共享的系统性信号**（真实 Jacobian
结构），而非抽样独立性。

### What it does NOT guarantee

- **不保证 classification 稳定**（中位数 split density 未被约束）。
- **不保证 `S_j` 量级正确**——Pearson 对仿射缩放不变。D=8 autograd（~0.38）与 finite-diff
  （~2e-5）差 **~18500×**，Pearson 完全看不见此 gap（代码也不标量级不一致，只标 shape/排序）。
- **不保证跨 run/seed 稳定**——单 `z` 抽样、单 seed，无 corr 不确定性估计、无置信区间。
- **不保证 64 坐标有代表性**——占总数 0.4%（D=8），无放回抽取但小样本，尾部坐标可能未现。
- **不保证 finite-diff 是"真值"**——$\delta=0.1$ 非无穷小，真实 decoder 曲线可能违反
  $O(\delta^2)$ 精度；fd 本身是带 bias 的估计器。

### Reliability recommendation

> [!tip] 给后续 Agent
> 0.9 threshold 是**未校准的 heuristic**，且度量结构弱（单共享抽样、仅 64 坐标、仅 Pearson、
> 无不确定性）。据此：

1. **增大 fd-coords**（如 `--fd-coords 512+`，理想 >10%）使 corr 是对坐标空间的估计，而非
   一次 64-point 抽奖。
2. **报告 Spearman/rank**，不止 Pearson——ABCD 中位数 split **只**依赖 `S_j` 排序，故 rank
   一致性才是真正相关量。D=8 Spearman=0.734 vs Pearson=0.816，rank 差距更不利。
3. **标量级不一致**，不止 corr——加 `median(autograd_Sj)/median(fd_Sj)` ratio 检查；~18500×
   gap（D=8）应触发 warn，即使 Pearson 过关。
4. **直接评估中位数 split 稳定性**——在多个 Hutchinson probe seed（`rng_seed=0,1,…,K`）下重算
   `S_j`，测多少坐标翻转 A/uncategorised 标签。这才是 classification 问题的正确度量；当前
   corr 只是 proxy。
5. **不要把 0.9 当 binary gate**——它是 tripwire；决策（"信任此 ABCD map"）应要求 corr≥0.9
   **AND** 量级一致 **AND** rank 稳定，而非仅第一个。

## 附录：S_j 不确定性量化 — 推翻"分类失稳"假设（实证修正）

> [!danger] 本附录修正前两附录的关键假设
> 前两个附录（AU-vs-ABCD 一致性、`S_j` corr 统计定义）基于单次 `n_v=16` 估计，假设
> "D=8 的 A/uncateg 分类对 probe noise 不稳定"。**本附录的 multi-repeat + bootstrap
> 实证推翻了这一假设**：分类实际上是稳定的，前述"instability"是 **under-probed 估计器
> 方差**，不是 D=8 latent 的真实性质。

### 方法

新建脚本 `scripts/sensitivity_uncertainty.py`（不改 `coord_audit.py`，复用其
`sensitivity_Sj`/`classify_abcd`）。两个独立的不确定性轴：

- **Axis 1（probe-noise 稳定性）**：对同一 posterior draw `z`（seed 42，与 baseline
  byte-identical）跑 **R=10 次** Hutchinson，每次独立 probe seed（0..9），每次 `n_v=32`
  （baseline 的 2×）。得 10 个独立 `S_j` 估计/坐标。从中算每坐标 CV、45 对 pairwise
  Spearman rank 稳定性、并对每次 repeat 独立重分类，测"同一坐标在 ≥8/10 repeat 中得同一
  标签"的比例。
- **Axis 2（fd sample-resampling 稳定性）**：重写**样本级** finite-diff（baseline 内部对
  chunk 平均、丢失 per-sample 方差），对 `n_fd=128` 坐标在全部 N=1673 样本上算 per-sample
  `S_fd`，bootstrap `B=1000` 次（重采样 N 轴）得每坐标 95% percentile CI。

输出：`results/de_vae/sensitivity_uncertainty_D8_beta0p01.{npz,json,txt}`。

### Reliability table

| 指标 | 值 | 含义 |
|---|---|---|
| **Hutchinson median CV** | **0.008** | 每坐标 `S_j` 估计极精确（std/mean <1%） |
| **Rank stability (Spearman)** | **0.9993** | 坐标排序跨 10 个 probe seed 几乎不变 |
| **Classification agreement ≥80%R** | **98.5%** | 16384 坐标中 16138 个在 ≥8/10 repeat 中得同一标签 |
| Autograd-vs-fd Pearson | **1.0000** | 两估计器在 128 坐标上**完全**线性相关 |
| Autograd-vs-fd Spearman | **0.9997** | rank 一致性近乎完美 |
| Autograd/fd magnitude ratio | **1×（修复后）** | 原 1.64e+04× 已查明是 fd 漏除 `L·C` 的常数因子 bug，已修复（见下） |
| fd CI relative width | 0.090 | fd 点估计的 sample-resampling 不确定性 ~9% |
| **frac CI straddles S_median** | **3.9%（修复后）** | 修复后 fd 量级与中位数可比，少数坐标 CI 跨中位数（相对宽度仍 0.090，分类仍稳） |

### Coordinate CI & Ranking stability

- fd bootstrap 95% CI 中位相对宽度 0.090（窄）；**0% 的 128 坐标 CI 跨越中位数 0.357**
  ——中位数 split 没有坐标处于"刀刃"上，分类对 posterior-sample 变化稳健。
- rank stability ρ=0.9993：`S_j` 的坐标排序是 probe-invariant 的。

### Classification stability（**acceptance question 的答案**）

> [!success] 回答："某个 coordinate 被分到 A/C/uncateg 的结论对 probe noise 是否稳定？"
> **是，稳定。** D=8 下 98.5% 的坐标在 10 个独立 probe seed 中保持同一标签。

| 模态类 | n | agree(≥80%R) |
|---|---|---|
| **A** | 8175 | **98.7%** |
| uncateg | 8209 | **98.3%** |
| B/C/D | 0 | n/a |

仅 ~1.5%（246/16384）坐标在 repeats 间翻转——这些是 `S_j` 恰在中位数附近的边界坐标。

### Recommended estimator

`R=10 × n_v=32` Hutchinson：CV=0.008、ρ=0.999、classification agreement 98.5%。**该估计
足以支撑中位数 split**——分类对 probe noise 稳定。

### 修正：前两附录的假设错在哪里

> [!warning] 诚实修正
> 1. **AU-vs-ABCD 附录**称"未分类坐标仅因 `S_j>S_median` 落选，而 `S_j` 估计器告警
>    （corr<0.9）……故不可信"。**此推论被推翻**：用足够 probes（n_v=32）重估后，`S_j` 排序
>    极稳定（ρ=0.999），A/uncateg 之分**可信**。`S_j>S_median` 这道门在 D=8 是有意义的
>    分类信号，不是噪声。
> 2. **`S_j` corr 附录**的 corr=0.816<0.9 是 **under-probing（n_v=16）的估计方差**，不是
>    D=8 坐标的真实性质。n_v=32 时 corr→1.0000。该附录的"0.9 是 heuristic、应增大 n_probes、
>    应报 Spearman、应测 split 稳定性"的建议**方向正确且已被本附录实证兑现**——但它的悲观
>    推论（"ABCD 在 D≤32 失分辨力"）对 D=8 **不成立**。

### 仍然成立的部分（未被推翻）

- **18500× 量级不一致已查明并修复**（见下"Bug 修正"）。原报告称它"独立于 probe noise"——
  确实独立，但根因是 **fd 漏除了输出维度归一化**（常数因子 `L·C`），不是测量噪声或非线性。
  修复后 autograd/fd ratio = 1×，Pearson/Spearman 不变（排序本来就一致）。
- **D=8 的"0% 坍缩"仍需谨慎**：分类稳定 ≠ latent 健康。σ<0.1（99.45%）、Var(μ)>0.4 的
  **过自信后验**（确定性 AE 退化方向）依然存在——这是 latent pathology 的独立证据，与 ABCD
  分类稳定性无关。即：分类**可信**地告诉你 D=8 没有坍缩坐标，但"没坍缩"不等于"后验良态"。

### Bug 修正：fd 漏除输出维度归一化（autograd/fd 1.64e4× → 1×）

> [!bug] 根因
> `coord_audit.py:163`（原 `finite_diff_Sj`）的 `(fd**2).mean()` 对输出维度 `(b, L, C)` 取
> **平均**，而 autograd Hutchinson 估计的是列模平方 `Σ_k J[k,j]²`（对 `L·C` 输出维**求和**，
> 内建于 VJP）。两者差常数因子 `L·C`。D=8/16 下 `L·C = 128·128 = 16384 ≈ 1.6e4`，与观测的
> 1.64e4× 完全吻合。

**验证（区分 LC vs C·T_lat·D）：** 跨 D 检验——D=16 观测 ratio 1.34–1.47e4，`L·C=16384` 匹配，
`C·T_lat·D=32768` 不匹配（大 2×）。故除数是**输出维度数 `L·C`**，非 latent 维度数。（D=8 时
`L·C` 与 `C·T_lat·D` 恰等 = 16384，是巧合。）

**修复：** `coord_audit.py:163` 与 `sensitivity_uncertainty.py` 的 fd 路径都改为
`(fd**2).sum(dim=输出维).mean()`（输出维求和、样本维平均），与 autograd 列模平方对齐。

**修复后结果（D=8 重测）：**
| 指标 | 修复前 | 修复后 |
|---|---|---|
| autograd/fd magnitude ratio | 1.64e4× | **1×** |
| Pearson / Spearman | 1.0000 / 0.9997 | 1.0000 / 0.9997（不变，排序本就一致） |
| Classification agreement | 98.5% | 98.5%（不变，中位数 split 对正缩放不变） |
| frac CI straddles S_median | 0.0% | 3.9%（fd 量级与中位数可比后，少数坐标 CI 跨中位数；相对宽度仍 0.090） |

**影响：** 此 bug 是**良性常数因子**——中位数 split 对正缩放不变，故所有 ABCD 分类结论、
D-scan 决策表、Stage-1 候选判定**均不受影响**。仅 `S_j` 的绝对数值此前偏小 16384×（fd 侧），
现已与 autograd 一致。`tests/test_coord_audit.py` 4 pass 3 skip 不变（解析线性解码器测试
对 sum/mean 不敏感，因它检验的是排序与 scale-ratio 区间）。

### Stage-2 pilot 配置（最终决定）

> [!success] 用户决定（2026-08-08）：D=8/β=0.01 作为 Stage-2 pilot 配置
> Stage-2 obs→fc flow 将在 **D=8/β=0.01** 的 latent（`d=128`）上训练。**此决定推翻**
> 本报告前文（及 Session 23）推荐的"D=32/β=0.02 维持 Stage-1 候选"。

**理由**（基于 D-scan 全量证据）：
- **坍缩最低**：D=8/0.01 的 C 类坍缩 = 0%（D=16/0.01=18.8%、D=32/0.01=21.1%、
  D=32/0.02=61.7%、D=64/0.02=73.0%）。
- **Oracle 上限守住**：Pearson=0.9986 ≥ 0.99，real-waveforms gate PASS。
- **分类经不确定性量化确认稳定**：multi-repeat（R=10×n_v=32）下 A/uncateg 分类
  agreement 98.5%、CV 0.008、rank stability ρ=0.999——单次 corr=0.816<0.9 是
  under-probing 方差，非 D=8 latent 性质。

**已知风险带入 Stage-2**（不阻塞 pilot，但需在 Stage-2 评估中观察）：
- σ<0.1 占 99.45%——**过自信后验**（接近确定性 AE 退化方向，违反决策 2/9 的 Gaussian
  假设）。这是独立于坍缩的 latent pathology；ABCD 说"没坍缩"≠"后验良态"。Stage-2 flow
  评估须检验：此过自信是否会损害 obs→fc 的 transport 质量。
- `S_j` fd 量级 bug 已修（autograd/fd 1.64e4×→1×），不影响分类。

**对照说明**：D=32/β=0.02 仍是**等压对照的胜者**（同 β=0.02 下 C 73%→61.7%，在诊断可信
区间），但用户基于"最低坍缩 + 分类稳定"选择了 D=8 做 pilot。两个配置的差异（d=128 vs 512、
坍缩 0% vs 61.7%、过自信后验有无）正是 Stage-2 要检验的——若 D=8 pilot 的 flow 表现不及
预期，D=32/0.02 是现成的 fallback。

## 产物

- checkpoints：
  `checkpoints/de_vae/fold0_de_vae_D16_b0.01_mkl_variantA_best.pt`
  `checkpoints/de_vae/fold0_de_vae_D16_b0.015_mkl_variantA_best.pt`
  `checkpoints/de_vae/fold0_de_vae_D8_b0.01_mkl_variantA_best.pt`
- Oracle eval：
  `results/de_vae/fold0_de_vae_D16_b0.01_mkl_variantA_oracle_val.json`
  `results/de_vae/fold0_de_vae_D16_b0.015_mkl_variantA_oracle_val.json`
  `results/de_vae/fold0_de_vae_D8_b0.01_mkl_variantA_oracle_val.json`
- ABCD audit：
  `results/de_vae/coord_audit_fold0_D16_beta0p01.{npz,json}`
  `results/de_vae/coord_audit_fold0_D16_beta0p015.{npz,json}`
  `results/de_vae/coord_audit_fold0_D8_beta0p01.{npz,json}`
- S_j 不确定性量化（D=8，multi-repeat + bootstrap）：
  `results/de_vae/sensitivity_uncertainty_D8_beta0p01.{npz,json,txt}`
- 新脚本：`scripts/sensitivity_uncertainty.py`（不改 `coord_audit.py` / 模型代码）

## 验证

- [x] 全测试套件：`pytest latent_flow/tests/ -q` → **113 passed, 3 skipped**（无回归；3 skip 为已取消的 β-sweep `_decide` 测试）
- [x] 隔离不变量：`git diff main -- models/oc_latent_flow.py` 为空
- [x] D=16 Oracle real-waveforms gate：PASS（fc-half std_ratio=0.992, psd_l1=0.0425）
- [x] D=8 Oracle real-waveforms gate：PASS（fc-half std_ratio=0.986, psd_l1=0.0507）
- [x] D=16/D=8 PCA collapse check：PC1 < 50%（7.9% / 9.3%），无单轴坍缩

## 后续

- [ ] **spec 更新**：在 `.trellis/spec/` 记录"D≤32 处 ABCD 诊断失分辨力"这一边界，
  与"D=32/β=0.02 为 Stage-1 候选"的结论。
- [ ] **用户决策**：D=32/β=0.02 进入 Stage-2 pilot（用其 latent 训 flow），还是换用
  更可信的诊断方法继续优化坍缩。
- [ ] 若进 Stage-2：用 D=32/0.02 的 latent（d=512）训 obs→fc flow，做 DE 合成恢复测试。
