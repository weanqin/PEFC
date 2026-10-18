---
date: 2026-08-07
name: de-vae-stage15-audit
task: 08-07-de-vae-stage15-audit
type: 坐标级后验审计（Stage-1.5）— ABCD 分类 + decoder sensitivity S_j
dataset: modma
fold: 0
seed: 42
status: Phase 1-4 完成。D=32/β=0.02 在等压对照下优于 D=64/β=0.02（C 类坍缩 73.0%→61.7%，A 类 24.9%→37.3%，Pearson 持平 0.9995）。D=32 为新 Stage-1 候选。
checkpoint: checkpoints/de_vae/fold0_de_vae_D32_b0.02_mkl_variantA_best.pt（新候选）+ checkpoints/de_vae/fold0_de_vae_D64_b2e-2_mkl_variantA_best.pt（对照）
variant: de-vae variant A (encoder strides (2,2,2,1), T_lat=16, latent_dim ∈ {32,64}, beta_kl ∈ {0.02,0.01} mean, lambda_tv=0.05, lambda_corr=0.05)
related_spec: .trellis/spec/domain.md（关键 finding：后验二分——A 类健康 vs C 类坍缩，D 类=0）
---

# DE-VAE Stage-1.5：Variant-A 坐标级后验审计（de-vae-stage15-audit）

## 实验目的

Stage-1 架构已在 08-07 冻结为 Variant A。本任务优先级 1：对 β=0.02 的
Variant-A checkpoint 做坐标级后验审计，把 24.8% 的 σ<0.1 坐标分解为
A（确定性有效）/ B（理想活跃）/ C（真坍缩）/ D（病态过度自信）四类，
以 **D 类占比**（而非笼统的 σ<0.1 比例）作为后续 β sweep 的选型依据。

诊断在验证集（5 个 held-out 受试者，1673 个 fc-half 窗）上，对全部
131072 个 latent 坐标（C=128 × T_lat=16 × D=64）计算：

- `Var_x(μ_j)`（跨输入的均值方差）
- `mean_x(σ_j)`
- `KL_j`
- **decoder sensitivity `S_j = E_z[‖∂Dec(z)/∂z_j‖²]`**（新指标，= diag(J^T J)）

## 方法要点：S_j 的正确算法（重要修正）

`S_j` 是本任务唯一全新的数学量。初版用 "输出能量梯度"代理——
`∂(Σ_k x̂_k²)/∂z_j`——这是错的：它给出 `Σ_k 2 x̂_k J[k,j]`（x̂ 加权和），
**不等于** Jacobian 列模平方 `Σ_k J[k,j]²`。有限差分交叉校验抓住了这个
bug：corr=0.58 < 0.9，Gate G1 触发告警（见 §Rollback）。

修正后的正确且可行算法是 **Hutchinson 对角估计**：

```
diag(J^T J) = E_v[ (J^T v) ⊙ (J^T v) ],   v ~ N(0, I) over outputs
```

每个 `J^T v` 是一次 vector-Jacobian-product（`<x̂, v>` 对 z 反传一次），
共 `n_v=16` 次反传——而非 16384 次逐输出反传。实测与中心有限差分的
Pearson 相关 **corr = 0.911**，通过 >0.9 门槛。fp32、AMP 关（autograd 经
bf16 不稳）。

## 关键结果（β=0.02，Variant A）

| 类 | 占比 | 条件 | 含义 |
|---|---|---|---|
| **A** 确定性有效 | **24.9%** | Var(μ)>0.1 ∧ σ<0.1 ∧ S_j>median | 健康，高信息低不确定 |
| **B** 理想活跃 | 0.5% | Var(μ)>0.1 ∧ σ∈[0.1,0.8] ∧ S_j>median | 标准 active（极少） |
| **C** 真坍缩 | **73.0%** | Var(μ)<0.01 ∧ σ>0.9 ∧ KL<0.01 | posterior collapse，先验主导 |
| **D** 病态过度自信 | **0.00%** | Var(μ)<0.01 ∧ σ<0.1 ∧ S_j<median | 不存在 |
| 未分类 | 1.6% | — | 边界 |

**后验是二分的，无中间地带，无重叠：**
- 低 Var(μ) 坐标（73.0%）中 σ>0.9 的占 **100%**（p0=0.973, p50=0.992,
  p100=1.005）——全部坍缩到先验。
- σ<0.1 坐标（24.9%，即 08-07 的 24.8% 锚点）中 Var(μ)>0.1 的占 **100%**
  ——全部是健康 A 类。
- σ 分布双峰且中间有空隙：[0.1, 0.9] 区间内几乎没有坐标。

## 含义：原 PRD 的 β-sweep 目标失效

任务前提假设 24.8% 的 σ<0.1 坐标是"健康 A + 病态 D 的混合"，需用 ABCD
拆分并用 β sweep 消灭 D。**实测 D=0，A=24.9% 全健康**。因此：

- **Gate 2（最小化 D 类）在此 checkpoint 上是空操作**——没有 D 可降。
- **"提高 β 以降低 σ<0.1 比例"是错误方向**：σ<0.1 的全是健康 A，提 β
  只会把它们推向坍缩（A→C），扩大 C 类（73% → 更高）。
- 真正的后验健康问题是 **C 类 73% 坍缩**——而 σ<0.1 指标从未测量过它。

## 下一步（返回 Plan 重新定义）

β sweep 的**选型目标必须从"降 σ<0.1 / 降 D"改为"降 C（坍缩）/ 提 A"**。
待与用户确认后修订 prd.md R3/Gate 2，再进入 Phase 3 训练。候选修订：
- Gate 2：最大化 **A 类占比**（或等价地最小化 **C 类占比**），tiebreak
  保 Sampled Oracle Pearson。
- 或：β **下调**方向扫描（β<0.02），因为坍缩通常是 β 过高所致；
  原 {0.03,0.04,0.064} 上调方向可能完全错向。

## 输出物

- `results/de_vae/coord_audit_fold0_beta0p02.npz`（var_mu/sig_mean/kl_mean/
  S_j/class_map，均 (C,T_lat,D)）+ `.json`（counts + headline）。
- 待：修订后的 β sweep 结果（Phase 3-4，目标重定义后）。

## Rollback 记录（Phase 1, Gate G1）

- 初版 `sensitivity_Sj` 用输出能量梯度代理 → 与有限差分 corr=0.58，
  触发 [RP G1] 告警。诊断确认公式错误（见上）。改为 Hutchinson 对角
  估计后 corr=0.911 通过。有限差分交叉校验按设计捕获了 bug——
  **没有让错误的 S_j 进入分类**。这是 G1 gate 的价值。

## 复现

```
# env npi, from parent of latent_flow/
D:/program/anaconda/envs/npi/python.exe -m latent_flow.scripts.coord_audit \
    --fold 0 --split val --dataset modma --seed 42 --beta 0.02 \
    --ckpt checkpoints/de_vae/fold0_de_vae_D64_b2e-2_mkl_variantA_best.pt \
    --max-samples 3000 --n-probes 16 --fd-coords 64 --fd-delta 0.1
```

---

# Phase 2：D=32 在修复架构（Variant A）上重新评估（2026-08-08）

## 背景

Phase 1 发现 D=64/β=0.02 的真正问题是 **C 类坍缩 73%**（D 类=0）。本阶段
检验：在修复后的 Variant A 架构（strides (2,2,2,1)，真实 T_lat=16）上，
更小的 latent budget（D=32, d=512）是否能减少坍缩、同时保住 Oracle 上限。

之前的 D-ablation（08-06，判定 D=64 胜 D=32）是在**有缺陷的 Baseline**
（8→16 复制）上做的；所有现存 D=32 checkpoint 均为 Baseline 架构（无
variant 标签），不可信。本阶段在 Variant A 上重做。

## 对照设计（mean-KL 等压对照）

KL primitive（`models/de_vae.py:_kl`）是 **mean over D**，设计注释明言
"beta directly comparable across D … same beta applies the same per-
coordinate regularization at D=16/32/64"。因此：

- **D=32/β=0.02 vs D=64/β=0.02 = 等压对照**（每坐标 KL 压力相同，只差
  容量 d=512 vs 1024）—— **主判定**。
- D=32/β=0.01 = β 敏感性探针（更低正则压力）。
- D=64/β=0.01 不训练（mean-KL 下 β=0.02 是等压列，D=64/0.01 无干净对照价值）。

固定控制：variant A, segment_len=128, lambda_tv=0.05, lambda_corr=0.05,
50 epochs, seed 42, fold 0, bf16 AMP, max_grad_norm=1.0, mean-KL。

## 决策表（fold0/seed42, val split）

| D | β | role | Oracle Pearson | fc MAE | gate | A% | B% | C% | D% | uncateg% | S_j corr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 0.02 | anchor | 0.9995 | 0.0208 | PASS | 24.9 | 0.5 | **73.0** | 0.0 | 1.6 | 0.911 ✓ |
| **32** | **0.02** | **主对照** | **0.9995** | 0.0208 | PASS | **37.3** | 0.2 | **61.7** | 0.0 | 0.8 | 0.936 ✓ |
| 32 | 0.01 | probe | 0.9993 | 0.0240 | PASS | 50.0 | 0.0 | 21.1 | 0.0 | 28.9 | 0.884 ⚠ |

## 主对照结论：D=32/β=0.02 优于 D=64/β=0.02

判定规则：D=32/0.02 "更优" iff Pearson ≥ 0.99 AND C_class < 73.0%。

- **Oracle Pearson 持平**（0.9995 = 0.9995）→ 重建上限保住 ✓
- **C 类坍缩 73.0% → 61.7%（↓11.3 pp）** ✓
- **A 类健康 24.9% → 37.3%（↑12.4 pp）** —— 更小的 latent budget 让更多
  坐标保持活跃（资源更集中，冗余坍缩坐标更少）。
- S_j Hutchinson 校验 corr=0.936（>0.9，通过）→ ABCD 可信。

**→ D=32/β=0.02 是新的 Stage-1 候选。** 之前在缺陷架构上的"D=64 胜"判定
被推翻——在修复架构下 D=32 更优。

## β 敏感性探针：D=32/β=0.01（仅供参考，不参与主判定）

- σ<0.1 坐标从 37.3%（β=0.02）升到 68.2%（β=0.01），C 类从 61.7% 降到
  21.1%——更低 KL 压力让后验更活跃，符合预期。
- 但有两个警示（诚实记录）：
  1. **S_j corr=0.884 < 0.9**（门触发警告）：更低 β 下采样 z 分布更分散，
     decoder Jacobian 数值噪声增大，ABCD 分类中 S_j 相关部分可信度下降。
  2. **uncategorised=28.9%**（vs 0.02 的 0.8%）：7% 坐标 Var(μ) 落在
     [0.01,0.1] 中间态，加上 S_j 中位数附近的不确定，更多坐标无法归类。
- 探针的**与 S_j 无关的 σ/Var(μ) 分布数据**仍可信：β=0.01 让 D=32 后验
  显著活跃化（σ<0.1: 37%→68%），但 reconstruction 略差（MAE 0.0208→0.0240）。
- 探针不改变主判定（主判定靠 β=0.02 等压列）。

## 含义

- **Stage-1 候选更新为 D=32/β=0.02 Variant-A**（d=512，不再是 D=64/d=1024）。
- 这推翻了 08-06 的 D=64 胜出结论——该结论基于缺陷 Baseline 架构。
- D=32 仍未"治愈"坍缩（C 类仍 61.7%），但比 D=64（73%）明显改善。
- 下一步：是否进入 Stage-2 pilot（用 D=32/0.02 的 latent 训 flow），还是
  继续优化坍缩（D=16？更深的 encoder？）—— 待用户决策。

## Phase 2 复现

```
# env npi, from parent of latent_flow/
# 训练（2 个新 checkpoint；D=64/0.02 复用）
D:/program/anaconda/envs/npi/python.exe -m latent_flow.scripts.train_de_vae \
    --fold 0 --variant A --segment-len 128 --latent-dim 32 --beta-kl 0.02 \
    --lambda-tv 0.05 --lambda-corr 0.05 --epochs 50 --seed 42 \
    --save-path checkpoints/de_vae/fold0_de_vae_D32_b0.02_mkl_variantA_best.pt
# (同理 beta-kl 0.01)

# 评估 + 审计（每个 D=32 checkpoint）
D:/program/anaconda/envs/npi/python.exe -m latent_flow.scripts.eval_de_vae \
    --fold 0 --split val --seed 42 --ckpt <ckpt>
D:/program/anaconda/envs/npi/python.exe -m latent_flow.scripts.coord_audit \
    --fold 0 --split val --dataset modma --seed 42 --beta <B> --n-probes 16 \
    --fd-coords 64 --ckpt <ckpt>
```
