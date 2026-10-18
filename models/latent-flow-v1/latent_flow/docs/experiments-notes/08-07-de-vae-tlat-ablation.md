---
date: 2026-08-07
name: de-vae-tlat-ablation
task: 08-07-de-vae-tlat-ablation
type: 架构消融（T_lat 真实来源对比 — 8→16 虚假复制修复）
dataset: modma
fold: 0
seed: 42
status: 采纳 Variant A 为架构方向；σ<0.1=24.8% 仍未过 20% 门槛，Stage-1 仍未冻结（β sweep 转下一任务）
checkpoint: checkpoints/de_vae/fold0_de_vae_D64_b2e-2_mkl_variantA_best.pt（采纳架构）
variant: de-vae variant A (encoder strides (2,2,2,1), T_lat=16 real, latent_dim=64, beta_kl=2e-2 mean, lambda_tv=0.05, lambda_corr=0.05)
related_spec: .trellis/spec/domain.md（1 个新 finding：8→16 复制机制 + Variant A 采纳）
---

# DE-VAE T_lat 真实来源消融：8→16 虚假复制修复（de-vae-tlat-ablation）

## 实验目的

验证上一任务（08-06）的 P0 诊断发现的根因假设：DEVAE 的 `ConvEncoder`
在 `L=128` 上用 4×stride-2 卷积压到 `T_down=8`，再由
`AdaptiveAvgPool1d(t_lat=16)` 插值回 16（"8→16 虚假复制"），导致：

1. 后验低方差：σ<0.1 = 33.7%（目标 <20%）。
2. 协方差病态：`S_mu` Ledoit-Wolf 有效秩仅 9.6 / 1024。

若假设成立，修复架构使 `T_lat=16` 成为真实时间维度（Variant A），应同时改善
两项指标。

## 控制变量（固定）

- D=64，β_mean=0.02（mean-KL，沿用 08-06 P0 裁决胜者）
- λ_tv = λ_corr = 0.05（不变）
- seed=42，fold=0，L=128（obs=fc=128，window=256）
- 50 epochs，bf16 AMP，max_grad_norm=1.0，env npi（RTX 5070 Ti）

## 三个 Variant

| Variant | encoder strides | T_down | t_lat | pool 行为 | d |
|---|---|---|---|---|---|
| **Baseline** | (2,2,2,2) | 8 | 16 | 插值 8→16（复制） | 1024 |
| **Variant A**（采纳） | (2,2,2,1) | 16 | 16 | 恒等 | 1024 |
| Variant B（排除） | (2,2,2,2) | 8 | 8 | 恒等 | 512 |

实现方式（隔离决策）：将 `ConvEncoder`/`ConvDecoderFc` 拷贝到
`models/de_vae.py` 为 `DEVAEConvEncoder`/`DEVAEConvDecoderFc`（参数化
`strides`），**不修改 `oc_latent_flow.py`**（维持 08-06 隔离不变量）。
Baseline 复用旧 checkpoint（不重训）；R-T1 回归测试证明
`DEVAEConvEncoder(strides=(2,2,2,2))` 与 `ConvEncoder` bitwise 等价
（diff=0.0），旧 checkpoint 无漂移加载。

## 结果

### 决定性证据：成对差异方差（P0-5）

`Var(μ_{2k} − μ_{2k+1})` 对 k=0..7 的均值，归一化为与 pair-mean 方差的比值：

| Variant | pair_diff_var | pair_null_var | **ratio** |
|---|---|---|---|
| Baseline | **0.000000** | 0.6712 | **0.0000** |
| Variant A | 1.0192 | 0.3357 | 3.0365 |
| Variant B | 1.5057 | 0.4908 | 3.0678 |

**Baseline 的 pair_diff_var 是精确零** —— `μ_{2k}` 与 `μ_{2k+1}` 逐点相等。
这不是近似复制，是 `AdaptiveAvgPool1d(16)` 在 length-8 输入上的确定性产物
（每个输出位置是相邻输入位置的加权平均，相邻对共享几乎全部输入支撑）。
Variant A/B 的 ratio ≈ 3.0，证明相邻时间步具有独立信息。**核心假设 100% 确证。**

### 完整诊断对比

| 指标 | Baseline | **Variant A** | Variant B | 门/目标 |
|---|---|---|---|---|
| σ<0.1 | 33.7% | **24.8%** ↓ | 86.0% ↑↑ | <20% |
| σ>0.8 | 65.4% | **74.5%** ↑ | 0.0% | >60% |
| logvar clamp_lo | 32.7% | **24.7%** ↓ | 79.9% | — |
| **Sampled Pearson** | 0.9976 | **0.9974** | 0.9692 ↓ | >0.99 |
| pair_diff ratio | 0.0000 | **3.04** | 3.07 | >5× baseline |
| S_mu eff_rank | 9.2 | **9.8** | 8.2 | >30（期望） |
| S_z eff_rank | 8.2 | **9.4** | 6.7 | — |
| S_z/S_mu ratio | 0.89 | **0.96** | 0.82 | — |
| perturb δ_norm | 1.47e-4 | **3.37e-4** (2.3×) | 2.85e-4 | — |
| 训练 epochs | 50 | 50 | **13**（early stop） | — |
| smoke recon MAE | 0.0237 | 0.0242 | 0.1703 | — |

### Variant A 门判定

| 门 | 结果 |
|---|---|
| pair_diff > 5× baseline | ✅ 通过（A=1.02, baseline=0.0, ∞×） |
| Sampled Pearson > 0.99 | ✅ 通过（0.9974，天花板保持） |
| σ<0.1 < 20% | ❌ 未通过（24.8%，但从 33.7% 改善 9pp） |
| S_mu rank > 30 | ❌ 远未达到（9.8，仅 +0.6） |

决策场景：**`partial_improvement`**。

## 阶段性结论

1. **"8→16 虚假复制"假设完全确证**：Baseline 的 pair_diff_var 为精确零，
   一半名义时间维度是严格冗余。这是 σ<0.1 异常的**一个**根因。

2. **Variant A 是明确更好的架构**：重建天花板无损（Pearson 0.9974≈0.9976），
   后验健康全面改善（σ<0.1 33.7%→24.8%, σ>0.8 65.4%→74.5%, clamp_lo 32.7%→24.7%），
   无冗余维度（pair_diff ratio 3.04），方向扰动响应提升 2.3×。
   **采纳 Variant A（encoder strides (2,2,2,1)）作为 DEVAE 架构方向。**

3. **但架构修复单独不足以冻结 Stage-1**：σ<0.1=24.8% 仍 >20% 门，S_mu 有效秩
   9.8 远低于预期 30+。这说明"8→16 复制"约解释 9pp 的 σ 改善，**剩余 ~25%
   低方差维度另有成因** —— 最可能是 KL 压力下的标准后验坍塌（β 把多余维度
   推向先验）。架构缺陷（已修）+ β 调度问题（待解）是叠加的两个问题。

4. **Variant B 明确失败**：t_lat=8 容量不足（Pearson 0.9692 <0.98，early stop
   epoch 13，σ<0.1=86% 更糟）。**B 不可取。**

## 决策与下一步

- **采纳 Variant A** 为 DEVAE 架构（冻结 encoder strides (2,2,2,1), T_lat=16 real）。
- Stage-1 **仍未冻结**：σ<0.1=24.8% >20% 门，S_mu rank 9.8 远低于 30。
- **β sweep（0.03/0.04）转下一独立任务**：在 Variant A 架构上重新评估 β 上调
  能否把 σ<0.1 压到 <20%。架构修复是 β sweep 可解释的必要前提（已满足）。
- 所有基于 Baseline（T=8→16）的 checkpoint 仅可用于代码逻辑 Pilot，不得进入
  正式 Stage-2 训练或 DE 推断。

## 相关 spec 更新

- `.trellis/spec/domain.md`：新增 finding "DE Stage-1 T_lat 真实来源消融
  （8→16 复制确证 + Variant A 采纳）"。
- `.trellis/spec/models/architecture.md`：DEVAE variant 字段 + 8→16 复制机制说明。

## 文件（08-07）

- `models/de_vae.py` — 新增 `DEVAEConvEncoder`/`DEVAEConvDecoderFc`（参数化 strides）
  + `_VARIANT_REGISTRY` + `DEVAE(variant=...)`。`oc_latent_flow.py` 零改动。
- `scripts/train_de_vae.py` — `--variant {baseline,A,B}` CLI。
- `scripts/p0_diagnostics.py` — 新增 `p0_5_pair_difference`/`p0_4_sz_eff_rank`/
  `p0_6_perturbation`。
- `scripts/run_tlat_ablation.py` — 消融驱动 + 决策门。
- `tests/test_de_vae.py` — R-T1/R-T2/R-T3 新测试（14/14 通过）。
- Checkpoints: `fold0_de_vae_D64_b2e-2_mkl_variant{A,B}_best.pt`。
- Results: `results/de_vae/tlat_ablation_fold0.json`。

## 复现命令

```bash
# 训练 Variant A（采纳架构）
D:\program\anaconda\envs\npi\python.exe -m latent_flow.scripts.train_de_vae \
    --fold 0 --variant A --segment-len 128 --latent-dim 64 --beta-kl 2e-2 \
    --lambda-tv 0.05 --lambda-corr 0.05 --epochs 50 --seed 42 \
    --save-path checkpoints/de_vae/fold0_de_vae_D64_b2e-2_mkl_variantA_best.pt

# 全诊断 sweep + 决策表
D:\program\anaconda\envs\npi\python.exe -m latent_flow.scripts.run_tlat_ablation --fold 0
```
