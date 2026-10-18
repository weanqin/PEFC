---
date: 2026-08-09
name: fm-gate-rework
aliases:
  - FM gate v2
  - feasibility gate rework
  - T-002 leakage fix
  - T-002D cross-channel
task: 08-09-fm-gate-rework
type: 评估
dataset: modma
fold: 0
status: 完成-阴性
tags:
  - experiment
  - evaluation
  - fm-gate
  - feasibility-gate
  - learnability-probe
  - cross-channel-probe
  - leakage-correction
  - inconclusive
  - result-negative
  - correction
checkpoint: checkpoints/de_vae/fold0_de_vae_D8_b0.01_mkl_variantA_best.pt
variant: variantA
related_spec: "[[feasibility-gate]]"
supersedes: "[[08-08-fm-feasibility-gate]]"
---

# FM Feasibility Gate v2（2026-08-09，rework）

> [!summary] 一句话结论
> 修正后的 Stage-2 FM 可行性闸门（修复 T-002 $\Delta z$ 代数泄漏 + 新增 T-002D 跨通道探针 + $R_{\mathrm{transition}}$ 降为描述性）在**同一冻结 latent** 上重跑，仍返回 **INCONCLUSIVE**，但**原因正确**：自历史可预测性真实但弱（直接 $z_{\mathrm{fc}}$ 预测 $R^2 \approx 0.10$，CI 下界低于阈值），且**跨通道结构对未来预测无额外贡献**（$\mathrm{gain}_{\mathrm{cross}} \approx 0$，CI 跨 0，仅 22.7% 通道为正）。原 [[08-08-fm-feasibility-gate]] 的"transition 真实且线性可预测（$R^2=0.489$）"结论**已被废止**——那个 $R^2$ 主要是 $\Delta z$ 自带的 $-z_{\mathrm{obs}}$ 代数项造成的伪影。

## 起因

外部审稿质疑 [[08-08-fm-feasibility-gate]] 的 INCONCLUSIVE 结论，提出两点结构性缺陷。本任务在 commit 前**对代码与原始数据**（`results/fm_gate/fold0_pairs.npz`）独立复核，两点均确认成立：

1. **$R_{\mathrm{transition}}$ 文档自相矛盾**：代码（`transition.py:157`）计算 $R = L_{\mathrm{id}}/L_{\mathrm{shuf}}$，但 spec/实验记录写 $L_{\mathrm{shuf}}/L_{\mathrm{id}}$。按代码口径，$R=1.0364$、CI `[1.012, 1.059]` **完全 >1** 意味着**真实配对距离显著大于被试内随机 re-pairing**，而非原文的"不可区分"。
2. **T-002 $\Delta z$ 代数泄漏**（核心）：target $\Delta z = z_{\mathrm{fc}} - z_{\mathrm{obs}}$ 自带 $-z_{\mathrm{obs}}$。在 npz 上重算复核——shipped 的 $R^2_{\mathrm{ridge}}=0.489$ 与 shuffle-target null 的 **0.466** 几乎一致，且接近独立 null 的理论值 **0.499**，即大部分 $R^2$ 是 $-z_{\mathrm{obs}}$ 代数项，而非真实 obs→fc 可预测性。直接预测 $z_{\mathrm{fc}}$ 的 R² 仅约 **0.103**。

> [!info] P0-4 决策（2026-08-09）
> 用户确认 Stage-2 FM 为**全通道条件化**（$Z_{\mathrm{obs}}^{1:C} \to z_{\mathrm{fc}}^{(c)}$）。因此新增的跨通道探针 T-002D **在范围内**，per-channel 自历史探针降为下界基线。

## 实验目的

用**结构性正确**的探针重测同一冻结 latent（DEVAE D=8/β=0.01，variant A，fold 0）的 obs→fc transition 可学性，区分：
- 自历史信号是否真实（剥离 $-z_{\mathrm{obs}}$ 代数伪影），
- 跨通道过去是否对单通道未来提供**额外**预测信息（EC 科学目标的核心问题），
- 据此给出可信的 Stage-2 go/no-go。

## 配置与执行

- **冻结编码器**：DEVAE `D=8 / beta=0.01 / variant A`（与 v1 同一 checkpoint `checkpoints/de_vae/fold0_de_vae_D8_b0.01_mkl_variantA_best.pt`）
- **latent 基质**：**复用** v1 的 `results/fm_gate/fold0_pairs.npz`（8881 windows × 128 ch × d=128），**无重新编码**
- **每通道维度**：$d = T_{\mathrm{lat}} \cdot D = 16 \cdot 8 = 128$；全部探针逐通道计算，通道中位数聚合
- **split**：train=6394 / val=814（**fold 0 仅 5 个 val 被试** → CI 天生偏宽，这是诚实信号，不人为收窄）
- **执行流水线**：
  - `run_transition_audit.py`（T-001，仅文档修正 + 加 `R_definition` 字段，数值不变）
  - `run_learnability_probe.py --out-suffix _v2 --mlp-hidden-dim 256`（T-002 v2：三种 target mode + 宽 MLP）
  - `run_cross_channel_probe.py --out-suffix _v2 --pca-k-max 256`（T-002D，新增）
  - `run_fm_gate_controller.py --suffix _v2`（T-006，重写 verdict 逻辑）
- **关键超参**：
  - T-002：`ridge_alpha=1.0`，`mlp_hidden_dim=256`（宽，无瓶颈，公平非线性比较），`mlp_epochs=200/patience=20`，`n_boot=1000`（frozen-model cluster bootstrap，95% percentile CI）
  - T-002D：`pca_k_max=256`，`pca_var_floor=0.95`（实际 $k=256$，解释方差 82.6%），`ridge_alpha=1.0`，`n_boot=1000`
- **代码版本**：`db534d8`（分支 `fm-gate-rework`）
- **关联任务**：[[08-09-fm-gate-rework]]（已归档至 `.trellis/tasks/archive/2026-08/`）

### 修正后的探针契约（与 v1 的差异）

| 探针 | v1 | v2 |
|---|---|---|
| T-002 主指标 | 预测 $\Delta z$，R² vs $L_{\mathrm{zero}}$ | **预测 $z_{\mathrm{fc}}$ 直接**，R² vs **persistence** $L_{\mathrm{persist}}$ |
| T-002 泄漏防护 | 无 | **$\mathrm{Skill}_{\Delta}$** vs 代数 null $\Delta\hat{z}_{\mathrm{alg}} = -z_{\mathrm{obs}} + \overline{\Delta z}_{\mathrm{train}}$ |
| T-002 control | 无 | **shuffle-target**（被试内 shuffle $z_{\mathrm{fc}}$ 重训） |
| MLP | `128→64→128`（瓶颈） | `128→256→128`（宽，公平比 Ridge） |
| T-002D | 不存在 | **新增**：全通道 $Z_{\mathrm{obs}}^{1:C}$ → $z_{\mathrm{fc}}^{(c)}$，PCA+Ridge |
| $R_{\mathrm{transition}}$ | Case III PASS 硬门槛 | **降为描述性**（endpoint proximity ≠ learnability） |

## 结果

### T-006 最终 verdict

> [!warning] INCONCLUSIVE（v2，原因正确）
> `final_verdict = INCONCLUSIVE`，`routing_action = COLLECT_MORE_EVIDENCE_OR_LONGER_DATA`。
> Case III PASS **两项独立条件均失败**：
> - 自历史：`R²_direct_mlp ci_low = 0.0280 <= MIN_R2 = 0.05`
> - 跨通道：`gain_cross ci_low = -0.0957 <= MIN_GAIN = 0.02`
>
> （非 `FAIL_CHEAT`：$L_{\mathrm{id}}$ 很大；非 `FAIL_NOISE`：`R²_direct_mlp ci_high = 0.174 > 0`。是"信号真实但弱 + 无跨通道增益"的中间态。）

### T-001 Transition Auditor（test split，文档修正，数值同 v1）

| 量 | 点估计 | 95% CI |
|---|---|---|
| $L_{\mathrm{id}}$ | 473.4257 | [376.8645, 564.0297] |
| $L_{\mathrm{shuf}}$ | 458.0189 | [366.8416, 539.7574] |
| $R_{\mathrm{transition}}$ | **1.0364** | **[1.0125, 1.0590]** |

> [!example] v2 解读（与 v1 不同）
> 按代码口径 $R = L_{\mathrm{id}}/L_{\mathrm{shuf}}$，CI 完全 >1 意味着**真实配对距离显著大于被试内随机 re-pairing**——真实 obs→fc 配对带有系统性结构，但**不是**"未来比随机未来更接近当前"那种。endpoint proximity ≠ transition learnability（$z_{t+1}=-z_t$ 完全可预测但 $\|Δz\|$ 很大）。故 $R_{\mathrm{transition}}$ 不再卡 PASS，仅作描述性。

### T-002 Learnability Probe v2（128 通道，1000 cluster bootstrap，channel-median）

| 量 | 点估计 | 95% CI | 说明 |
|---|---|---|---|
| **`R²_direct_ridge`**（主） | **0.1029** | [0.0262, 0.1792] | 直接预测 $z_{\mathrm{fc}}$ vs persistence——真实但弱的自历史信号，复现独立测算的 0.103 |
| `R²_direct_mlp`（主） | 0.1000 | [0.0280, 0.1741] | 宽 MLP 与 Ridge 持平 |
| `Skill_dz_ridge` | 0.3143 | [0.2754, 0.3769] | vs 代数 null；低于旧 $R^2_{\mathrm{dz}}$，证实大部分旧值是 $-z_{\mathrm{obs}}$ 伪影 |
| `Skill_dz_mlp` | 0.1959 | [0.1511, 0.2624] | 同上（MLP） |
| `R²_dz_ridge`（legacy） | 0.4887 | [0.4579, 0.5153] | 旧主指标（被泄漏污染），保留供审计 |
| `R²_dz_mlp`（legacy） | 0.3994 | [0.3693, 0.4306] | 同上 |
| **`R²_dz_shuffle_ridge`**（control） | **0.4650** | [0.4238, 0.4965] | **≈ legacy 0.489 → 泄漏一锤定音** |
| `nonlinear_gain`（宽 MLP） | −0.0015 | — | ≈0（公平比较）；v1 的 −0.10 是 64 维瓶颈伪影 |
| `L_persist` | 1.7309 | — | persistence 基线 |
| `L_alg` | 2.2627 | — | 代数 $-z_{\mathrm{obs}}$ null 基线 |
| `L_zero`（T-003） | 3.0148 | — | 零速度基线（legacy） |

> [!example] T-002 v2 关键数值
> 直接预测 $z_{\mathrm{fc}}$ 的 $R^2 \approx 0.10$（真实但弱）；shuffle-target control 的 $R^2 = 0.465$ 与 legacy $\Delta z$ 的 $0.489$ 几乎一致——**证明旧 $R^2$ 主要是 $-z_{\mathrm{obs}}$ 代数项，而非真实可预测性**。`Skill_dz_ridge=0.314` 为正说明 $\Delta z$ 回归仍捕获了**少量**超越代数 null 的信号，与直接 $R^2\approx0.10$ 量级一致。

### T-002D Cross-Channel Probe（新增，全通道 → 逐通道，PCA k=256，82.6% 方差）

| 量 | 点估计 | 95% CI | 说明 |
|---|---|---|---|
| `R²_cross_ridge` | **0.0768** | [0.0071, 0.1534] | 全通道预测 $z_{\mathrm{fc}}^{(c)}$ vs persistence |
| **`gain_cross`**（$R^2_{\mathrm{cross}} - R^2_{\mathrm{self}}$） | **−0.0261** | **[−0.0957, 0.0505]** | **负，CI 跨 0**——跨通道无额外贡献 |
| `gain_cross_positive_frac` | 0.2266 | — | 仅 22.7% 通道 gain > 0 |

> [!example] T-002D 关键数值（本次 rework 的核心新发现）
> 全通道过去**并不优于**单通道自历史（$R^2_{\mathrm{cross}}=0.077 < R^2_{\mathrm{self}}=0.103$），跨通道增益为负且 CI 跨 0，仅 22.7% 通道为正。**在 D=8 latent 上，跨通道动力学对未来 latent 没有提供可测的额外预测信息**——这对 EC（有效连接）科学目标是关键的负面信号。

### 与 v1 对比

| 指标 | v1（原） | v2（修正） | 变化 |
|---|---|---|---|
| 主 R² | 0.489（$\Delta z$，泄漏） | 0.103（直接 $z_{\mathrm{fc}}$） | **−0.386，泄漏被剥离** |
| nonlinear_gain | −0.10（瓶颈伪影） | −0.0015（宽 MLP） | 公平比较后≈0 |
| 跨通道 | 未测 | gain_cross = −0.026 | **新增，无跨通道结构** |
| $R_{\mathrm{transition}}$ 角色 | PASS 硬门槛 | 描述性 | 降级 |
| verdict | INCONCLUSIVE | INCONCLUSIVE | 同，但原因从"transition 太弱"变为"自历史弱 + 无跨通道增益" |

## 分析与结论

- **是否达成假设**：**否**（仍未达 PASS）。修正后的 PASS 条件（$R^2_{\mathrm{direct}}$ ci_low > 0.05 AND $\mathrm{Skill}_{\Delta}$ ci_low > 0 AND $\mathrm{gain}_{\mathrm{cross}}$ ci_low > 0.02）两项失败。

> [!warning] 核心解读（v2，诚实）
> 在 D=8/β=0.01 的冻结 latent 上：(1) 真实的 obs→fc **自历史**可预测性存在但弱（$R^2 \approx 0.10$，CI 偏宽因 fold 0 仅 5 val 被试）；(2) **跨通道结构对未来预测没有额外贡献**（$\mathrm{gain}_{\mathrm{cross}} \approx 0$，CI 跨 0，仅 22.7% 通道为正）。这对 EC 科学目标是关键负面信号，但**不否定 D=8 本身**——它只是首次被正确测量。

- **关键发现**：
  1. **v1 的 $R^2=0.489$ 是代数伪影**：shuffle-target control（0.465）≈ legacy（0.489）≈ 独立 null（0.499），三者一致一锤定音。真实自历史信号约 $R^2 \approx 0.10$。
  2. **D=8 latent 无可测跨通道未来预测结构**：$\mathrm{gain}_{\mathrm{cross}} < 0$，CI 跨 0。这是 v1 无法回答的问题（v1 没有跨通道探针），也是本次 rework 最重要的新发现。
  3. **非线性结构既无自历史也无跨通道**：宽 MLP（公平比较）的 `nonlinear_gain ≈ 0`，说明线性探针已吃满可测信号；对一个本应学非线性 velocity field 的 FM，可挖空间有限。

- **置信度**：所有 v2 R² 用 frozen-model cluster bootstrap（1000 resamples，95% CI）。fold 0 仅 5 val 被试 → CI 偏宽是**主要不确定性来源**，结论方向（自历史弱、跨通道无增益）稳健，但具体数值可能在更多被试下移动。

## 下一步

- **多 fold / EDRA 重跑**（最优先）：fold 0 仅 5 val 被试，CI 偏宽。更多被试能收窄 CI，可能把弱自历史信号推到显著（过 MIN_R2），也可能改变 gain_cross 的判断。
- **D=32/β=0.02 对照**（对 EC 目标关键）：在更高容量、后验不过自信的 latent 上重跑 gate，看 gain_cross 是否转正。若 D=32 上跨通道增益显著为正，说明 D=8 的容量瓶颈压没了跨通道信息——那才是适合 Stage-2 full FM 的 latent。
- **D2 multi-lag 兑现**：当前单相邻-lag；多 lag 可揭示信号是否在更短时间尺度存在。
- **是否进 full Stage-2 FM**：**不应基于当前 INCONCLUSIVE 草率推进**。建议先做上述 1–2 项 follow-up，若 gain_cross 在 D=32 或多 fold 下转正，再投入 full FM 训练。

## 产出物索引

| 类型 | 路径 |
|---|---|
| Verdict JSON（T-006 v2） | `results/fm_gate/fold0_verdict_v2.json` |
| Verdict JSON（T-006 v1，原，未触碰） | `results/fm_gate/fold0_verdict.json` |
| Learnability probe JSON（T-002 v2） | `results/fm_gate/fold0_learnability_probe_v2.json` |
| Cross-channel probe JSON（T-002D v2，新增） | `results/fm_gate/fold0_cross_channel_probe_v2.json` |
| Transition audit JSON（T-001，加 R_definition 字段） | `results/fm_gate/fold0_transition_audit_test.json` |
| Latent 对（R1，复用 v1） | `results/fm_gate/fold0_pairs.npz`（1.08 GB）|
| 分析代码 | `analysis/fm_gate/{shuffle,bootstrap,transition,learnability,cross_channel,verdict}.py` |
| 脚本 | `scripts/{run_transition_audit,run_learnability_probe,run_cross_channel_probe,run_fm_gate_controller}.py` |
| 单测 | `tests/test_fm_gate.py`（12 tests，全过）|
| 任务 PRD / 设计 / 实施 | `.trellis/tasks/archive/2026-08/08-09-fm-gate-rework/{prd,design,implement}.md` |
| Checkpoint（冻结编码器） | `checkpoints/de_vae/fold0_de_vae_D8_b0.01_mkl_variantA_best.pt` |

> [!note] JSON 产物被 gitignore
> `results/**/*.json` 由 spec `experiments/tracking.md` 约定为可重新生成，不入 git。所有 v2 JSON 可用上表脚本 + `fold0_pairs.npz` 重新生成。

## 关联笔记

- **废止**：[[08-08-fm-feasibility-gate]] —— v1 结论已废止（其顶部 Correction 节指向本笔记）
- 前序（Stage-1 pilot 选型）：[[08-08-de-vae-stage15-d16-d8]] —— D=8/β=0.01 选为 Stage-2 pilot
- 前序（Stage-1.5 audit）：[[08-07-de-vae-stage15-audit]]
- 前序（Stage-1 VAE）：[[08-06-de-stage1-vae-ablation]]
- 相关 spec：[[feasibility-gate]]（rework 后的探针契约）、[[domain]]、[[tracking]]、[[registry]]（★待实现）
