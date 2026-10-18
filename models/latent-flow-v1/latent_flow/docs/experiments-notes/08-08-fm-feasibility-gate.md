---
date: 2026-08-08
name: fm-feasibility-gate
aliases:
  - FM gate
  - feasibility gate
  - T-006 verdict
task: 08-08-fm-feasibility-gate
type: 评估
dataset: modma
fold: 0
status: 完成-阴性-已修正
tags:
  - experiment
  - evaluation
  - fm-gate
  - feasibility-gate
  - modma
  - transition-audit
  - learnability-probe
  - inconclusive
  - result-negative
checkpoint: checkpoints/de_vae/fold0_de_vae_D8_b0.01_mkl_variantA_best.pt
variant: variantA
related_spec: "[[domain]]"
---

# FM Feasibility Gate（2026-08-08）

> [!warning] Correction（2026-08-09，task `08-09-fm-gate-rework`）
> 下方原一句话结论**已废止**（superseded）。两点结构性缺陷经代码与原始数据复核确认：
>
> 1. **$R_{\mathrm{transition}}$ 解读错误**：代码（`transition.py:157`）计算的是 $R = L_{\mathrm{id}} / L_{\mathrm{shuf}}$（本节公式已勘误）。按代码口径，$R=1.0364$、CI `[1.012, 1.059]` **完全在 1 以上**意味着**真实配对距离显著大于被试内随机 re-pairing**，而非原文的"不可区分"。
> 2. **T-002 $\Delta z$ 回归有代数泄漏**：target $\Delta z = z_{\mathrm{fc}} - z_{\mathrm{obs}}$ 自带 $-z_{\mathrm{obs}}$ 项。复核（在原始 `fold0_pairs.npz` 上重算）显示 shipped 的 $R^2_{\mathrm{ridge}}=0.489$ 与 shuffle-target null 的 $0.466$ 几乎一致，且接近独立 null 的理论值 $0.499$——即大部分 $R^2$ 是 $-z_{\mathrm{obs}}$ 代数项造成，而非真实 obs→fc 可预测性。直接预测 $z_{\mathrm{fc}}$ 的 R² 仅约 **0.10**。
>
> 因此原 `PREDICTIVE_SIGNAL_DETECTED` 结论无效。**D=8/$\beta$=0.01 未被否定**——只是从未被正确测量。已在 task `08-09-fm-gate-rework` 中重构探针（直接 $z_{\mathrm{fc}}$ 预测、Skill vs algebraic null、shuffle-target control、跨通道 T-002D、宽 MLP），**完整重跑结果见独立报告** [[08-09-fm-gate-rework]]（v2 verdict：`results/fm_gate/fold0_verdict_v2.json`）。

> [!summary] 一句话结论 ~~（已废止，见上方 Correction）~~
> Stage-2 FM 可行性闸门返回 **INCONCLUSIVE**（路由 `COLLECT_MORE_EVIDENCE_OR_LONGER_DATA`），不是 PASS：`obs → fc` 的 transition 真实且**线性**可预测（$R^2_{\mathrm{ridge}}=0.489$），但它的幅度**与被试内随机 shuffle 不可区分**（$R_{\mathrm{transition}}=1.036$，CI 完全在 1.0 以上，`ci_high=1.059 \geq 0.95` 直接卡住 PASS）。证据真正是混合的——既非 near-identity cheat，也非 noise，而是个"transition 太弱"的尴尬中间态。

## 实验目的与假设

在为 Stage-2 full flow-matching（FM）模型投入 GPU 之前，跑一组便宜、大多 training-free 的诊断探针，判定冻结 DEVAE latent 空间里的 $z_{\mathrm{obs}} \to z_{\mathrm{fc}}$ transition 是否**可学**。闸门须区分三种 regime：

- **near-identity cheat**（Stage 2 学到什么都不做），
- **unpredictable noise**（有 transition 但随机），
- **genuinely learnable**（flow 能恢复信号），

只在第三种才 green-light full Stage-2 训练。

数学约定：每通道 latent $d = T_{\mathrm{lat}} \cdot D = 16 \cdot 8 = 128$；transition 距离 $L_{\mathrm{id}}[c] = \mathrm{mean}_n \|z_{\mathrm{fc}}[n,c] - z_{\mathrm{obs}}[n,c]\|^2$；shuffle 控制 $L_{\mathrm{shuf}}$ 在**被试内**置换；比值 $R_{\mathrm{transition}} = L_{\mathrm{id}} / L_{\mathrm{shuf}}$（分母恒为 $L_{\mathrm{shuf}}$，永不为 $L_{\mathrm{id}}$）。

> [!warning] 公式勘误（2026-08-09，task `08-09-fm-gate-rework`）
> 本节原写 $R_{\mathrm{transition}} = L_{\mathrm{shuf}} / L_{\mathrm{id}}$，与紧随其后的"分母恒为 $L_{\mathrm{shuf}}$"自相矛盾。代码（`analysis/fm_gate/transition.py:157-161`）实际计算的是 $R = L_{\mathrm{id}} / L_{\mathrm{shuf}}$。已更正为代码口径。这一更正也改变了 $R=1.0364$ 的解读：按代码口径它表示**真实配对距离显著大于被试内随机 re-pairing**（CI `[1.012, 1.059]` 完全在 1 以上），而非原文的"不可区分"。详见下方「Correction (2026-08-09)」节与 v2 verdict。

## 配置与执行

- **冻结编码器**: DEVAE `D=8 / beta=0.01 / variant A`（commit `8fb572f`），checkpoint `checkpoints/de_vae/fold0_de_vae_D8_b0.01_mkl_variantA_best.pt`（Stage-2 pilot 配置，见 [[08-08-de-vae-stage15-d16-d8]]「Stage-2 pilot 配置」节）
- **探针维度**: $d = T_{\mathrm{lat}} \cdot D = 128$/通道；全部探针**逐通道**计算，再用**通道中位数**聚合进 verdict（中位数对死通道稳健）
- **执行脚本**（流水线）:
  - `python scripts/extract_fm_gate_latents.py` → R1 latent 对（`results/fm_gate/fold0_pairs.npz`）
  - `python scripts/run_transition_audit.py` → T-001
  - `python scripts/run_learnability_probe.py` → T-002 + T-003 $L_{\mathrm{zero}}$
  - `python scripts/run_fm_gate_controller.py` → T-006 verdict
- **关键超参**:
  - T-001: `n_shuffle=100`, `n_boot=1000`（subject-level cluster bootstrap，95% percentile CI）
  - T-002: `ridge_alpha=1.0`, `mlp_hidden_dim=64`, `mlp_epochs=200`, `mlp_patience=20`, `n_boot=1000`（frozen-model bootstrap：Ridge+MLP 拟合一次，重采样 val 被试）
  - 审计 split: **test**（1673 windows, 10 held-out subjects）
  - 训练/验证: `n_train=6394`, `n_val=814`
- **模型变体**: frozen DEVAE variant A，stride `(2,2,2,1)`, $T_{\mathrm{lat}}=16$
- **代码版本**: `8fb572f`（docs: D=8/beta=0.01 选为 Stage-2 pilot）；本任务新增代码在 `analysis/fm_gate/` 与 `scripts/`
- **关联任务**: [[08-08-fm-feasibility-gate]]
- **阈值（pilot heuristic，非理论绝对）**: `MIN_TRANSITION_FLOOR=1e-4`, `MAX_R_TRANSITION=0.95`, `MIN_R2=0.05`

### 方法论要点

- **被试内 shuffle guard（T-001）**: 置换永不跨被试——每个被试独立 draw 一个 permutation，应用到该被试全部通道。由 `analysis/fm_gate/shuffle.within_subject_permutation` 强制，并在 `tests/test_fm_gate.py` 单测覆盖（跨被试 shuffle 必须失败/被证 guard 存在）。
- **Cluster（被试级）bootstrap**: T-001 重采样时重组每被试距离和；T-002 用 frozen-model bootstrap（模型拟合一次，重采样 val 被试）。1000 resamples，95% percentile CI。fold 0 只有 10 个 test 被试 → CI 可能偏宽，这是诚实的，不人为收窄。
- **Lag-curve DEFERRED（D2）**: 当前 pipeline 每窗口只产出一个相邻 obs/fc 对（`x_obs = x_target[:, :L]`, `x_fc = x_target[:, L:]`, $L=128$）；多 lag 抽取是 follow-up，本任务仅报单相邻-lag 点。
- **T-003 合并进 T-002**: $L_{\mathrm{zero}} = \mathrm{mean}(\|\Delta z_{\mathrm{val}}\|^2)$ 在 T-002 内逐通道计算并导出，无独立模板。

## 结果

### T-006 最终 verdict

> [!warning] INCONCLUSIVE（非 PASS）
> `final_verdict = INCONCLUSIVE`，`routing_action = COLLECT_MORE_EVIDENCE_OR_LONGER_DATA`。
> **不是** Case III（PASS）：`R_transition` 的 `ci_high=1.0590 \geq MAX_R_TRANSITION (0.95)`，transition 真实但弱可学。
> **不是** `FAIL_CHEAT`（Case I）：$L_{\mathrm{id}}$ 很大（473.4），非 near-zero identity。
> **不是** `FAIL_NOISE`（Case II）：$R^2_{\mathrm{mlp}}$ `ci_high=0.422 > 0`，transition 是可预测的。
> 它是那个尴尬的中间态：transition 真实且线性可预测，但其幅度不大于被试内的随机 shuffle。

### T-001 Transition Auditor（test split, 1673 windows, 10 subjects, channel-median）

| 量 | 点估计 | 95% CI |
|---|---|---|
| $L_{\mathrm{id}}$ | 473.4257 | [376.8645, 564.0297] |
| $L_{\mathrm{shuf}}$ | 458.0189 | [366.8416, 539.7574] |
| $R_{\mathrm{transition}}$ | **1.0364** | **[1.0125, 1.0590]** |

> [!example] T-001 关键数值
> $R_{\mathrm{transition}} = 1.036$，CI `[1.012, 1.059]` **完全在 1.0 以上**。
> $R_{\mathrm{transition}} \approx 1$ 意味着 paired 距离与 shuffled control 统计上不可区分——`obs → fc` 的 transition 在幅度上**与被试内随机 re-pairing 一样大**。PASS 要求 `ci_high < 0.95`，而 `1.059 \geq 0.95` → 这一项单独就 block 了 PASS。

- `zero_transition_flag = false`（$L_{\mathrm{id}}$ CI 上界不与 $1\mathrm{e}{-8}$ 重叠，非零 transition）
- `denominator_check`: $R_{\mathrm{transition}}$ 分母恒为 $L_{\mathrm{shuf}}$（永不为 $L_{\mathrm{id}}$）
- `within_subject_check`: 由 `analysis/fm_gate/shuffle.within_subject_permutation` 强制
- `lag_curve`: 单 lag 点（`{lag: 1, distance: 473.4257, note: "single adjacent-lag pair; multi-lag deferred (D2)"}`）
- 逐通道 `zero_transition_frac = 0.0`（无通道被判零 transition）

### T-002 Learnability Probe（128 通道, 1000 cluster bootstrap, channel-median）

| 量 | 点估计 | 95% CI |
|---|---|---|
| $R^2_{\mathrm{ridge}}$ | **0.4887** | [0.4577, 0.5156] |
| $R^2_{\mathrm{mlp}}$ | **0.3896** | [0.3585, 0.4218] |
| `nonlinear_gain` ($R^2_{\mathrm{mlp}} - R^2_{\mathrm{ridge}}$) | **−0.0998** | — |
| $L_{\mathrm{zero}}$（T-003） | 3.0148 | median over channels |

> [!example] T-002 关键数值
> $R^2_{\mathrm{ridge}} = 0.489$ —— $\Delta z$ 可由 $z_{\mathrm{obs}}$ **强线性预测**。
> $R^2_{\mathrm{mlp}} = 0.390$ —— MLP 比 Ridge **更差**。
> `nonlinear_gain = −0.10` —— **负**：transition 结构是线性的，**没有** Ridge 之外可供 flow 利用的非线性结构。
> `status = PREDICTIVE_SIGNAL_DETECTED`（$R^2_{\mathrm{mlp}}$ CI 远在 0 之上）。

- `target_check`: 预测 $\Delta z = z_{\mathrm{fc}} - z_{\mathrm{obs}}$（从不直接预测 $z_{\mathrm{fc}}$）
- 逐通道 `no_signal_channel_frac = 0.0`（无通道被判无信号）

### T-004 FM Skill Calculator — 未运行

> [!note] FM_NOT_TRAINED
> `t004_status = "FM_NOT_TRAINED"`。尚无 pilot flow-matching 模型存在；FM 分支是本任务**设计**的 contract stub（见 design.md §5），供下游 FM-build 任务接入。T-006 controller 在 T-004 json 缺席时跳过 early-reject 分支，仅用 T-001/T-002 出 verdict，并显式标注。controller 现在可跑、前向兼容。

### 与 SOTA / 基线对比

本闸门不与某个预测训练 SOTA 直接比指标——它是一个**可行性判定**，衡量"冻结 latent 空间的 transition 是否值得上 full FM"。参照系是闸门自身的三个阈值（pilot heuristic）：

| 判据 | 阈值 | 实测 | 通过？ |
|---|---|---|---|
| Case I（非 cheat）| $L_{\mathrm{id}}$ `ci_high < 1e-4` | 564.03 | 是（远高于 floor，非 identity）|
| Case II（非 noise）| $R^2_{\mathrm{mlp}}$ `ci_high > 0` | 0.4218 | 是（可预测）|
| Case III part A（transition 显著强于 shuffle）| $R_{\mathrm{transition}}$ `ci_high < 0.95` | **1.0590** | **否**（block）|
| Case III part B（最低可学）| $R^2_{\mathrm{mlp}}$ `ci_low > 0.05` | 0.3585 | 是 |

三项里 Case III part A 失败，故 PASS 条件不满足。其余三项（非 cheat / 非 noise / 最低 $R^2$）都过——这正是 INCONCLUSIVE 的来源：transition **有**且**可线性预测**，只是其幅度被被试内 shuffle 完全解释。

## 分析与结论

- **是否达成假设**: **否**（未达 PASS）。假设是"若可学则 transition 应显著强于 shuffle 且 flow 有非线性结构可挖"；实测 transition 可学（线性）但其幅度不大于 shuffle，且无线性之外的增益。

> [!warning] 核心矛盾（诚实记录）
> transition 是**真实的**（$L_{\mathrm{id}}=473.4 \gg 0$）且**可预测的**（$R^2_{\mathrm{ridge}}=0.489$），但它的**幅度与被试内随机 shuffle 不可区分**（$R_{\mathrm{transition}}=1.036$，CI 完全在 1 以上）。这不是"没有 transition"，也不是"transition 是噪声"——它是"transition 太弱，被被试内的自然变异性完全淹没"。

- **关键发现**:
  1. **transition 是线性的，不是非线性的。** `nonlinear_gain = −0.10`：MLP 比 Ridge 更差。意味着 $z_{\mathrm{obs}} \to z_{\mathrm{fc}}$ 的映射没有非线性结构可供 flow-matching（一个非线性 transport）去挖——线性回归已吃满。对一个本应学非线性 velocity field 的 FM 来说，这是不利信号。
  2. **paired 距离与 shuffle 同量级**（$L_{\mathrm{id}}=473.4$ vs $L_{\mathrm{shuf}}=458.0$，$R=1.036$）：真实的 $z_{\mathrm{fc}}$ 与 $z_{\mathrm{obs}}$ 的对应关系，在距离上不比"同被试随机配的另一个 $z_{\mathrm{fc}}$"更近。即 transition 信号弱到被被试内自然变异性吞没。
  3. **$L_{\mathrm{id}} \gg L_{\mathrm{zero}}$ 的尺度差异是良性的**：$L_{\mathrm{id}}$ 是 128 维向量的逐元素平方距离（量级 ~数百），$L_{\mathrm{zero}}$ 是**标准化后**的 $\Delta z$（T-002 在 train split 标准化 X、y），故两者不可直接比；它们在不同标准化口径下。这点 design.md 未显式说明，记录在此以防误读。

- **置信度**: T-002 的 $R^2$ CI 相对窄（frozen-model bootstrap）；T-001 的 $R_{\mathrm{transition}}$ CI 完全在 1 以上（cluster bootstrap），但 fold 0 仅 10 个 test 被试，cluster bootstrap 的 CI 天生偏宽。结论方向（transition 幅度不大于 shuffle）稳健，但具体数值 $1.036$ 可能在更多被试/数据下移动。

## 下一步

> [!info] v2 重跑结果（2026-08-09，task `08-09-fm-gate-rework`）
> 重构后的探针在**同一冻结 latent**（`fold0_pairs.npz`，无重新编码）上重跑，结论仍为 **INCONCLUSIVE**，但**原因正确**且证据可信：
>
> | 探针 | 指标 | 值 (channel-median, 95% CI) | 解读 |
> |---|---|---|---|
> | T-002 直接 $z_{\mathrm{fc}}$ 预测 | `R²_direct_ridge` | **0.1029** [0.0262, 0.1792] | 复现独立测算的 0.103；真实但弱的自历史信号 |
> | T-002 直接 $z_{\mathrm{fc}}$ 预测 | `R²_direct_mlp` | 0.1000 [0.0280, 0.1741] | 宽 MLP 与 Ridge 持平（公平比较） |
> | T-002 代数 null skill | `Skill_dz_ridge` | 0.3143 [0.2754, 0.3769] | 低于旧 R²_dz=0.489——证实大部分旧值是 $-z_{\mathrm{obs}}$ 代数项 |
> | T-002 shuffle-target control | `R²_dz_shuffle_ridge` | **0.4650** | 与 `R²_dz_ridge=0.489` 几乎一致 → 泄漏确认 |
> | T-002 公平非线性增益 | `nonlinear_gain` | −0.0015 | ≈0（宽 MLP）；旧的 −0.10 是 64 维瓶颈伪影 |
> | T-002D 全通道→逐通道 | `R²_cross_ridge` | 0.0768 [0.0071, 0.1534] | 全通道并不优于自历史 |
> | **T-002D 跨通道增益** | **`gain_cross`** | **−0.0261** [−0.0957, 0.0505] | **负，CI 跨 0**——D=8 latent 中无可测的跨通道未来预测结构 |
>
> **v2 verdict**: `INCONCLUSIVE`（`R2_direct_mlp ci_low=0.0280 <= MIN_R2=0.05` 且 `gain_cross ci_low=-0.0957 <= MIN_R2=0.02`）。PASS 条件未满足的两个独立维度都失败：自历史 R² 的 CI 下界低于阈值，且跨通道增益 CI 跨零。
>
> **核心解读（v2）**：在 D=8/β=0.01 的冻结 latent 上，(1) 真实的 obs→fc 自历史可预测性存在但弱（R²≈0.10，CI 偏宽因 fold 0 仅 5 个 val 被试）；(2) **跨通道结构对未来预测没有额外贡献**（gain_cross≈0）——这对 EC（有效连接）科学目标是关键负面信号，但不否定 D=8 本身（见下方「下一步」D=32 对照）。原报告"transition 真实且线性可预测但幅度太弱"的结论被取代：真实信号存在，但既非线性强可预测（旧 0.49 是泄漏），也无跨通道增益。
>
> 完整 v2 verdict：`results/fm_gate/fold0_verdict_v2.json`。

- **多 lag 抽取（D2 兑现）**: 当前只有单相邻-lag 点。多 lag 能显示 transition 可预测性如何随 lag 衰减——若短 lag 下 $R_{\mathrm{transition}}$ 显著下降（更接近 identity），则说明信号在更短时间尺度上存在，只是 1-lag 看不到。这是一个关键且便宜的 follow-up。
- **更多被试 / 更长数据**: $R_{\mathrm{transition}}$ CI 偏宽主因是 fold 0 仅 10 个 test 被试。EDRA 或更多 fold 能否收窄 CI、把 $1.036$ 推到 1 以下，是 routing action 直接建议的方向。
- **T-004 FM branch 兑现**: pilot FM 建好后重跑 T-006 controller（带 T-004 json），用 `Skill_FM = 1 - L_{\mathrm{FM}}/L_{\mathrm{zero}}` 抓"low-loss-no-skill" cheat。当前 controller 已是 forward-compatible。
- **重估 transition 定义**: 探查"transition 弱"是否部分来自 DEVAE D=8 的**过自信后验**（σ<0.1 占 99.45%，见 [[08-08-de-vae-stage15-d16-d8]]「Stage-2 pilot 配置」已知风险）——过自信的确定性 latent 可能让 $z_{\mathrm{obs}}, z_{\mathrm{fc}}$ 都非常确定，从而把 transition 信号压平。考虑用 D=32/β=0.02 fallback（坍缩更高但后验不过自信）对照重跑 gate。

- **待处理**:
  - checkpoint `fold0_de_vae_D8_b0.01_mkl_variantA_best.pt` 仍**未登记**进 `docs/model_registry.yaml`（该 registry 尚不存在，见 [[registry]] ★待实现）——需后续补登记。
  - 多 lag 与 EDRA 重跑是独立 follow-up，不在本任务范围。

## 产出物索引

| 类型 | 路径 |
|---|---|
| Verdict JSON（T-006 v1，原） | `results/fm_gate/fold0_verdict.json` |
| **Verdict JSON（T-006 v2，重构后）** | `results/fm_gate/fold0_verdict_v2.json` |
| Transition audit JSON（T-001） | `results/fm_gate/fold0_transition_audit_test.json` |
| Learnability probe JSON（T-002 v1，原） | `results/fm_gate/fold0_learnability_probe.json` |
| **Learnability probe JSON（T-002 v2，泄漏修正）** | `results/fm_gate/fold0_learnability_probe_v2.json` |
| **Cross-channel probe JSON（T-002D v2）** | `results/fm_gate/fold0_cross_channel_probe_v2.json` |
| Latent 对（R1，8881 windows × 128 ch × d=128） | `results/fm_gate/fold0_pairs.npz`（1.08 GB）|
| 分析代码 | `analysis/fm_gate/{shuffle,bootstrap,transition,learnability,cross_channel,verdict}.py` |
| 脚本 | `scripts/{extract_fm_gate_latents,run_transition_audit,run_learnability_probe,run_cross_channel_probe,run_fm_gate_controller}.py` |
| 单测 | `tests/test_fm_gate.py`（12 tests，全过）|
| 任务 PRD / 设计（v1） | `.trellis/tasks/archive/2026-08/08-08-fm-feasibility-gate/{prd,design}.md` |
| **任务 PRD / 设计 / 实施（v2 重构）** | `.trellis/tasks/08-09-fm-gate-rework/{prd,design,implement}.md` |
| Checkpoint（冻结编码器） | `checkpoints/de_vae/fold0_de_vae_D8_b0.01_mkl_variantA_best.pt` |

## 关联笔记

- 前序（Stage-1 pilot 选型）: [[08-08-de-vae-stage15-d16-d8]] —— D=8/β=01 被选为 Stage-2 pilot 配置；本 gate 即在其冻结 latent 上跑。
- 前序（Stage-1.5 audit）: [[08-07-de-vae-stage15-audit]]
- 前序（Stage-1 VAE）: [[08-06-de-stage1-vae-ablation]]
- 相关 spec: [[domain]]（SOTA 与历史发现）、[[tracking]]（run 记录与命名约定）、[[registry]]（checkpoint 登记，★待实现）
- 关联 checkpoint: 见 frontmatter `checkpoint` 字段
