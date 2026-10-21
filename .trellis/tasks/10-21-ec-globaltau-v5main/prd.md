# EC_pert：07-04-global-tau 方案 × gpfm-mamba-v5-main 代理脑

## Goal

以 latent-flow-v1 的 07-04-global-tau 归因扰动 EC 推断方案为**唯一方法骨架**，指定 gpfm-mamba-v5-main（v5-main-01 θ\*）为代理脑，计算 EC_pert。扰动形式、度量标准、σ 搜索方法及其他一切 EC 推断设置**完全复用 07-04-global-tau**，不得改用其他默认配置或替代方案。

## 用户钉定约束（2026-10-21，逐条为硬约束）

1. **方法骨架**：07-04-global-tau（`run_oc_connectivity_global_tau` + `attribution/bisect_sigma.py`）——两阶段（σ_ref 响应扫描 → 全局 q 分位 τ → log 网格 σ\* 搜索）、加性高斯扰动 + μ_bias 方向符号、per-target MSE 度量、−(log σ\*+½log 2πe) 强度映射，全部逐字复用；σ 搜索原语必须 import `bisect_sigma.py`，禁止自实现。
2. **代理脑**：gpfm-mamba-v5-main（`outputs/gpfm-mamba-f0/v5-main-01/checkpoints/gpfm-mamba-v5-best.pt`）。
3. **扰动施加位置**：数据进入模型**之前**、z-score **之后**、**x^p 段**上；扰动后该段记 **x^p_pert**；基于 x^p_pert 计算 **x^f_pert**；**入模的 x_0 恒为扰动后版本**。
4. **数据流显式化**：任务配置（design.md §数据流）必须明确体现 x^p_pert、x^f_pert 与 x_0 的关系。
5. **一致性**：EC_pert 计算流程与结果语义与 07-04-global-tau 保持一致（同公式、同阈值机制、同边界语义）。

## 数据流（硬约束之展开；详见 design.md §2）

```
x^p (z-score 后 62×384)
  └─(+ σ·ξ_a + μ_bias^a，仅通道 a)→ x^p_pert ──作为 x_0 入模──→ [v5-main θ* EulerSampler]
                                                                    └→ x^f_pert (62×128)
x^p ──作为 x_0 入模──→ [同 θ*，同 (eps,eta)] ─→ x^f_base
MSE_b^a = mean_{win,time}(x^f_pert − x^f_base)²  →  τ_global(q=0.5) → σ*_ab（log 网格单调走查）
→ conn_ab = −(log σ*_ab + ½log2πe)；sign_ab = sign(mean-shift) → EC_pert = sign ⊙ conn
```

## Requirements

- R1 新路径包 `models/ec-globaltau-v1/`：runner + `GpfmProbeSite`（实现 `SignalProbeSite.perturb_and_measure` 同款契约），**import-only** 复用两源包，源包零逻辑改动。
- R2 07-04 设置复用对照表（research/globaltau-v5main-adaptation.md §3）中「同」项逐项落地；唯一开放拍板项 = n_trials，其余非「同」行均为代理脑契约映射（`euler_steps=16`、窗 384/128、fold0 test 10 例、**EMA 权重、bf16 autocast**——评审阻断 B1/B2 修法，briefing 拍板）。
- R3 确定性契约：同 seed 复跑逐位一致（cudnn deterministic + ξ 按源播种跨 σ 复用 + (eps,eta) 被试级固定跨源跨 σ，runner 内断言）；对 perturb-ec-pipeline.md §1 批内成对惯例的有意偏离（baseline 构造期一次 + 每 (源,σ) 单独调用，07-04 同形态）随档声明，以「同输入同批形跨调用逐位一致」断言替代 D(r=0)=0。
- R4 产物：per-subject JSON（σ\* 矩阵/connectivity/signed/raw response/sign/全超参，字段对齐 **07-04 JSON schema 全集**含 mu_bias_per_channel 与 n_sources_succeeded）落 `outputs/10-21-ec-globaltau-v5main/run_01/`。
- R5 登记：τ、边界钳位比例、耗时等诊断如实入 experiment.md；**新 EC 口径条目入 evaluation-calibers.md**；如纳入 GC 对照（briefing 拍板），ρ 按描述性登记不做对比交付。

## Acceptance Criteria

- [ ] AC1 扰动语义断言通过：扰动仅施加于 x^p 指定通道、在 z-score 后入模前；入模 x_0 恒为 x^p_pert；基线与扰动前向使用同一 (eps,eta) 且跨 σ 逐位一致。
- [ ] AC2 设置复用核验：代码中 q=0.5、σ_ref=0.3、log_grid[0.01,0.3]×24、walk_monotone_curve、−(log σ\*+½log2πe)、μ_bias=0.5·ch_std 均直接来自/等价 07-04 定义，无替代实现。
- [ ] AC3 冒烟跑通：fold0 test 至少 1 例端到端产出 07-04 schema JSON（62 源全成功）。
- [ ] AC4 确定性：同 seed 复跑同被试 σ\* 矩阵逐位一致。
- [ ] AC5 零源包改动（latent-flow-v1 / gpfm-mamba-v5 / gpfm-mamba-v1 只读）；产物仅 outputs + 新路径包。
- [ ] AC6 （若纳入）refs 对照描述性登记完成。

## 边界与不变式

- EC_pert 结果**不构成代理脑资格判定**（三项检验另立任务）；与 refs/历史数字仅描述性并置。
- 不修改 07-04 源脚本； latent-flow/gpfm 两源包只读。
- 单 fold（fold0）；不做跨 fold/跨代理脑对比（v5-fmt-01 等留后续拍板）。

## 子研究路由（1.0b / 1.0c）

- **分类结果**：`study2`（EC 推断算法，框架 Study II）
- **依据**：EC_pert 计算属 Study II Critical Perturbation Connectivity Inference；代理脑为 Study I 候选 gpfm-mamba-v5-main。
- **新路径/方法检测（1.0c）**：`新路径：globaltau-v5main（07-04 方案 × gpfm-v5-main 跨线适配）→ research 已触发（research/globaltau-v5main-adaptation.md，2026-10-21 本会话产出）`
- **research 产物**：`research/globaltau-v5main-adaptation.md`
