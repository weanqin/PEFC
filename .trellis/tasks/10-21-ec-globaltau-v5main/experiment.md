# 实验日志 — EC_pert：07-04-global-tau × gpfm-mamba-v5-main

> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务 outputs 目录。

## Run 001 — 加载与扰动语义断言 smoke（S1/S2）

### 目的与假设

- 验证 adapter 加载链（EMA 权重生效）、三断言（ξ 跨 σ 逐位一致 / (eps,eta) 跨源跨 σ 逐位一致 / 扰动仅落 x^p 指定通道）与同输入跨调用逐位一致；随后 fold0 test 1 例端到端冒烟产出 07-04 schema JSON。

### 配置

- 代码：`models/ec-globaltau-v1/`（新建，import-only）；θ\* 数值口径 = EMA swap + bf16 autocast（briefing 拍板 2026-10-21）
- 参数：q=0.5、σ_ref=0.3、log_grid[0.01,0.3]×24、μ_bias=0.5·ch_std（母体=实际入模窗集）、euler_steps=16、n_trials=（briefing 拍板）
- 环境：base conda @ 本机 4080 SUPER；cudnn deterministic
- 数据：fold0 test 例（edra v4，edra_process_v4 @250 Hz，384/128/stride 64）

### 结果

- **S1 五断言全过**：扰动隔离（非源通道逐位不动）/ 同输入跨调用逐位一致 / **EMA swap 生效（swap 域内外 max|Δ|=1.158）** / ξ 跨 σ 逐位一致 / (eps,eta) 跨源跨 σ 逐位一致；剂量-响应抽查 MSE(0.20)≥MSE(0.05) 靶占比 100%。（评审阻断 B1 修法验证：explore 先例系 raw-weights，swap 内外差异显著存在。）
- **S2 冒烟（s12 × 32 窗）**：**1.50 min/例**；1321 窗子采样 32；τ_global(q=0.5)=**0.045381**；边界钳位 **@σ_min 46.1% + @σ_max 49.7% = 95.8% 近二元**（latent-flow MODMA 侧 91.5–92.3% 同型）；62/62 源成功；σ\* offdiag 中位 = 0.30（σ_max 端）；signed 负 51.0% / 正 49.0%；conn offdiag ∈ [−0.215, 3.186]（= σ_max/σ_min 钳位值）。
- **S3 确定性**：同 seed 复跑 σ\*/τ/conn **逐位一致**（bitwise）。
- **S5 refs 对照（s12，描述性）**：r_signed_aligned=**−0.149**、r_abs_aligned=0.151、Spearman −0.110/0.131、flipped 无方向偏好（−0.142/0.137）——与归档方法2「一致弱负 ρ=−0.126±0.019」模式同向（单例不外推）。
- 产物：`outputs/10-21-ec-globaltau-v5main/run_01/`（subject_0.json 全 07-04 schema + surrogate 块、refs_corr_s12.json）；`run_01_rerun/`（S3 复跑副本）。

### 结论

- 支持：07-04-global-tau 方案在 gpfm-mamba-v5-main 代理脑上端到端跑通，EC_pert 计算与 07-04 同构（AC1–AC5 兑现；S1 门禁 + S3 逐位 + schema 全集）；EMA+bf16 口径下阈值机制行为正常（近二元钳边、剂量单调）。
- 描述性并置：s12 与 GC refs 弱负相关，与 v5-main 系既有阴性判读同向；**不构成代理脑资格判定**。
- 待二次拍板：主跑规模（10 例 × 32 窗 ≈ 15 min / × 全量 1321 窗 ≈ 8–16 h 后台化）。
