# 实验日志 — gpfm-mamba-edra-f0

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 000 —— fold0 单 seed 完整训练（预登记 · 未启动 · 用户门禁）

- **状态**：未启动。implement.md §5 完整训练须待 4.5 基准摘要经用户确认放行后方可执行；本块为 Run-0 预登记（implement.md 0.2），启动时在本块补齐 resolved config 全量要素与 run_id。
- 目的：fold0 单 seed 前置验证训练（Study I 代理脑，gpfm-mamba-v1 变体）。
- 配置（已定，跑前 dump resolved config 到 run 目录）：
  - 划分：受试者级 5-fold 分层（MDD/HC）split_seed=42，fold0；qc_pass=True 过滤（实测 **13 例：HC 5 / MDD 8**）；内部验证 2 例（HC 1 / MDD 1，val_subject_seed=42）；训练折 8 例 10568 窗 / 内部验证 2 例 2642 窗 / 测试 3 例 3963 窗（stride=64，每例 1321 窗）
  - z-score：逐通道，仅训练折 8 例估计
  - 先验：K_t=OU γ=1 τ=i·π/p p=50 iso=1e-2（折叠进时间核）pp_jitter=1e-4；K_s=Ricker3D σ=0.080 m + jitter_s=1e-3
  - 模型：Mamba d_model=16 d_state=16 d_conv=4 expand=2 ×4 层 + 预 RMSNorm 残差；EMA 0.9999
  - 损失：1·mae+1·cons+0.1·tv+0.1·corr（future 128 列，一步生成 x̂1=x_t+(1−t)v_θ）
  - 训练：seed=1234、AdamW lr=1e-3 wd=1e-4 cosine、clip 0.5、batch 256、epochs 100；t=sigmoid(N(−0.8,0.8²)) clamp [1e-4,1−1e-4]
  - best 判据：内部验证总 loss（固定 t 网格 {0.1,0.3,0.5,0.7,0.9}+固定 eps=seed 2026 确定性复算）
  - 主指标（§6）：测试折 MAE（ODE 终点样本均值口径）

## Run 001 —— 训练折核诊断复跑（2026-10-12）

### 目的与假设

- 在 fold0 训练折**正式 z-score 口径**下复跑 GPR 核诊断（适配研究 Q1 仅 2 被试、逐被试 z-score 代理，为指示性）：确认 OU 时间核数值稳定（R²(μ)/post_std/条件数）与 Ricker3D 空间核 σ=0.080 默认档结论（min_eig/邻位相关/负瓣/有效秩）；若诊断推翻默认档，改档属口径变更须登记。

### 配置

- 代码：`models/gpfm-mamba-v1/diagnose_kernel.py`（本任务首建，commit 见 Run 块结论补记）
- 参数：数据 = fold0 训练折 8 例（qc_pass=True、z-score 仅训练折、窗 384/128 stride 64，正式管道）；诊断窗 = 8 例 × 40 窗（确定性等距抽样）；先验 = OU γ=1 p=50 iso=1e-2 pp_jitter=1e-4 + Ricker3D σ_s∈{0.045,0.080,0.137} jitter_s=1e-3；先验采样 400 条/窗（诊断专用 seed=1234）
- 环境：base conda @ 本地云服务器（.trellis/research/environment.md）
- 数据：`/root/autodl-tmp/edra_process_v4/`（只读）
- 命令：`python3 models/gpfm-mamba-v1/diagnose_kernel.py`

### 结果

- 时间侧（OU，fold0 训练折正式口径）：R²(μ)=+0.0495（适配研究指示值 0.01–0.03，量级一致 ≈0）；post_std mean 0.971（min 0.369 接缝处 / max 1.005）；Σ_pp 条件数 7.56e2（iso=1e-2 下 ~1e2–1e3，数值稳定）；GT future 粗度 0.127，先验样本粗度 0.30（≈2.4× GT，介于适配研究 iso=1e-4 的 1.5–2.2× 与 iso=1e-1 的 2.2–3.8× 之间，符合 iso=1e-2 折中预期）；GT future 窗内 std 0.833。
- 空间侧（Ricker3D 三档全 PSD + jitter 后 Cholesky 成功）：

| σ_s | min_eig | +jitter 条件数 | 邻位相关（解析/经验）| 负对占比 | 过零 vs 阵列 | 有效秩 exp(H)/Σλ/λmax |
|---|---|---|---|---|---|---|
| 0.045 | +9.92e-4 | 2.3e3 | +0.664 / +0.674 | 77.4% | 0.078 < 0.190 | 27.0 / 13.4 |
| **0.080（默认）** | **+3.98e-7** | **1.4e4** | **+0.881 / +0.874** | **37.1%** | 0.139 < 0.190 | 10.1 / 4.5 |
| 0.137 | +2.09e-10 | 3.3e4 | +0.958 / +0.948 | 0% | 0.237 > 0.190 | 4.3 / 1.9 |

- 产物：`.trellis/tasks/10-12-gpfm-mamba-edra-f0/research/kernel-re-diagnosis.md`（含电极几何：最近邻 mean 0.031 m、最大间距 0.190 m）

### 结论

- **支持（默认档维持）**：σ=0.080 档复现适配研究全部关键数值——min_eig +3.98e-7（真实余量）、邻位 +0.88、负瓣 −0.055 / 37%、+jitter 条件数 1.4e4（float64 预计算安全）、过零 0.139 m 在阵列内；时间侧 OU 数值稳定且后验展宽匹配 z-score 尺度。**不构成口径变更，σ=0.080 默认档维持。**
- 三档语义确证：0.045 强中心-周边（77% 负对）、0.080 平衡、0.137 全局相干（无负瓣、过零出阵列、有效秩 ~2–4）。
- 口径备注：design 所引「有效秩 7.5/62」定义未注明；本次报告 exp(H)=10.1 与 Σλ/λmax=4.5 两口径，排序与定性结论不变（已记入诊断 md）。
- 派生事项：无（无推翻默认档证据）。
