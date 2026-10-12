# GP先验流匹配Mamba代理脑训练（edra fold0）

## Goal

基于 Mamba 骨干（d_model=16）构建 GP 条件先验流匹配（Flow Matching）模型，在 edra 数据集 fold0（单 seed）上训练 Study I 代理脑：给定过去 384 采样点（62 通道），条件生成未来 128 采样点。训练损失为 mae+cons+tv+corr 复合损失（初始权重 1/1/0.1/0.1），不使用显式速度预测损失项。

## 背景与定位

- 方法来源：TSFlow（Kollovieh et al., **ICLR 2025**, arXiv:2410.03024）的 **GPR 条件先验**：`q0(x0^f | y^p) = N(Σ_fp Σ_pp⁻¹ y^p, Σ_ff − Σ_fp Σ_pp⁻¹ Σ_pf)`，past 部分取 δ(y^p)；核在 RBF / OU / 周期核中选（+白噪声项，固定超参、不拟合数据）。
- 本任务是 Study I（EEG 代理脑）模型选型的第一条实现路径，属技术路线表**新路径** `gpfm-mamba`（1.0c 已触发适配研究，见路由区块）。
- 对论文的偏离（用户指定，不可改）：① 骨干由 DiffWave+S4 换为 Mamba（d_model=16）；② 去掉速度回归损失，loss 直接对模型输出计算复合损失。
- **范围边界**：本任务只做 fold0 单 seed 训练 + 前置验证落盘；5-fold 完整训练须待用户明确同意（合格性门禁），固定 θ* 不在本任务范围。

## Requirements

- **R1 数据与划分**：edra v4（`/root/autodl-tmp/edra_process_v4/`，62 通道 500 Hz），5-fold 受试者级分层划分（MDD/HC），固定 fold 标识与划分 seed，全程沿用 fold0；**受试者过滤口径（2026-10-12 用户门禁拍板 b）：manifest `status=='ok'`（46 例 HC24/MDD22）**；**划分固化**（用户要求）：fold0 划分 manifest 落 `models/gpfm-mamba-v1/splits/fold0_status_ok_seed42.json`，此后构建校验一致、漂移即报错；窗口 `[past=384, future=128]`；全部预处理（受试者筛选、z-score、窗切分）在 dataloader 内完成，z-score 统计仅用训练折估计；筛选口径登记 experiment.md。
- **R2 GPR 条件先验（时空可分核，用户 2026-10-12 指定）**：联合先验 `Cov(x0[c,t], x0[c',t']) = K_s(c,c')·K_t(t,t')`；GPR 条件后验按闭式解实现（后验均值算子 A 与逐通道情形一致、后验协方差保持可分 `K_s ⊗ K'_{t,f|p}`）；K_t 默认 OU（γ=1，τ_i=i·π/p，p=50，iso=1e-2 折叠进时间核）；**K_s 已定：Mexican-hat 的 Ricker 形式——采用三维正定形式 `(1−d²/(3σ²))e^{−d²/2σ²}`（严格 1D Ricker 在实测电极点集上非 PSD，已数值否决）**，σ 默认 **0.080 m**（用户定；邻位 +0.89、过零 0.139 m 在阵列内、负瓣 −0.05、有效秩 7.5/62）+ jitter_s=1e-3，σ 扫描 {0.045, 0.080, 0.137} 入诊断步；past 全观测 ⇒ A、L_s、L_t 均为固定算子预计算；偏离论文逐通道独立口径之处显式登记。
- **R3 模型**：Mamba 骨干，`d_model=16`（其余超参在 design.md 定并登记）；条件信息（过去 384 点序列）以序列前缀方式输入；模型输出未来段的速度场 `v_θ(x_t, t, X_past)`，时间嵌入采用正弦编码。**官方 mamba-ssm 路线（已定并完成：mamba_ssm 2.3.1 + causal_conv1d 1.7.0，2026-10-12 用户委派安装、本机验证通过）。**
- **R4 训练目标（用户定稿口径）**：FM 路径 `x_t = t·x1 + (1−t)·x0 + σ_min·η`（σ_min=1e-4），**t ~ logit-normal：t = sigmoid(z)，z~N(−0.8, 0.8²)**（不用 U[0,1]；中位 t≈0.31，clamp [1e-4, 1−1e-4]），`x0 ~ 时空可分 GPR 条件先验`；**训练期生成=一步映射** `x̂1 = x_t + (1−t)·v_θ`（训练全程不做 ODE 积分；多步 Euler 仅用于训练后采样）；**不使用显式速度预测损失项**；`loss = 1·mae + 1·cons + 0.1·tv + 0.1·corr`（权重可配置），四项均作用于 x̂1，定义域为未来 128 列：
  - `mae = mean_{b,c,τ} |x̂1 − x1|`（逐元素全均值）
  - `cons = mean_{b,c} [ |std_τ(x̂1) − std_τ(x1)| + |mean_τ(x̂1) − mean_τ(x1)| ]`（std ddof=1）
  - `tv = mean_{b,c,τ} |x̂1[τ+1] − x̂1[τ]|`（纯平滑惩罚，不对照 GT 差分；口径变更须登记）
  - `corr = mean_{b,c} [1 − r_bc]`，r 为和式 Pearson（`S_xy/(√(S_xx·S_yy)+eps)`，eps=1e-6，r 截断 [−1,1]，近平坦通道置 r=0）
  - **best checkpoint 判据：内部验证总 loss**（固定 t 网格 {0.1,0.3,0.5,0.7,0.9}+固定 ε 确定性复算，跨 epoch 可比）
- **R5 采样**：训练完成后用 Euler ODE 求解器从 GPR 先验出发积分生成未来段（步数可配置，默认 16）；支持多条独立样本采样。
- **R6 评估与门禁**：训练后按 `.trellis/spec/research/surrogate-brain-qualification.md` 执行 **fold0 单 seed 前置验证**：指标 1（GC 连接一致性：Pearson/Spearman + 置信度）、指标 2（人工视觉检验，用户判定）、指标 3（PSD 相似性），数值与图落盘 experiment.md；**未经用户同意不得启动 5-fold**；预测质量（Q1）与代理脑有效性（Q2）结论分开陈述。
- **R7 产物与可复现**：产物落 `/root/npi/outputs/gpfm-mamba-f0/run_<id>/`（outputs 根用户已定：仓库内 `/root/npi/outputs/<任务名>/`，`.gitignore` 排除）；checkpoint 命名 `变体名+用途+标识`（best/last），保留策略先声明；每个 Run 在 `{TASK_DIR}/experiment.md` 登记「目的/配置（含 seed、指标口径）/结果/结论」；代码版本 commit 哈希入 Run 块。
- **R8 环境纪律**：默认零新增依赖（评估代码 numpy-only 实现）；mamba-ssm 安装由外部会话执行（用户 2026-10-12 指定），本任务不得安装任何包；如确需其他安装须先获用户同意（环境契约）。
- **R9 训练/推理监控与断点续传（用户 2026-10-12 新增/更新）**：① 训练每 step 输出进度行（epoch/step、四项损失、it/s、ETA），每 epoch 追加机器可读 `metrics.jsonl`；推理/评估各阶段（采样批、GC、PSD）打印进度与 ETA，**且评估侧 log/结构化指标一并留存**（`logs/eval.log` + `eval_metrics.jsonl`）便于事后分析；② **每个 epoch 结束后**评估内部验证**总 loss**（固定 t 网格确定性复算）并更新 `best` checkpoint（创新优时覆写），同时无条件覆写 `last`；③ **断点续传**：`last` 保存全量续传状态（model+EMA+optimizer+scheduler+epoch/step+RNG 状态+best 指标），`--resume` 从中断处继续；续跑视为同一 Run 的延续（artifacts-management 口径，不另开 Run）；④ **1-epoch 基准测试（含用户确认门禁）**：全量训练折 1 epoch，记录 peak 显存（max_memory_allocated）、epoch 墙钟时间、it/s 与 100-epoch 外推时长，登记 experiment.md 与 run 目录；**基准摘要（显存/时长/吞吐/四项损失走势/t 分桶 sanity）汇报用户，由用户确认进入 fold0 单 seed 完整训练、或先根据基准结果优化模型——未获确认不得启动完整训练，决策记入 experiment.md**。

## Acceptance Criteria

- [ ] fold0 dataloader 通过单测：受试者级无跨折泄漏、分层平衡、z-score 统计仅来自训练折、输出形状 `X_past∈R^{C×384}` / `X_future∈R^{C×128}`（C=62）
- [ ] GPR 先验模块通过单测：`A=Σ_fp Σ_pp⁻¹` 与 numpy 直接求解一致（容差内）；从 `N(μ, Σ_{f|p})` 采样的经验均值/协方差逼近解析值；核矩阵 PSD（含 jitter 处理）
- [ ] Mamba 模块通过单测：前向形状正确、因果性（未来 token 不影响过去 token 的输出）、梯度有限
- [ ] 复合损失四项通过手工小张量单测（与独立实现的参考值一致）
- [ ] fold0 单 seed 完整训练跑通：loss 曲线落盘、无 NaN、训练日志与 resolved config 落 `outputs/<变体>/`
- [ ] **R9 落实**：每 step 进度行含 ETA 与四项损失；`metrics.jsonl` 每 epoch 追加无缺口；每 epoch 后 best/last checkpoint 就位（**best 按内部验证总 loss 更新，固定 t 网格确定性口径**）；评估侧 log/指标留存 run 目录；**断点续传冒烟通过**（中途 kill → `--resume` 续跑 → loss/指标曲线连续、RNG 与 optimizer/scheduler 状态恢复）
- [ ] **1-epoch 基准**：peak 显存、epoch 墙钟时间、it/s、100-epoch 外推时长已记录（experiment.md + run 目录）；**基准后用户确认门禁已执行**（摘要汇报 + 用户决策留痕 experiment.md，未确认未启动完整训练）
- [ ] 采样与生成可视化：测试集若干窗口的多样本轨迹图（真实 vs 生成，经统一 visualizer）
- [ ] 前置验证三项指标落盘 experiment.md（GC 数值+图、PSD 对比图、生成样本图），Run 块四要素齐全
- [ ] 未启动 5-fold；未固定 θ*；全部产物在声明的变体目录内，无跨变体写入
- [ ] experiment.md、`.trellis/research/progress.md` 已同步本节点结论

## 待用户拍板（规划期 · 2026-10-12 第二次更新）

1. ~~K_s 空间核~~ **已定**：Mexican-hat 的 Ricker 形式（三维正定 `(1−d²/(3σ²))e^{−d²/2σ²}`，**σ=0.080 m** + jitter_s=1e-3，用户定；严格 1D Ricker 实测非 PSD 已否决）；identity 消融臂默认不跑（需要归因时建薄变体）
2. ~~GPR 核方向~~ **已定**：时空可分 `K_s ⊗ K_t`（用户指定）；K_t 默认 OU（γ=1, p=50, iso=1e-2）
3. ~~Mamba 路线~~ **已定并完成**：官方 mamba-ssm 2.3.1 + causal_conv1d 1.7.0（已装已验证）
4. ~~GC/PSD 口径~~ **已定**：future-only
5. ~~outputs 根~~ **已定**：`/root/npi/outputs/<任务名>/`
6. （默认口径确认）tv 为纯平滑惩罚、corr 逐通道口径、`x̂1 = x_t+(1−t)v_θ` 参数化、σ_s=0.080 默认值——如需调整请在启动确认时一并提出
7. ~~启动确认~~ **已确认（2026-10-12 门禁决策）**：进入 §5 完整训练；数据口径 (b) status_ok 46 例；划分固化；EMA 0.999（AI 建议值，向用户解释后默认采纳，如需 0.9999 属口径变更重跑登记）

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

> 任务创建时由 hook 自动播种。规划阶段必须填写以下四项；全部落定前不得进入 1.4 start 门禁（见 `.trellis/workflow.md` Phase 1.0b / 1.0c）。

- **分类结果**：`study1` —— EEG 代理脑：训练 X_past→X_future 条件生成模型属框架 Study I（模型选型与训练），已 `task.py set-scope study1`
- **依据**：任务核心是 Study I 代理脑的训练路径实现（条件未来生成），不涉及 Study II 扰动推断要素
- **新路径/方法检测（1.0c）**：`新路径：gpfm-mamba → research 已触发` —— 「GP 条件先验流匹配（TSFlow）× Mamba」未见于 `.trellis/research/context.md` 技术路线表（Study I 此前无已登记模型变体），已派发 trellis-research 适配研究
- **research 产物**：`research/gpfm-mamba-adaptation.md`（适配结论 / 风险 / 可执行方案）
