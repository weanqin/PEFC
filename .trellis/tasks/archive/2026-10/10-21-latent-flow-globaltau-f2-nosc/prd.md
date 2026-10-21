# latent-flow global-tau fold2 nosc 本机重跑（07-04 方案首跑存证）

## Goal

用 07-04-global-tau-sigma 方案（global-quantile-tau σ\* 归因 EC 推断）在本机以 fold2 nosc（MODMA SOTA）ckpt 重跑 1 例 × 32 窗，产物落 outputs 随档，诊断如实登记。

## Background

- 前任务 10-21-latent-flow-modma-nosc-check（2026-10-21 归档）已完成：geometry.pt 重建（MNE GSN-129 模板，逐位验证命中）、fold2 nosc 加载断言（mexhat-graph/dataset=modma、buffer 逐位一致）、probe 首跑。本任务为 global-tau EC 推断主口径的本机首跑。
- 07-04 方案 = 脚本默认参数：q=0.5（全局中位 τ）、σ_ref=0.3、log 网格 [0.01,0.3]×24、SignalProbeSite 信号空间 + biased noise 方向符号、n_steps=10、batch=32、splitter seed=42。
- 旧库历史基线（仅描述性参照）：fold0 nosc 10 例 × 32 窗 @ 17.7 min/例（5070 Ti），σ\* 近二元（@min 44.4% / @max 47.9% 钳边）。
- 脚本依赖已核验：自带 N_CHANNELS 定义（无 probe 的 NameError 缺陷）；geometry.pt 在位解除 mexhat 构建堵点。

## Requirements

- R1 运行配置：`run.py run_oc_connectivity_global_tau --fold 2 --checkpoint <fold2 nosc> --subjects 1 --n-trials 32 --output-dir <abs>`；其余参数一律 07-04 默认（用户拍板 2026-10-21）。
- R2 产物：per-subject JSON 落 `outputs/10-21-latent-flow-globaltau-f2-nosc/run_01/`。
- R3 登记：tau_global、边界钳位比例（@min/@max）、耗时、响应统计入 experiment.md；与旧库 fold0 历史例仅描述性并置（不同 fold/被试，**不做对比交付**）。
- R4 零仓库代码改动（包体零逻辑改动纪律延续）。

## Acceptance Criteria

- [ ] AC1 run 成功完成：JSON 落盘、`n_sources_succeeded=128`、`n_source_failures=0`
- [ ] AC2 诊断指标（tau_global / @min/@max 钳位 / elapsed_min / noise_model）如实登记于 experiment.md
- [ ] AC3 零仓库代码改动；新增工件仅 outputs 产物

## 边界与不变式

- 单例首跑存证，不外推；不构成代理脑资格（三项检验另立任务）。
- 不做 10 例对标旧库（如需另行拍板）；不做 fold 间/变体间对比。
- MODMA 数据与源库只读；输出仅落 outputs（gitignored）。

## 子研究路由（1.0b / 1.0c）

- **分类结果**：`study1`（EEG 代理脑，框架 Study I）
- **依据**：latent-flow（OC-LatentFlow）为 Study I 候选线，其 bisection σ 扰动归因→有向 EC 属该候选的组成部分；本任务为其既定方案（07-04）的本机重跑。
- **新路径/方法检测（1.0c）**：`已有路径：latent-flow（技术路线表 Study I 候选行；07-04-global-tau-sigma 为旧库既定方案，非新方法）`
- **research 产物**：`无`
