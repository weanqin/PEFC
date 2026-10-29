# MODMA fold1-4 × 07-05 架构损失训练 + v4 网格 EC 汇总

## Goal

07-05 架构与损失（标准 graph-mexhat OCLatentFlow、stage-2 flow-only、无正则，复用旧库各折 stage-1 VAE）在 MODMA resting fold1-4 训练（50ep/batch64/patience10/lr1e-3/seed42），随后 v4 方法 EC（sigma [0.01,0.48]×24、sigma_ref 0.3、q 0.5）逐折计算并汇总（fold1: 10 例、fold2-4: 11 例）

## 用户决策记录（consent）

- 2026-10-29 用户下达完整工作令（原文见 consent.md），明确：07-05 架构+损失、逐字超参（50/64/10/1e-3）、EC 超参（[0.01,0.48]×24 / σ_ref 0.3 / q 0.5）、顺序 fold1-4、汇总输出。用户供全规格的明确执行令 = 任务同意凭据。

## Requirements

- R1 训练：新增 `models/latent-flow-v1/train_modma_0705.py`（07-05 口径：标准 graph-mexhat 构建——无 exit-GNN/无 symw/无正则；stage-2 only，包内 `_run_stage2` flow-only 路径；VAE = 旧库 `fold{N}_oc_latent_flow_vae_best.pt` 只读；数据 = 包内 MODMA 管线 500窗/obs375/fc125/逐被试 z-score/5折 seed42/5 例 held-out val）；超参 50ep/batch64/patience10/lr1e-3/seed42；包体零改动。
- R2 EC：`ec_modma_symw_v4.py` 增 `--arch {exitgnn-symw, std-mexhat}` 开关（默认 exitgnn-symw = 旧行为不变）；fold1-4 用 std-mexhat + sigma [0.01,0.48]×24 / σ_ref 0.3 / q 0.5 / 32 窗 / seed42 / conn v4；fold1: 10 例、fold2-4: 11 例。
- R3 汇总：各 fold 训练指标（best val loss@ep、早停位）+ EC 指标（τ 分布、钳边型态、P(resp>τ)=0.500 构造性校验、源失败数）总表；产物只落 `outputs/10-29-modma-0705-f1234/`。
- R4 记录归档：experiment.md 逐 Run + summary + 归档。

## Acceptance Criteria

- [ ] AC1 fold1-4 训练各出 config/history/ckpt 三件套（或早停），best val loss 登记
- [ ] AC2 各 fold EC 全例成功（fold1 10/10、fold2-4 11/11），P(resp>τ)=0.500 逐例成立，payload 含 v4 网格 hparams
- [ ] AC3 跨 fold 汇总表落档（summary json）
- [ ] AC4 experiment.md + summary + 归档

## 边界与如实声明

- 现行包代码为 07-02 后形态：W 自由非对称（07-05 原始训练时旧库强制 W 对称）——「架构+损失组成」对齐，W 对称机制这一轴不对齐且现路径不可达（`--sym-w` 硬依赖 unfreeze-decoder），已向用户登记过（10-28 会话）。
- bf16 AMP 非逐位确定；各折为新训实例，无 07-05 锚点可对拍（07-05 仅有 fold0）。
- EC 网格 [0.01,0.48]×24 为用户指定的 v4 网格，非 07-05 MODMA 家规 [0.01,0.3]×16——跨口径不可与 07-05 fold0 复现值直接并置。
- 不做 GC 相关（用户未令；且 GC 现算基线仅 fold0）。

## 子研究路由（1.0b / 1.0c）

- **分类结果**：`study1`（EEG 代理脑变体训练 + 归因扰动 EC）
- **依据**：代理脑候选线的折扩展训练评估，非新 EC 算法。
- **新路径/方法检测（1.0c）**：`已有路径：latent-flow-v1 变体层（train_modma_symw / ec_modma_symw_v4 先例同层）`
- **research 产物**：`无`
