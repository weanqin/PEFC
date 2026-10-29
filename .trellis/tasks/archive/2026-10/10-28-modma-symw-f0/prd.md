# MODMA fold0 × symw-exit-GNN 变体训练 + v4 EC + GC 对比

## Goal

把 mild 线 10-25-symw-gnn-exit 变体（exit-GNN 流 + SymWAdam + decoder 联训 + anti-uniform λ=0.2，lr1e-3/150ep/b64/p30/seed42）移植到 MODMA resting fold0 训练（复用旧库 fold0 stage-1 VAE + 包内 MODMA 管线）；用 v4 conn 公式算 10 例 test 逐被试 EC（07-05 参数口径）；与 10-27 现算 GC（AIC 口径，经旧库对拍）做逐被试 + 组平均相关并汇报

## 用户决策记录（consent）

- 2026-10-28 会话内，用户对「在 MODMA resting fold0 上用 mild 线 symw-exit-GNN 变体训练 + v4 EC + GC 对比」的口径逐项下达（数据/模型、v4 EC、GC 对比、两级相关汇报），并在 Trellis 编辑门拦截后对「立轻量任务继续执行」明确回复**同意**（本文件即该同意的落档）。
- 执行口径：stage-2 only，复用旧库 `fold0_oc_latent_flow_vae_best.pt`（与 07-05 线同 VAE 基座）；EC 参数 = 07-05 家规（q=0.5/σ_ref=0.3/grid log[0.01,0.3]×16/32 窗/seed42），conn 公式 = v4（−(lnσ*+½ln2πe)+5/7）。

## Requirements

- R1 变体移植：新增变体层文件 `models/latent-flow-v1/train_modma_symw.py`（stage-2 训练入口），口径与 mild `run_symw_fold0` configs/stage2_config.json 逐字段对齐（exit-GNN 流替换 + SymWAdam W←(W+Wᵀ)/2 + decoder 联训 encoder/proj 冻结 + anti-uniform λ=0.2 + lr1e-3/150ep/b64/p30/seed42）；数据 = 包内 MODMA 管线零复制（500 窗/step250/obs375/fc125、逐被试 z-score、SubjectFoldSplitter 5 折 seed42、train 折内 5 例 held-out val）；复用 `train_mild_depress._run_stage2_mild`（loader 无关）；包体零改动。
- R2 EC driver：新增 `models/latent-flow-v1/ec_modma_symw_v4.py`，main 模式镜像 `ec_modma_globaltau_unbiased` 主路径（无偏 μ=0、两阶段 global-τ、stage-2 每源单 autocast 域），差异仅：symw-exit-GNN ckpt 装载（训练侧同一构建函数）+ conn v4；gccorr 模式对 `outputs/10-27-gc-fold0-aic-anchor/gc_aic/gc`（10 例 AIC 口径 GC，subject_file+connectivity_matrix 契约）做 offdiag 16256 对 |EC|/raw × aligned/flipped × Spearman/Pearson，含组均值矩阵相关。
- R3 产物隔离：只落 `outputs/10-28-modma-symw-f0/`（run_01 训练 / ec_main / gccorr），不触碰旧任务树与旧库（VAE 只读）。
- R4 汇报：逐被试 + 组平均两级相关数值、与 mild 线同型对照表、两方法差异说明、边界声明（描述性并置，不构成代理脑资格判定）。

## Acceptance Criteria

- [ ] AC1 训练完成：150 ep 跑满或 patience 早停，config/history/ckpt 三件套落 run_01，payload 含 provenance（cli/pinned/vae sha/git head）
- [ ] AC2 EC 10/10 例成功（源失败=0），payload 含 method/hparams/conn 公式 v4/noise 契约，P(resp>τ)=0.500 构造性校验逐例成立
- [ ] AC3 gccorr 双级读数落档：per_subject + group + group_mean（Spearman/Pearson × aligned/flipped）
- [ ] AC4 experiment.md 逐 Run 记录 + summary 交付 + 归档

## 边界

- bf16 AMP 非逐位确定（先例登记）；本任务为新变体首训，无对拍锚点，结果为新口径登记。
- GC = 10-27 现算 AIC 口径（1–125Hz/max8/nf65/fs250/全长），与 07-05 原始 GC 锚点同口径（9 例验证集差 +0.002 已验）。
- 描述性并置不构成代理脑资格判定；不外推。

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

- **分类结果**：`study1`（EEG 代理脑：latent-flow 变体训练 + 归因扰动 EC，框架 Study I 候选线）
- **依据**：任务主体 = 训练代理脑变体并做 EC×GC 观测对照，属代理脑资格线的变体扫描，非新 EC 推断算法。
- **新路径/方法检测（1.0c）**：`已有路径：latent-flow-v1 变体层（10-25-symw / 10-24-exit-gnn / 10-28-no-reg-flow-only 先例同层）`——无新方法路径，research 不触发。
- **research 产物**：`无`
