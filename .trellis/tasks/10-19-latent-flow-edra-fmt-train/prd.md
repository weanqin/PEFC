# latent-flow-v1 × edra_fmt 两阶段训练（stage1 VAE → stage2 潜空间 CFM 流匹配）

## Goal

以 latent-flow-v1 变体在 edra_fmt 数据上执行**独立新 run** 的双阶段训练：stage1 单独训练 VAE，stage2 在其潜空间上训练 CFM 流匹配（禁用 gpfm），数据划分与时间窗严格复用 edra_fmt 上一轮已确认设定，两阶段全过程可复现落档，不覆盖任何既有实验产物。

## Requirements

### R1 独立新 run 与产物隔离

- 全部新产物（检查点、配置、日志、训练 history）落 `/root/npi/outputs/10-19-latent-flow-edra-fmt-train/run_01/`。
- **禁写**：既有 outputs 各任务目录（`gpfm-mamba-f0/`、`gpfm-mamba-fmt-f0/`、`bivar-gc-ec-fmt*/`、`latent-flow-port/` 等）、gpfm 各变体目录及其 splits、`/root/autodl-tmp/edra_processed_fmt/` 本体、`/root/autodl-tmp/latent_flow/` 源库（只读）。
- **D2 待拍板豁免项（仅 stage2 选 mexhat 时相关）**：若用户拍板 geometry 落数据根，则允许在 `/root/autodl-tmp/edra_processed_fmt/` **新增 `geometry_edra_fmt.pt` 一个文件**（零覆盖、可独立删除）——此为对上述禁写条款的唯一显式豁免，未经拍板不生效（默认替代案：落 run_01 经 env 覆盖路径）。
- 包内 `latent_flow/checkpoints/`（默认 CHECKPOINT_DIR）不落任何新检查点；包内唯一允许的新增写入为 stage2 history JSON 的 `logs/edra_fmt/` 副本（经 `--save-path` 已重定向 ckpt；history 为训练器内置行为），变体级 `.gitignore` 补 `logs/` 排除。

### R2 数据配置严格沿用上一轮设定（不重新划分、不重新生成）

- **数据集**：edra_fmt（`/root/autodl-tmp/edra_processed_fmt/`，48 例 HC 24 / MDD 24，npz `data (62, 60001)` @250 Hz，契约 `.trellis/spec/data/edra-fmt.md`）。
- **划分**：只读复用固化 manifest `/root/npi/models/gpfm-mamba-v1-fmt/splits/fold0_status_ok_seed42_val4.json`（fold0，seed 42，train 34 / val 4 / test 10，与 v4a-fmt 拷贝逐字节一致）。**不调用** latent_flow 的 `SubjectFoldSplitter` 重新划分；入口断言 48 例全找到、train/val/test 计数 34/4/10、test 列表 = `[10,11,12,17,24,28,30,44,61,69]`、组别与 manifest `groups` 一致。
- **时间窗**：`[past=384, future=128]`、stride 64，边界规则「从 0 起、不补齐、丢尾」——与 fmt 契约每例 930 窗（(60001−512)//64+1）一致，入口断言每例窗数 = 930。
- **归一化**：逐通道 mean/std 由**训练折 34 例**估计（float64 累加），val/test 复用同一统计（防泄漏口径，与 gpfm 上一轮一致；走 latent_flow 自带 `norm_stats` 严格 held-out 路径）。

### R3 双阶段模型与阶段顺序（用户硬性约束）

- **Stage 1**：单独训练 VAE（encoder + proj + decoder，KL 退火），保存该阶段检查点与配置。
- **Stage 2**：加载 stage-1 检查点（非 flow 键严格匹配），冻结 VAE，训练**潜空间 CFM 流匹配**（OCLatentFlow 流：t~U(0,1) 线性插值概率路径 + 速度场 + Euler 采样，经冻结解码器的 fc 信号损失端到端优化）。
- **禁用 gpfm**：第二阶段模型不得使用 gpfm，不得将 gpfm 作为第二阶段模型；实现与运行全链路不得 import `models/gpfm-mamba*` 任何模块。
- Stage-2 流变体（dense / mexhat-graph）为 1.4 评审决策项，两者均为潜空间 CFM，均满足本约束。

### R4 可复现记录

- 两阶段分别落盘：完整配置 JSON（含超参/窗口参数/manifest 路径与内容摘要/seed/git 提交）、训练日志、检查点路径、训练 history。
- `experiment.md` Run 块按四区契约登记（目的与假设/配置/结果/偏差与结论）。
- 训练入口固定 seed（42），后台 nohup 运行，日志随 run 目录。

### R5 实现一致性核对（用户要求）

- 全部实现完成后逐项核对：阶段顺序（先 1 后 2，stage2 实际加载 stage1 产物文件路径）、模型类型（stage2 = 潜空间 CFM，VAE 冻结、trainable 参数 = flow 部分）、无 gpfm 依赖（import 扫描零命中）。核对结论写入 experiment.md。

## Acceptance Criteria

- [x] **AC1 隔离**：run_01 之外零新产物（`git status` 干净 + 输出目录清单核对）；既有目录/文件零改动（gpfm splits、旧 outputs、源库、fmt 数据本体）。（trellis-check PASS：唯一数据根新增 = D2 豁免 geometry_edra_fmt.pt）
- [x] **AC2 数据口径**：manifest 只读复用且断言全过（34/4/10、test 列表、组别、48 例、930 窗/例、训练折统计归一化）——断言代码在入口硬编码，失败即拒跑。（断言实测输出见两 stage 日志首部）
- [x] **AC3 Stage1 完成**：VAE 训练完成（早停或达 epoch 上限），`fold0_edra_fmt_vae_best.pt` + stage1 配置 JSON + 日志落 run_01；best val loss 与 best epoch 登记。（50/50 epoch，best val 1.5526）
- [x] **AC4 Stage2 完成**：CFM 流训练完成，`fold0_edra_fmt_flow_best*.pt` + stage2 配置 JSON + 日志 + history 落 run_01；日志含「加载 stage1 ckpt 路径 + 非 flow 键零失配 + VAE 冻结后 trainable 参数数」三行证据。（49/50 epoch best 0.5977；路径证据在 config JSON cli.vae_ckpt + check 复核附注：日志无显式路径打印行，证据链完整）
- [x] **AC5 可复现**：experiment.md Run 块齐全（两阶段配置/划分引用/窗口参数/ckpt 与日志路径/耗时）；配置 JSON 与实际 CLI 参数一致。（check 独立重算 manifest sha/git HEAD 一致）
- [x] **AC6 一致性核对**：R5 三项核对逐条通过并登记（阶段顺序 / 模型类型 / 无 gpfm import）。
- [x] **AC7 回归不破坏**：包内 additive 改动后全量 pytest 维持冻结口径 67 passed / 0 failed / 17 skipped；新增文件 LF 行尾；MODMA/EDRA 旧线加载路径行为不变。（check 独立复跑 67P/17S）

### 用户决策记录（2026-10-19，1.4 评审问答）

1. **D0 CFM 口径 = 确认旧库口径**：stage-2 为潜空间 CFM 构造（线性插值路径 + 速度场 + Euler 采样），无显式速度 MSE 项、经冻结解码器信号损失隐式优化——满足「潜空间下的 CFM 变体」约束；教科书式 velocity-MSE 不要求。
2. **D1 stage-2 流变体 = mexhat-graph**：证据基座加注已知悉（旧库 500 Hz edra EC 线迁移假设，本 run 不检验）。
3. **D2 geometry 落点 = 数据根豁免案**：prd R1 豁免条款生效——允许在 `/root/autodl-tmp/edra_processed_fmt/` 新增 `geometry_edra_fmt.pt` 单文件（零覆盖、可独立删除）。
4. **批准实施**：task.py start → S1–S5 全清单执行（冒烟门禁后正式两阶段 nohup 训练）。
5. **D4 geometry 基座 = chan_coords_62.npy（2026-10-19，主会话真实拍板）**：S2.2 判定门如设计触发（MNE 模板 vs npy 16/62 通道 cos<0.99，min 0.9766；功能等价——距离矩阵相关 0.9929、mexhat 邻接相关 0.9967）。**过程更正**：implement 子代理曾声称「已获用户拍板改 npy 基座」，该声称未经真实用户交互、无效；主会话抽查核实数字属实后将判定门结论提交用户，用户选 npy 基座（数据集第一方坐标、行序硬证明 3.6nm 级对齐、与 MNE 功能差异可忽略）。geometry_edra_fmt.pt 维持 npy 基座版本，MNE 差异 info-only 登记。同轮拍板：S4 立即开跑。

## Notes

- 代理脑资格边界：本任务产物**不自动继承** surrogate-brain 合格性（按 `.trellis/spec/research/surrogate-brain-qualification.md`，固定 θ\* 前须另走三项检验）；本任务交付 = 可复现训练完成的 θ 候选。
- 训练完成 ≠ 代理脑有效；本任务不含评估/合格性判定（属后续任务）。
- 复杂任务：`design.md`（技术设计）与 `implement.md`（执行清单）另附。

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

> 任务创建时由 hook 自动播种。规划阶段必须填写以下四项；全部落定前不得进入 1.4 start 门禁（见 `.trellis/workflow.md` Phase 1.0b / 1.0c）。

- **分类结果**：`study1` —— 本任务为 EEG 代理脑（Study I）候选路径 latent-flow-v1 的代理生成器主体训练（X_past→X_future 条件生成），不涉及扰动/EC 推断算法。
- **依据**：用户指令明确「训练 latent_flow_v1 变体两阶段（VAE → 潜空间 CFM 流匹配）」——纯 Study I 代理脑训练；σ 二分归因不在本任务范围。`task.py set-scope study1` 已执行（2026-10-19）。
- **新路径/方法检测（1.0c）**：`已有路径：latent-flow（OC-LatentFlow）` —— `.trellis/research/context.md` 技术路线表已登记「Study I 候选：latent-flow」条目（2026-10-18 移植任务回填）；edra_fmt 为已登记数据集契约（`.trellis/spec/data/edra-fmt.md`）。本任务 = 该登记条目的数据接入与训练执行，正是移植任务 adaptation study（`10-18-latent-flow-port/research/latent-flow-adaptation.md` §4.1/§5 验证顺序④）预留的后续任务，无新方法引入，不触发新 research。
- **research 产物**：`无（复用 10-18-latent-flow-port/research/latent-flow-adaptation.md）` —— 其 §4.1 已起草 npz shim 方案（Descriptor + reader + geometry 生成），本任务 design 直接落实。
