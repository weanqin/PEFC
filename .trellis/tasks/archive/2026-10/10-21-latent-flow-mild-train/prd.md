# latent-flow-v1 × Mild Depression 代理脑训练（1500 点 trial、x^p:x^f=3:1、5-fold）

## Goal

用 Mild Depression 数据集（`/root/autodl-tmp/Mild Depression/`，40 例 HC 20 / MDD 20、预切 30 trial/例、128ch GSN-128、fs=250 Hz、x_p=1125 / x_f=375，即 1500 点/trial、x^p:x^f=3:1）按 latent-flow-v1 两阶段范式（stage1 VAE → stage2 mexhat CFM 流）训练 EEG 代理脑 θ 候选。划分采用 5-fold（用户 2026-10-21 拍板），交付各折 ckpt + config + history 与 smoke 评估，全过程可复现落档、不覆盖任何既有产物。**不含**代理脑合格性三项检验（另立任务）。

## Requirements

### R1 独立新 run 与产物隔离

- 全部训练产物落 `/root/npi/outputs/10-21-latent-flow-mild-train/`（每折一子目录 `run_fold{0..4}/`，含 configs/checkpoints/logs + 任务级汇总表）。
- **禁写**：既有 outputs 各任务目录、gpfm 各变体目录及 splits、`/root/autodl-tmp/` 下既有数据集本体（modma/、edra*/、latent_flow 源库）。
- **待拍板豁免项（D1，同 edra_fmt 任务 D2 先例）**：若用户拍板 geometry 落数据根，则允许在 `/root/autodl-tmp/Mild Depression/` **新增 `geometry_mild.pt` 一个文件**（零覆盖、可独立删除）；未拍板前默认替代案 = 落 run 目录、wrapper 显式传路径。
- 包体（`latent_flow/` 子包内）**零改动**；新增文件全部落变体层（`models/latent-flow-v1/` 根）：训练入口 wrapper、geometry 构建脚本、5-fold manifest。stage2 history 的 `logs/mild/` 副本为训练器内置行为（经 `--save-path` 已重定向 ckpt），变体级 `.gitignore` 已排除 `logs/`。

### R2 数据口径（新数据契约，首次登记）

- **数据集**：Mild Depression（`/root/autodl-tmp/Mild Depression/`）——40 例（h_1..h_20 = HC，subject_id 21–40；m_1..m_20 = MDD，subject_id 1–20），每例 .mat 含 `x_p (30,128,1125)` + `x_f (30,128,375)` + label/label_code/subject_id/fs/trial 元数据；`EEG_ALL.mat (1200,128,1500)` 为合集源（本任务只读不消费，以带标签的逐例文件为准）。fs=250 Hz。
- **时间窗**：trial 预切、不重新切窗——每 trial 即一个 1500 点窗（obs=x^p 1125 + forecast=x^f 375，比例 3:1 由数据本身保证），入口硬断言 shape/fs/label/trial 数（每例 30 trial、每折窗数 = 30×例数）。
- **划分**：5-fold 分层（用户拍板）——MDD 20 + HC 20 各自 `sorted(glob)` 后独立 `RandomState(42).permutation` 洗牌（RNG 调用序列钉死）分 5 段，每折 test 8 例（4+4）；val 4 例 = 两群该折 test 段下一段（环形轮转）前 2 例，train 28 / val 4 / test 8。manifest 一次性生成、钉死落 `models/latent-flow-v1/splits/mild_5fold_seed42.json`（例文件钉绝对路径），训练入口只读复用并全量断言，不重新划分。
- **归一化**：逐通道 mean/std 由**训练折 28 例全部 trial（x_p+x_f 全长）**估计（float64 累加），val/test 复用同一统计（严格 held-out，同 edra_fmt 口径）。
- **数据契约落 spec**：新增 `.trellis/spec/data/mild-depression.md`（按 dataset-contract-template），后续合格性/EC 任务引用。

### R3 双阶段模型与阶段顺序（同 edra_fmt 先例的硬性约束）

- **Stage 1**：单独训练 VAE（encoder + proj + decoder，KL 退火），dense 构建无需 geometry。
- **Stage 2**：加载 stage-1 ckpt（非 flow 键严格匹配）、冻结 VAE，训练潜空间 CFM 流匹配（旧库 CFM 口径：线性插值路径 + 速度场 + Euler 采样，经冻结解码器信号损失隐式优化）；流变体 = **mexhat-graph**（geometry = MNE GSN-HydroCel-128 模板），`--dense-fallback` 保留为反切开关。
- 模型规格：OCLatentFlow(n_channels=128, latent_dim=64, t_lat=16, obs_len=1125, forecast_len=375, beta_kl=1e-4, use_fc_decoder=False)；训练循环零复制，import 复用包内 `_run_stage1/_run_stage2/Stage1Step/Stage2Step/_KLAnnealCallback`。
- 禁用 gpfm：全链路不得 import `models/gpfm-mamba*`。

### R4 可复现记录

- 每折每阶段落盘：完整 config JSON（超参/窗口参数/manifest 路径+SHA256/seed/git HEAD/归一化统计哈希）、训练日志（nohup）、ckpt 路径、history。
- `experiment.md` Run 块按四区契约登记（5 折可共块、逐折结果表 + 汇总统计）。
- seed 固定 42；每折两阶段顺序执行（stage1 完成才有 stage2）。

### R5 实现一致性核对

- 完成后逐项核对：阶段顺序（stage2 实际加载 stage1 产物路径）、模型类型（VAE 冻结、trainable=flow）、无 gpfm import、包体零改动（`git status` 下 `latent_flow/` 子包无 diff）、manifest 断言全过。结论写入 experiment.md。

## Acceptance Criteria

- [ ] **AC1 隔离**：任务 outputs 目录之外零新产物（数据根仅 D1 豁免单文件，若获批）；既有目录/文件零改动；`latent_flow/` 包体零 diff。
- [ ] **AC2 数据口径**：manifest 生成一次即钉死；训练入口断言全过（40 例 20/20、每例 30 trial、shape 1125/375、fs=250、逐折 28/4/8 无重叠、train 折统计归一化）——失败即拒跑。
- [ ] **AC3 Stage1 ×5 折完成**：各折 VAE ckpt + config + 日志 + history 落 `run_fold{0..4}/`；best val loss 与 best epoch 逐折登记。
- [ ] **AC4 Stage2 ×5 折完成**：各折 mexhat CFM 流 ckpt + config + 日志 + history 落 `run_fold{0..4}/`；日志含「加载 stage1 ckpt 路径 + 非 flow 键零失配 + trainable 参数数」证据。
- [ ] **AC5 smoke 评估**：每折 stage2 后跑 1125/375 口径 test smoke（MAE/std_ratio），逐折登记。
- [ ] **AC6 可复现**：experiment.md Run 块齐全（配置/划分引用/窗口参数/ckpt 与日志路径/耗时/逐折汇总表）；config JSON 与实际 CLI 一致。
- [ ] **AC7 一致性核对**：R5 五项逐条通过并登记。
- [ ] **AC8 回归不破坏**：全量 pytest 维持冻结口径 67 passed / 0 failed / 17 skipped；MODMA/EDRA/edra_fmt 旧线加载路径行为不变。

## 用户决策记录（2026-10-21，1.4 评审问答）

1. **开跑同意 + 门禁偏离签认**：同意 `task.py start` 进入实施；surrogate-brain-qualification §2「fold0 单 seed 三项指标前置验证」门禁偏离**已签认豁免**（本任务不含合格性判定，三项检验另立任务）。
2. **D2 执行节奏 = b 案（先 fold0 检视）**：fold0 两阶段跑完即暂停，向用户呈 fold0 结果（best/smoke/耗时），确认后再续 fold1–4。
3. **D1 geometry 落点 = 数据根豁免案**：允许在 `/root/autodl-tmp/Mild Depression/` 新增 `geometry_mild.pt` 单文件（零覆盖、可独立删除）。
4. **D3 超参 = 沿用旧库 CLI 默认**：epochs 50 / patience 10 / lr 1e-3 / batch 32 / beta_kl 1e-4 / seed 42。
5. **D4 通道序假设 = 签认 E 序假设**（数据通道 i ↔ E(i+1)，GSN-128 规范序；无数据侧证据、模板比对仅防代码 bug 之局限已知悉；错排时可换 dense 重训）。

### 会话续（2026-10-21，fold0 检视后）

6. **fold0 波形判读 = 暂判不通过** → 用户拍板 **D3 变更：epochs 50 → 150 全量重跑**（patience 10 不变，fold0 重跑 + fold1–4 同口径）；50-epoch 旧产物版本保全（run_fold0/ 不覆盖，新跑落 `run_e150/fold{k}/`）。
7. **z-score 口径复核 = 用户确认无问题**（三问：模型输入 z-score ✓ / 输出对 z 域真实对比 ✓ / viz 全 z 域 ✓——训练折池化统计单域无失配，详见会话答复）。
8. **viz 口径变更 v2.1（用户指定）**：面板仅可视化 real-future（蓝）+ gen-future（medoid 红实样本 + Q05–Q95 带），**去掉 past 上下文灰线**；medoid/分位带计算口径不变（v2）。
9. **重跑范围二次收窄（2026-10-21，中止 Run 002 后）**：不做 5 折全量重跑；**仅 fold0 stage2 以 epochs 150 执行**（Run 003）——stage1 复用 run_fold0 已完成的 50-epoch VAE ckpt（E40 best 后 patience 触发早停=真收敛，150 上限重跑不变其行为）；Run 002 被中止的 `run_e150/` 未完成目录保留存档、不启用。
10. **窗内划分变更 5:1 + fold0 两阶段重跑（2026-10-21，Run 004 复查后）**：x^p:x^f 从 3:1（obs 1125 / forecast 375）改为 5:1（obs 1250 / forecast 250）——对拼接后 1500 点窗重新切分（数据文件与形状断言不变），fold0 **stage1 + stage2 以 epochs 150 全量重跑**（Run 005，patience 10 / seed 42 / mexhat-graph 不变，新产物落 `run_ratio5to1/fold0/`）；训练入口与 viz 参数化 `--obs-len/--forecast-len`（默认 1125/375 向后兼容，Run 001 口径可复现）。
11. **窗内划分再试 9:1（2026-10-21）**：obs 1350 / forecast 150 两阶段重跑（Run 006，同 epochs 150/patience 10）——coverage 仍 0.00%（去末端亦然）。
12. **最终拍板（2026-10-21）**：负证据链（Run 001–006：三口径 coverage 全 0%、medoid 主体欠幅、末端 Euler 发散复现）收尾本任务，**不指定 θ 候选、不做 5-fold 全量**；生成质量迭代（损失侧/条件分布校准/采样侧）另立 Study I 任务。

## Notes

- 代理脑资格边界：本任务产物**不自动继承** surrogate-brain 合格性（按 `.trellis/spec/research/surrogate-brain-qualification.md`，固定 θ\* 前须另走三项检验）；本任务交付 = 可复现训练完成的 θ 候选。
- 训练完成 ≠ 代理脑有效；本任务不含评估/合格性判定（属后续任务）。
- 通道序假设（风险登记）：数据文件无通道名，按 GSN-128 规范序 E1..E128 建 geometry（数据通道 i ↔ E(i+1)）；若实际通道序非规范序，mexhat 图先验错排（dense 流不受影响）——已列 design 风险节，构建时与 MODMA geometry（GSN-129 模板前 128）做坐标一致性强校验作为旁证。

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

> 任务创建时由 hook 自动播种。规划阶段必须填写以下四项；全部落定前不得进入 1.4 start 门禁（见 `.trellis/workflow.md` Phase 1.0b / 1.0c）。

- **分类结果**：`study1` —— 本任务为 EEG 代理脑（Study I）候选路径 latent-flow-v1 的代理生成器主体训练（X_past→X_future 条件生成），不涉及扰动/EC 推断算法。`task.py set-scope study1` 已执行（2026-10-21）。
- **依据**：用户指令「基于 latent-flow-v1 训练一个 EEG 代理脑」——纯 Study I 代理脑训练；扰动归因不在本任务范围。
- **新路径/方法检测（1.0c）**：`已有路径：latent-flow（OC-LatentFlow）` —— `.trellis/research/context.md` 技术路线表已登记「Study I 候选：latent-flow」条目；两阶段训练范式与 edra_fmt run（10-19 任务）同构，仅数据集换为 Mild Depression（新数据契约，非新方法），不触发新 research。
- **research 产物**：`无（复用 10-18-latent-flow-port/research/latent-flow-adaptation.md 与 10-19-latent-flow-edra-fmt-train 的 wrapper 模式）`。
