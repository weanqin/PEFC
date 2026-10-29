# symw 五折 test 集 gen-fc vs real-fc Pearson 一致性评估（镜像 fc_metric v2.1 口径）

## Goal

对 mild-latent-flow-symw-anti0.2-gnn-exit（symw）五折 test 集（每折 8 例×30 trial，w500 口径 obs375/fc125）：模型确定性生成 fc 段 → 逐被试 Pearson FC（gen-fc），同段真实数据同流程 → real-fc，镜像 gpfm-mamba-v1 fc_metric.py v2.1 口径（组平均上三角 Pearson r+Spearman ρ + 逐被试单样本 t 检验），五折逐折指标 + 5-fold 汇总平均；ckpt 取 result/checkpoints/mild/latent_flow_symw/（SHA 对拍 MANIFEST 登记值）。

本任务 = Study I 代理脑合格性检验**指标 1（FC 一致性，v2.1 Pearson 口径）**在 symw mild 线五折训练完成后的正式核算（surrogate-brain-qualification.md 判定流程第 1 条触发条件已满足）。只出数与登记，合格与否按规范通过标准如实陈述，不预设结论；FC 一致性属 Surrogate Brain Validity（Q2），与预测质量（Q1）分开陈述不混写。

## Requirements

### R1 数据与划分（只读复用既有契约）

- manifest：`models/latent-flow-v1/splits/mild_5fold_seed42.json`，运行前 SHA256 断言 = `47987909…5d881b`（登记值）
- 每折 test = 8 例（4 MDD + 4 HC，subject_id 以 manifest fold 记录为准，禁 stem 解析）× 30 trial 全量
- 窗口径 = w500（与 symw ckpt 训练口径一致）：每 trial 前 500 点，obs 375 / **fc 125**（trial 第 375–500 点，0.5 s @250 Hz，128 ch；数据 x_f 字段不参与）
- 归一化 = train 折前 500 点逐通道 mean/std（`compute_mild_train_norm` 同一实现）；norm SHA 按折对拍 `ec_mild_unbiased.EXPECTED_NORM_SHA_BY_FOLD` 登记值（折-统计错配拒跑）

### R2 生成侧（gen-fc 输入）

- ckpt：`result/checkpoints/mild/latent_flow_symw/fold{k}/fold{k}_mild_flow_best_graph_mexhat.pt`（k=0..4，result 目录只读）
- SHA256 逐文件对拍 `result/checkpoints/MANIFEST.md` 登记值 + 文件名 fold{k}_ 前缀与 --fold 锁定（防折-权重错配）
- 装载：`build_model_mild(_model_args(flow_exit_gnn=True), stage=2, obs_len=375, forecast_len=125)` + `load_flow_ckpt` strict（symw 零架构差异）
- 采样：`sample_forecast` 全确定性（z_obs=μ + 确定性 Euler 10 步 + 确定性 decoder，无随机源）⇒ **K=1**（K 条逐位相同）；seed 42 照设并随档登记

### R3 FC 口径（镜像 gpfm-mamba-v1/fc_metric.py 指标 1 v2.1，逐字对齐）

- 逐被试：该被试 30 个 fc 段拼接 → (128, 3750) float64 → `np.corrcoef` → 128×128 Pearson FC；**gen/real 两侧完全相同流程**
- 点估计：组内被试平均 → 每侧一个组平均矩阵 → 上三角（k=1，8128 边）展平 → Pearson r + Spearman ρ
- 显著性：逐被试一致性 r_s/ρ_s（同被试 gen FC vs real FC 上三角相关）→ 单样本双侧 t 检验（H0: 均值=0，df=7/折）
- 计算域 = z 域（与生成管线同域）；Pearson 对逐通道仿射变换不变 ⇒ 与原始 μV 域数学等价——双阈值数值断言守卫（每折首个 test 被试）：①float64 独立重算 z 域 vs 原始域 max|ΔFC| < 1e-12（纯舍入）；②管线消费的 float32 z 域 vs 原始域 float64 max|ΔFC| < 1e-6（float32 存储量化主导；规划期原定 1e-8 未计及数据集 float32 量化，实施前修订登记）

### R4 汇总（5-fold）

- 逐折表：组点估计 r/ρ + 被试级 mean±sd + t/p（每折 8 例）
- 5-fold 平均：①五折组点估计的算术平均；②被试级合并 40 例 mean±sd + 单样本 t 检验（df=39）；③五折被试级均值的平均（与②并置登记）
- 判定陈述对照指标 1 通过标准（组平均 Pearson r ≥ 0.95；辅助：逐被试 t 检验 p<0.05），如实登记不预设

### R5 产物与路径

- 新脚本：`models/latent-flow-v1/fc_metric_mild_symw.py`（复用 load_manifest / compute_mild_train_norm / MildTrialDataset / sample_forecast / load_flow_ckpt，禁止复制实现）
- 产物目录：`outputs/10-29-symw-fc-5fold/`——`fold{k}/fc_metric.json`（口径+数值+凭据）+ `fold{k}/fc_matrices.npz`（组平均+逐被试 FC）+ `summary_5fold.json` + `README.md`（五折并置表）
- `result/` 本任务**只读不写**（result-dir-policy：写入须用户另行点名）

## Acceptance Criteria

- [x] AC1 五折 5/5 全量跑通（每折 8 例 × 30 trial × 125 点），逐折 `fc_metric.json` 四要素齐全（组点估计 r/ρ、逐被试 r_s/ρ_s、t 检验、口径/凭据登记）
- [x] AC2 守卫全过：manifest SHA 断言 + norm SHA ×5 折对拍登记值 + ckpt SHA ×5 折对拍 result MANIFEST 登记值，全部通过
- [x] AC3 五折汇总交付：`summary_5fold.json`（逐折表 + R4 三项汇总）+ README 并置表
- [x] AC4 z 域/原始域 FC 等价数值断言通过（每折首个 test 被试：①float64 重算 < 1e-12；②float32 管线 < 1e-6）
- [x] AC5 确定性复跑：fold0 整折重跑一次，gen 样本逐位一致 + JSON 数值逐位一致
- [x] AC6 experiment.md Run 块四要素登记 + `.trellis/research/progress.md` 当前状态同步（含指标 1 判定陈述）

## Notes

- 轻量任务：PRD-only，不另立 design.md/implement.md（技术要点已内联 R1–R5）
- 预算：GPU 前向 5 折 × 240 trial × K=1（Euler 10 步小模型），每折约 2–5 min
- 生成侧确定性采样口径与 quality_check_mild.py / EC 线一致（同 sample_forecast 实现）

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

> 任务创建时由 hook 自动播种。规划阶段必须填写以下四项；全部落定前不得进入 1.4 start 门禁（见 `.trellis/workflow.md` Phase 1.0b / 1.0c）。

- **分类结果**：`study1`（EEG 代理脑，框架 Study I）
- **依据**：任务本质 = Study I 代理脑合格性检验指标 1（FC 一致性 v2.1）在 symw mild 线五折的正式核算（surrogate-brain-qualification.md 触发条件：真实 EEG + 5-fold 训练完毕），评估对象是代理脑本身（Q2），非 EC 推断算法
- **新路径/方法检测（1.0c）**：`不涉及`——已有路径：latent-flow × Mild Depression 线（context.md 技术路线表第 4 行）既有变体 symw 的合格性检验指标 1 执行，方法口径全部复用既有登记（fc_metric v2.1 / w500 契约 / result ckpt 注册件）
- **research 产物**：`无`
