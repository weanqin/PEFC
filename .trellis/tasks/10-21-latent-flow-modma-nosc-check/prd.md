# latent-flow MODMA no-sc ckpt 本机检查与首跑

## Goal

检查旧库 MODMA no-sc（screg l0.01 w0.1 nosc，MODMA SOTA）ckpt 自包含性，重建缺失的 geometry.pt（MNE GSN-129 模板 + ckpt pairwise_d 验证），并经包内标准 probe 首跑一次验证 MODMA 原生线本机可运行；数据契约按 latent-flow 约定。

## Background（规划期侦察结论，2026-10-21）

- no-sc ckpt 5 折齐全：`/root/autodl-tmp/latent_flow/checkpoints/fold{0-4}_oc_latent_flow_flow_best_graph_mexhat_screg_l0.01_w0.1_nosc.pt`；state_dict **自包含**（`encoder/decoder/proj/flow` 全带 + `flow.graph_mixer` buffers `W/pairwise_d/gamma_buf/gamma_sc/gamma_m/log_sigma`），无需另配 stage1 VAE。
- `latent_flow/docs/model_registry.yaml` 登记：**nosc = MODMA SOTA**（fold2 r_abs(GC)=0.585，即 FOUNDATION 0.585 出处），06-29 组件消融家族成员。
- 包内标准加载路径 `scripts/_connectivity_driver.load_model` 自动检测 mexhat-graph 变体（`W` 128×128 → dataset=modma）；mexhat 分支构建时需 `load_geometry('modma')` → `MODMA_DATA_ROOT/geometry.pt`。
- **geometry.pt 本机缺失且原物不可恢复**：旧库 `.gitignore` 排除 `data/modma/`、`latent_flow.7z` 不含、git 全历史（含 bundle heads）无此文件。
- MODMA resting 数据本机在位：`/root/autodl-tmp/modma`（HC 29 + MDD 24，`.mat` 键 `data`，128 ch × ~42501，250 Hz；无通道名元数据键）。
- 数据契约（latent-flow 约定）：`MODMADataset` 逐被试逐通道 z-score；窗 500 样本（2 s）/ obs 375 / step 250；5 折被试级 `SubjectFoldSplitter` seed=42；env 经 `MODMA_DATA_ROOT` 指根。

## Requirements

- **R1 geometry.pt 重建**（用户拍板 2026-10-21：MNE 模板 + 验证）：MNE 1.13.2 `GSN-HydroCel-129` 模板取 128 导坐标（米制），写入 `/root/autodl-tmp/modma/geometry.pt`（`{coords: (128,3) float32, channel_names: list[128]}`，满足 `load_geometry` 契约）。写入数据根属环境工件补全（同 10-19 D2 geometry 豁免单文件先例）。
- **R2 几何验证门**：以 fold2 nosc ckpt 内嵌 `flow.graph_mixer.pairwise_d` 为基准，登记 offdiag Pearson 相关 + 相对误差。模板序直接比对 corr ≥ 0.99 → 模板口径通过；若低但几何本身匹配（通道序不同），用距离行匈牙利匹配恢复通道置换后复验；仍不达 → 回退经典 MDS（由 pairwise_d 重建坐标，cdist 逐位一致）并如实登记回退。
- **R3 ckpt 加载断言**：经包内 `load_model` 加载 fold2 nosc → 打印 `mexhat-graph / dataset=modma`；加载后 `model.flow.graph_mixer.W` / `pairwise_d` 与 ckpt 存储值逐位一致（strict 断言零真实 mismatch）。
- **R4 probe 首跑**：经 `run.py` 包装（BLAS 守卫）跑 `probe_oc_latent_flow --fold 2 --checkpoint <nosc fold2> --subjects 1`（128 源、σ∈{0.1,0.2,0.3}）；产物（per-subject JSON + 热图 PNG）落 `/root/npi/outputs/10-21-latent-flow-modma-nosc-check/run_01/`。
- **R5 记录**：experiment.md 按模板登记 Run 001（geometry 重建+验证）/ Run 002（加载断言）/ Run 003（probe 首跑）；结果如实登记，不外推。

## Acceptance Criteria

- [ ] **AC1** `geometry.pt` 在位且 `load_geometry('modma')` 可读；验证数字（corr / 相对误差 / 是否回退）登记于 experiment.md。
- [ ] **AC2** fold2 nosc ckpt 加载成功（mexhat-graph、dataset=modma、零真实 state_dict mismatch）且 `W` / `pairwise_d` 逐位一致。
- [ ] **AC3** probe 首跑成功：per-subject JSON + PNG 落 outputs，CV 等指标如实登记（与 JET 基线的对比仅按脚本原样输出，不作新判读）。
- [ ] **AC4** 零仓库代码改动（包体零逻辑改动纪律）；新增工件仅 `geometry.pt`（数据根）与 outputs 产物。

## 边界与不变式

- **旧 θ\* 不自动继承 npi 代理脑资格**——本任务仅为可运行性验证与首跑存证；三项合格性检验（`surrogate-brain-qualification.md`）另立任务。
- 不做 nosc 与其他变体/fold 的横向对比评估；不触发 σ 二分完整归因流水线（`run_oc_connectivity_*`）。
- MODMA 数据与源库 `/root/autodl-tmp/latent_flow` 只读；`geometry.pt` 为本次唯一写入数据根的工件。
- 环境契约：base conda（Python 3.12 / torch 2.8.0+cu128 / MNE 1.13.2 已核实在位），对照 `.trellis/research/environment.md`。

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

- **分类结果**：`study1`（EEG 代理脑，框架 Study I）
- **依据**：latent-flow（OC-LatentFlow）为 Study I 候选代理脑线；本任务验证其 MODMA 原生线在本机的可运行性，属 Study I 候选侧工作。
- **新路径/方法检测（1.0c）**：`已有路径：latent-flow（技术路线表「Study I 候选：latent-flow」行，10-18 移植 / 10-19 edra_fmt 训练已登记；本任务无新方法）`
- **research 产物**：`无`
