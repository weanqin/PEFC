# gpfm-mamba-v5-fmt（fmt 数据训练）：v5 轴分解交替变体 × edra_fmt/250 Hz，产出 θ* 候选 v5-fmt-01

## Goal

以 gpfm-mamba-v5 变体（v2 基线 + 4×[Mamba→GNN 空间块→LayerNorm] 轴分解交替，**model/graph 原样复用、不改结构**）在 **edra_fmt（250 Hz，48 例）** 上训练，run-id **`v5-fmt-01`**（与 `v5-main-01`（edra_process_v4/500 Hz）并列，同 outputs 根 `outputs/gpfm-mamba-f0/`）；fold0 划分只读复用 **v1-fmt manifest（34/4/10）**。产出 θ\* 候选（EMA shadow），下游接入 **study2 NI-MSE-50-q0.90** EC_pert 计算路径（ni-mse-design.md v1.5 主实例化）——与 real GC refs（fmt 48 例）同管线，消除跨范式配对 caveat（用户 2026-10-20 拍板，收益明示）。

## Requirements

1. **变体目录 `models/gpfm-mamba-v5-fmt/`**：fmt 口径 config（v5 config 副本 + data_root=`/root/autodl-tmp/edra_processed_fmt`、fs=250、fmt 线口径 **LayerNorm / p=25 / 无低通（gen_lowpass off）**——逐字段以 `models/gpfm-mamba-v1-fmt/` 先例为准）+ v5 `model.py`/`graph.py` **原样**（轴分解交替、A_eff=γ_m·M+γ·W、几何 choke point 均不变，geometry_path 即 fmt 原生 `geometry_edra_fmt.pt`）+ train/evaluate 双遮蔽包装（run_id 前缀 `v5-` 沿用，`v5-fmt-01` 与 `v5-main-01` 天然隔离）。
2. **划分固化**：fmt fold0 manifest 只读复用 `models/gpfm-mamba-v1-fmt/splits/`（34/4/10，零重新划分）。
3. **单测**：v5 十项的 fmt 实例 + fmt 口径差异项（norm_type/p/低通/数据通道序与 geometry 对齐）；回归 v1/v1-fmt/v5 全绿 + config_hash 零漂移。
4. **训练链**：smoke → bench（材料登记）→ 主 Run `v5-fmt-01`（100 epoch 上限、batch 256、seed 1234、EMA 0.999、早停按 config 家族默认如实登记）；R9 类启动确认沿 v5-main 先例属执行期用户确认项。
5. **读出登记**：训练完成即做 A_eff/W 快读出（EMA shadow、σ/W 范数、M vs W 分解——对齐 v5-main-01 Run 006 协议），供与 v5-main-01 对照。

## Acceptance Criteria

- [ ] AC1：变体目录落地 + 三处注册一致（variant_name / checkpoint 主名 / run 前缀）+ 双遮蔽与路径守卫可跑
- [ ] AC2：单测全绿（v5 十项 fmt 实例 + fmt 差异项）+ 回归零漂移（v1 `c6ff80ac1c43` / v5 `1ac433fdd3d0` 等 pin 复验）
- [ ] AC3：smoke/bench 材料登记 experiment.md（参数量 = 152,647 预期不变——结构原样；显存/it/s/时长）
- [ ] AC4：主 Run `v5-fmt-01` 完成（best ckpt 落盘，config_hash pin 登记）；A_eff/W 快读出登记
- [ ] AC5：产物隔离（只新增 `models/gpfm-mamba-v5-fmt/` 与 `outputs/gpfm-mamba-f0/v5-fmt-01/`；v5/v1/v1-fmt 等只读）
- [ ] AC6：experiment.md 全程按 spec 契约（跑前登记/跑后补记，含口径勘误如实登记）

## Notes

- **非目标**：study2 NI-MSE-50-q0.90 的执行（θ\* 落地后另立项/续接 ni-mse-design.md v1.5）；Run 017 同口径评估（v5-main 侧同款未竟项，属各自任务）。
- 依赖与先例：v5 结构（任务 10-19 S1'）、v1-fmt 口径（任务 10-16-gpfm-mamba-fmt-f0）、v5-main 训练链先例（Run 003–005）。
- 计划来源：任务 10-19-gpfm-mamba-v4b-gnn-f0 implement.md S7'（2026-10-20 用户拍板新增）。

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

> 任务创建时由 hook 自动播种。规划阶段必须填写以下四项；全部落定前不得进入 1.4 start 门禁（见 `.trellis/workflow.md` Phase 1.0b / 1.0c）。

- **分类结果**：`study1`
- **依据**：训练 EEG 条件未来生成代理脑（gpfm-mamba 变体 × fmt 数据），属框架 Study I 训练任务；study2 消费（NI-MSE 接入）仅作下游背景不入本任务范围。
- **新路径/方法检测（1.0c）**：`已有路径：gpfm-mamba 行 v5 变体（context.md 技术路线表 Study I）+ fmt 范式先例（10-16-gpfm-mamba-fmt-f0）——v5 结构原样、仅换 fmt 数据/口径，无新方法路径`
- **research 产物**：`无`
