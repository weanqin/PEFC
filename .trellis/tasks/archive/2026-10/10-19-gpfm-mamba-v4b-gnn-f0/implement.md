# implement — gpfm-mamba-v5（v2 基线 × 轴分解交替）

> 2026-10-19 二次转向后执行清单（v4bg S1–S2 已完成并留档，见 experiment.md Run 001/002 与「v4bg 停用」登记；v4bg-main-01 永不启动）。
> 本轮范围 = 用户六项要求：结构落地 + 注册 + 最小验证；**不含任何训练 run**。

## S1' v5 结构落地（要求①–⑤）

- [ ] S1'.0 git 快照（`git status --porcelain` 存 experiment.md；基线含 v4bg 目录与 3 处悬置未提交文件）
- [ ] S1'.1 `models/gpfm-mamba-v5/`：`config.py`（v2 副本 + graph 字段 + variant_name="gpfm-mamba-v5" + d_model=62）、`graph.py`（MexHatAdjacency：A_eff 机制/正则/几何 choke point + GraphSpatialStep）、`model.py`（v2 前向 + 4×[Mamba→空间块→LayerNorm] 交替 trunk；钩子三参断言默认；d_model==n_channels 断言）、`train.py`/`evaluate.py` 双遮蔽包装（遮蔽 config+model；run_id 前缀 `v5-`）
- [ ] S1'.2 `tests/test_v5.py`（design §4 十项）
- [ ] S1'.3 registry 回填：`.trellis/research/context.md` gpfm-mamba 行变体注记 + v5 转向一句（要求⑤，随本轮落档）
- 验证：`python3 -m pytest models/gpfm-mamba-v5/tests/test_v5.py -q` 全绿；回归 v1/v4/v4a 全绿 + hash 零漂移；git 快照对比新增仅 `models/gpfm-mamba-v5/` 与 context.md 注记
- 回滚点：删新目录 + revert context.md 注记即净

## S2' 最小验证（要求⑥，随 S1' 单测一并交付）

- [ ] 构建：`VelocityNetwork()`（默认参）构建成功 + 参数量精确值登记（对照 design §2.5 预算 v2+19,965）
- [ ] 前向：随机张量 (4, 62, 384)/(4, 62, 128)/t → 输出 (4, 62, 128)；return_all → (4, 62, 512)
- [ ] v5 名称实例化：`build_config()` → variant_name/config_hash；`train.py --help`/前缀守卫可跑（不启动训练）
- [ ] 差异陈述：design §5 表随汇报输出

## S3' 训练/评估（后续阶段，本轮不做）

smoke → bench（W 范数 raw/EMA 双口径）→ **G-R9 用户确认** → 主 Run `v5-main-01` → Run 017 同口径评估（对照 v1/v2/v5）+ A_eff/W 读出（判定轨距离偏 Spearman）。

## S4'/S5'/S6' 质检与收尾

- trellis-check（全范围 + 零改动 git 复核）；三态判读（H4'，对照 v2）；spec 回填；3.4 提交。

## S7' v5-fmt 训练 + study2 接入（2026-10-20 用户新增计划；**同日转出为独立任务 `10-20-gpfm-mamba-v5-fmt-f0` 执行——本段留档为计划源，不再在本任务内勾选**）

- [ ] S7'.1 变体目录 `models/gpfm-mamba-v5-fmt/`：**fmt 口径 config**（v5 config 副本 + data_root=`/root/autodl-tmp/edra_processed_fmt`、fs=250、fmt 线口径 LayerNorm/p=25/无低通——对齐 `models/gpfm-mamba-v1-fmt/` 先例，逐字段以 v1-fmt config 为准）+ **v5 model/graph 原样复用**（轴分解交替结构不变）+ train/evaluate 双遮蔽包装（run_id 前缀 `v5-` 沿用——v5-fmt-01 与 v5-main-01 同 outputs 根并列，用户指定）
- [ ] S7'.2 划分固化：fmt fold0 manifest 只读复用 `models/gpfm-mamba-v1-fmt/splits/`（34/4/10，零重新划分）
- [ ] S7'.3 单测（v5 十项的 fmt 实例 + fmt 口径差异项：norm_type/p/低通/dataloader 通道序对齐 geometry）
- [ ] S7'.4 smoke → bench → 主 Run **`v5-fmt-01`**（启动门禁沿 v5-main 先例处理——用户直训授权是否沿用属执行期确认项）
- [ ] S7'.5 θ\* = v5-fmt-01（EMA shadow）接入 **study2 NI-MSE-50-q0.90**（ni-mse-design.md v1.5 主实例化：n₀=25、refs 同管线、跨范式 caveat 消除；逐被试配对主口径 n=10）
- 边界：本段为计划登记（2026-10-20 用户拍板，收益 = 代理脑与 refs 同 fmt 管线）；与 S4'/S5'/S6' 的先后关系由用户定（收尾本任务 → 立项执行，或并行）
