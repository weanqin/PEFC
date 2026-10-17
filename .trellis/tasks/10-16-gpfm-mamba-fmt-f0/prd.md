# 新预处理范式 v1/v4a fold0 训练（edra_processed_fmt）

## Goal

把 5-fold 已拍板候选集 {v1, v4a} 迁移到**新预处理范式数据集** `edra_processed_fmt`（EDRA release 自带 1–40 Hz@250 Hz 预处理，48 例 HC24/MDD24）：两变体分别完成 fold0 单 seed 前置验证（smoke → bench → 主 Run → 正式评估），为该范式下的 5-fold 主实验与跨范式对比建立基线；评估中 genGC vs real GC 一致性对比**暂时阻塞**。

## Background（勘察确认的事实）

- **新数据集已就绪**（2026-10-16 转换）：`/root/autodl-tmp/edra_processed_fmt/`，源 = release 自带 1–40 Hz 带通 @250 Hz，48 例，数据值零改动，仅转 v4 dataloader 契约格式（`data.py` 零修改加载，冒烟已过）；manifest 48 行全 status=ok（qc_pass 留空 → 仅 status_ok 口径可用）；身份审计确认与旧范式同源录制（`outputs/edra-paradigm-compare/`）。详见 [research/edra-fmt-dataset.md](./research/edra-fmt-dataset.md)（含与 edra_process_v4 的差异清单表）
- **接入约束**（README「训练接入注意」）：不得复用 v1/v4a 已固化 splits manifest（48≠46 例，`fixate_split` 防漂移按设计报错）→ 须在新 variant 目录首跑固化；已定版变体 config 不可加字段 → 走新薄变体目录（path-isolation 薄变体协议）
- **变体现状**：v1（d16 贯穿 trunk，GC 一致性最高）与 v4a（rank16 输入瓶颈 + trunk d62，质量最优）为候选集；v1 train.py 已备 default-off 钩子（早停/恒 lr/split_manifest_override/bottleneck_rank）
- **门禁**（surrogate-brain-qualification.md §2）：fold0 单 seed 前置验证经用户明确同意后方可 5-fold → 本任务止于 fold0 单 seed
- **genGC vs real GC**：`gc_psd.py` 内 VAR 生成侧↔真实侧一致性；现成 `--skip-gc` 会连 PSD（指标3）/空间归因整段跳过；`evaluate.py` 调用点 fs=500.0 硬编码（250 Hz 下须透传）

## Requirements

- **R1 新薄变体目录 ×2**（`models/gpfm-mamba-v1-fmt/`、`models/gpfm-mamba-v4a-fmt/`）：config 完整副本 + wrapper 三要素（config 遮蔽/前缀守卫 `v1fmt-`/`v4afmt-`/独立模块名加载）；outputs 落新根 `/root/npi/outputs/gpfm-mamba-fmt-f0/`；v1 共享模块改动仅限 default-off 钩子（design §1.1 五处，含 review 回填的轨迹图 fs 与 D4 norm 钩子），旧变体 config_hash 零漂移
- **R2 fold0 划分固化**：48 例受试者级 5-fold 分层（seed 42），内层 n_val_subjects=4（7/1/2 长期口径语义，实际整数分配以首跑固化为准，预期 34/4/10）；两变体 test 折逐 ID 一致（manifest 逐字节一致断言）
- **R3 训练纪律**：fold0 单 seed（1234）、早停 10、恒定 lr 1e-3（无 cosine 退火）、`period_p=25`、`gen_lowpass_hz=None`、`norm_type="layernorm"`；其余超参与基线逐字段一致；跨范式差异清单 = {数据范式（fs/预处理/48例）、p 50→25、低通 40→关、norm RMS→LN、v1 侧启用长期口径}，一切对比必随附（**多变量设计，非单变量**）
- **R4 评估**（每变体）：Q1 质量 + 指标1 FC（`fc_metric --outputs-root` 新根）+ 指标2 视觉（时间轴 250 Hz 正确标秒）+ 指标3 PSD（fs=250；判读口径按 design §1.4-4 预登记：1–30 Hz mean |Δlog10| ≤0.5、40–125 Hz 照登、全频段 mean_abs_logdiff 不作依据）；**genGC vs real GC 阻塞 = 产物中不得出现任何 GC 对比文件**（skip_gc 钩子 + 断言）
- **R5 可复现性**：resolved config dump / config_hash / seed / experiment.md 逐 Run 登记（目的/配置/结果/结论）；GC 阻塞、无低通、p=25、fs=250 四口径显式登记并声明与旧范式不可比部分

## Acceptance Criteria

- [ ] 两变体目录 + 钩子入库；单测全绿（含旧变体 hash 零漂移断言、**norm 钩子断言**（旧变体默认 RMSNorm 结构与 state_dict 键逐位不变；fmt 变体 5 个 norm 位全部 LayerNorm 且含 bias 键）、skip_gc 产物排除断言、fc_metric/轨迹图默认值不漂移断言；v1/v4a/v4 既有测试回归通过）
- [ ] fold0 manifest 固化于各自 splits/，两变体 test 折逐 ID 一致且重复运行校验一致
- [ ] v1-fmt 与 v4a-fmt 各完成：smoke（契约核对：930 窗/例、形状 float32、z-score 有限）→ bench → 主 Run（早停或满 100ep，ckpt+metrics 落盘）
- [ ] 两变体评估产物齐全：Q1 + 指标1 + 指标3（预登记口径补算）+ 视觉图；**无任何 GC 对比文件**（验收时逐产物核验）
- [ ] experiment.md Run 块四区齐全无空块；跨范式对比表随附差异清单 + 三条判读注记（步数/EMA、窗口物理时长、对照纯净度不对称）+ >40 Hz 功率占比观察（psd.npz 数据源）
- [ ] 结论标注「单次运行」；5-fold 启动建议留待用户拍板（不越门禁）

## Out of Scope

- genGC vs real GC 一致性对比（暂时阻塞；解除时点与 250 Hz 下 GC 口径另行拍板）
- 5-fold 主实验（须本任务前置验证通过 + 用户明确同意后另启任务）
- 跨范式连接组层面科学结论（依赖被阻塞的 GC 对比）；旧范式 v1 同纪律重跑（对照纯净度议题，本任务不动）
- r 扫描 {8,32}（已议不采纳）、v2/v4b 等落选变体复活

## Key Decisions（用户拍板，2026-10-16）

- **D1 阻塞 GC 对比**：genGC vs real GC 一致性暂时不出现在评估产物；PSD 与空间归因照跑
- **D2 生成侧不加 40 Hz 低通**：`gen_lowpass_hz=None`；新范式新口径（真实侧本就 1–40 Hz 带限，直接观察模型是否自行待在带内）；旧范式 Run 保持原口径，跨范式对比登记不可比
- **D3 `period_p` 50→25**：保物理时长（`gpr_prior.py:94` τ 网格采样点单位，fs 减半 p 减半；review 核验严格等价）；窗口 384/128 仍按样本数（README 契约）；先验与数据同时变，判读随附差异清单
- **D4 RMSNorm→LayerNorm**：两 fmt 变体全部 norm 位（4×MambaBlock pre-norm + head norm，仓库仅此两处构建点已核尽）换 LayerNorm；经 `norm_type` default-off 钩子实现（默认 "rmsnorm" = 旧变体结构与 state_dict 键零漂移）；容量微增（+80/+310 bias 参数）登记

## 子研究路由（1.0b / 1.0c）

- **分类结果**：`study1` —— EEG 代理脑训练与前置验证（框架 Study I）
- **依据**：任务主体是 gpfm-mamba 候选构型在新数据范式上的训练与代理脑检验
- **新路径/方法检测（1.0c）**：`已有路径：Study I gpfm-mamba（v1/v4a，技术路线表现有条目）`——无新方法，变化仅数据范式（转换+身份审计已完成），不触发 adaptation research；design 定稿前复检无变化
- **research 产物**：`无`（数据集事实卡 `research/edra-fmt-dataset.md` 为勘证备份，非 adaptation 研究）
