# Research Specs（科研工作流规范）

**适用范围**：本工作区全部科研代码与实验任务。具体目录清单与文档地图在**初始化时**确定并记录于 `.trellis/research/context.md`（未初始化时见档案内占位说明）。

本包是 Trellis 工作流的科研定制层，与数据契约包（`../data/`，本模板自带）、思考指南（`../guides/`）并列。

| 规范 | 内容 |
|---|---|
| [experiment-log.md](./experiment-log.md) | 实验日志格式（目的/配置/结果/结论）与更新时机 |
| [path-isolation.md](./path-isolation.md) | 多方法/多路径并行的目录隔离规则 |
| [reproducibility.md](./reproducibility.md) | seed/确定性/环境快照/复现三要素 |
| [artifacts-management.md](./artifacts-management.md) | outputs 组织与 checkpoint 管理、清理纪律 |
| [evaluation-calibers.md](./evaluation-calibers.md) | 指标定义先行、口径纪律与统计汇报 |
| [perturb-ec-pipeline.md](./perturb-ec-pipeline.md) | Study II 扰动→EC 推断流程首版（CRN 契约/增益算子/τ 双口径 v2/readout 与比较接口/隔离装载矩阵/首版实测基线；任务路由到 `study2` 时必读） |
| `../research/` 事实档案 | `.trellis/research/{context,progress,environment}.md`（SessionStart hook 自动注入；环境未记录的场景必须先问用户） |
| [experiment-template.md](./experiment-template.md) | 任务创建时自动播种到 `{TASK_DIR}/experiment.md` 的模板 |
| [surrogate-brain-qualification.md](./surrogate-brain-qualification.md) | Study I 代理脑合格性检验：FC 一致性（v2.1 Pearson 口径，GC 为补充参考）/ 人工视觉检验 / PSD 相似性三指标与固定 θ\* 前门禁（任务路由到 `study1` 时调用；指标 1/3 通过标准 2026-10-16 已回填） |
| [data-pipeline.md](./data-pipeline.md) | dataloader 统一接口（全部预处理在 dataloader 内、模型侧零变换）与 5-fold 分层划分（MDD/HC 受试者级） |
| [visualization.md](./visualization.md) | 全部产图经统一 visualizer，美学风格由 nature-figure 技能统一 |

## Pre-Development Checklist（写代码/跑实验前逐项确认）

- [ ] **环境**：运行环境（conda env / 本地或远程机器）已在 `.trellis/research/environment.md` 查到并确认；未记录 → 先问用户，不得猜测、不得自行安装依赖
- [ ] **路径隔离**：本任务的新方法/变体放入独立目录；`design.md` 已声明全部读写路径，与既有变体无冲突
- [ ] **方向一致**：设计与 `.trellis/research/context.md` 的技术路线、既有结论一致；有意偏离处已标注「需用户拍板」
- [ ] **数据口径**：输入数据来源、分组/筛选口径已对照项目数据契约（如存在 `.trellis/spec/data/`）写明
- [ ] **数据管道**：全部预处理（z-score、时间窗切分等）在 dataloader 内完成、模型侧零变换；5-fold 分层划分（MDD/HC）已定 fold 标识（[data-pipeline.md](./data-pipeline.md)）
- [ ] **跑前登记**：跑批/训练前先在 `{TASK_DIR}/experiment.md` 登记本次 Run 的「目的 + 配置」（含 seed 与指标口径，见 reproducibility / evaluation-calibers）

## Quality Check（报告完成前逐项确认）

- [ ] `experiment.md` 所有 Run 块四要素齐全（目的/配置/结果/结论），无空块
- [ ] 产物全部落在本变体目录（`outputs/<variant>/` 等），无跨变体写入
- [ ] 代码在声明的环境中可运行（附可复现命令）
- [ ] `.trellis/research/progress.md`「当前状态」已同步；重要节点已追加里程碑
- [ ] 路由为 `study1` 的任务：代理脑合格性检验已按 [surrogate-brain-qualification.md](./surrogate-brain-qualification.md) 执行并落盘（fold0 单 seed 前置验证 → 用户同意 → 5-fold；固定 θ\* 前必须完成三项检验）
