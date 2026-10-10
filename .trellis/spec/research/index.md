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
| `../research/` 事实档案 | `.trellis/research/{context,progress,environment}.md`（SessionStart hook 自动注入；环境未记录的场景必须先问用户） |
| [experiment-template.md](./experiment-template.md) | 任务创建时自动播种到 `{TASK_DIR}/experiment.md` 的模板 |

## Pre-Development Checklist（写代码/跑实验前逐项确认）

- [ ] **环境**：运行环境（conda env / 本地或远程机器）已在 `.trellis/research/environment.md` 查到并确认；未记录 → 先问用户，不得猜测、不得自行安装依赖
- [ ] **路径隔离**：本任务的新方法/变体放入独立目录；`design.md` 已声明全部读写路径，与既有变体无冲突
- [ ] **方向一致**：设计与 `.trellis/research/context.md` 的技术路线、既有结论一致；有意偏离处已标注「需用户拍板」
- [ ] **数据口径**：输入数据来源、分组/筛选口径已对照项目数据契约（如存在 `.trellis/spec/data/`）写明
- [ ] **跑前登记**：跑批/训练前先在 `{TASK_DIR}/experiment.md` 登记本次 Run 的「目的 + 配置」（含 seed 与指标口径，见 reproducibility / evaluation-calibers）

## Quality Check（报告完成前逐项确认）

- [ ] `experiment.md` 所有 Run 块四要素齐全（目的/配置/结果/结论），无空块
- [ ] 产物全部落在本变体目录（`outputs/<variant>/` 等），无跨变体写入
- [ ] 代码在声明的环境中可运行（附可复现命令）
- [ ] `.trellis/research/progress.md`「当前状态」已同步；重要节点已追加里程碑
