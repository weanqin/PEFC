# dl-research — 深度学习科研规范（基础模板）

面向深度学习科研的 Trellis spec 起点，**与具体研究方向无关**。安装后落在项目 `.trellis/spec/{research,data}/`。

## 安装

```bash
# 新项目（trellis init 时）
trellis init --registry gh:weanqin/weanqin-spec-marketplace/marketplace --template dl-research

# 已有项目（只补缺失文件，不覆盖本地已有）
trellis init --registry gh:weanqin/weanqin-spec-marketplace/marketplace --template dl-research --append
```

安装源会被记录，之后 `trellis update` 可从 marketplace 刷新 spec（本地修改会按冲突流程提示）。

## 安装后（两步）

1. **装配配套工作流部件**：研究记忆 hook、实施前审查 agent、workflow 门禁、研究档案模板——按本仓库 `docs/bootstrap/bootstrap.md` 的七步手册执行（spec 模板只覆盖 `.trellis/spec/`，其余部件官方 marketplace 不装载）。
2. **初始化研究方向**：由用户提供研究方向资料（路径或内容），填写 `.trellis/research/{context,progress,environment}.md` 并删除各文件顶部的 `uninitialized` 标记；在 `context.md` 文档地图中确定本项目的实验笔记目录与命名惯例、适用代码目录清单。

## 包含模块

| 层 | 文件 | 内容 |
|---|---|---|
| research/ | index.md | 入口：Pre-Development 检查清单 + Quality Check |
| | experiment-log.md | 实验日志 Run 四要素（目的/配置/结果/结论）与时机 |
| | path-isolation.md | 多方法/变体并行的一条路径=一个目录隔离规则 |
| | experiment-template.md | 任务创建时自动播种的实验日志模板 |
| | reproducibility.md | seed / 确定性 / 环境快照 / 复现三要素 |
| | artifacts-management.md | outputs 组织、checkpoint 命名与保留、清理纪律 |
| | evaluation-calibers.md | 指标定义先行、口径纪律、统计汇报 |
| data/ | index.md | 数据契约包入口 |
| | dataset-contract-template.md | 每个数据集一份实测事实契约的模板 |

## 演化约定

本模板内容装进项目后归项目所有，本地可自由收紧/扩展；验证有效的新规范按仓库根 README 的「回沉淀流程」泛化回 marketplace。模板 id `dl-research` 只做增量演进，破坏性重写会发布新 id。
