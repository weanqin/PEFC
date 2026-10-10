# 产物与 Checkpoint 管理规范

**原则**：产物可追溯（哪个 Run、哪份代码、哪个 seed 产出）、可清理（保留策略先声明）、不失控（大文件不进版本库）。

## 目录组织

- 一切产物落 `outputs/<变体名>/...`（目录契约见 [path-isolation.md](./path-isolation.md)）；禁止散落在仓库根目录或代码目录
- **双向互链**：experiment.md 的 Run 块写产物路径；产物目录内存 run 元信息（resolved config、启动命令、日期）——从结论能回溯到产物，从产物能找到 Run

## Checkpoint 规则

- 命名含**变体名 + 用途 + 简短标识**：`best`（按主指标）/ `last` / `periodic-<间隔>`；禁止 `model.pth` 这类无主名
- **保留策略先声明**：通常 best + last 各一份，periodic 按需保留；训练结束即按策略清理中间 checkpoint，避免磁盘失控
- resume（断点续跑）视为同一 Run 的延续：在原 Run 块补记断点位置与续跑差异，不另开 Run

## 大文件与版本库

- checkpoint / 数据 / 大图不进 git 追踪（`.gitignore` 或放仓库外路径）；代码版本以 commit 哈希或「文件名+日期快照」记录在 Run 块（非 git 仓库场景）

## 清理纪律

- 删除任何既有产物前，必须列出完整清单征得用户确认；「看起来没用了」不构成删除理由
- **归档优于删除**：疑似无用先移入 `archive/` 子目录并留一行说明，观察一段时间再议
