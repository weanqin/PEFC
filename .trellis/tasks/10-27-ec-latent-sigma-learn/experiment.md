# 实验日志 — <任务名>

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 001 — <一句话名称>（YYYY-MM-DD）

### 目的与假设

- 要回答什么问题 / 验证什么假设？

### 配置

- 代码：`<变体目录/文件 + 版本快照>`
- 参数：<与默认不同的超参 / 阈值 / seed>
- 环境：<env 名 @ 机器（本地/远程），对照 .trellis/research/environment.md>
- 数据：<输入路径 + 筛选口径（qc_pass、分组等）>
- 命令：<可复现启动命令>

### 结果

- 指标：<数值 / 小表>
- 产物：`<本变体 outputs/... 路径>`
- 图表：<路径 + 一句话说明>

### 结论

- 支持 / 证伪 / 不确定 + 一句话依据
- 下一步 / 派生决策项：<同步到 .trellis/research/progress.md>
