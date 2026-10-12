# 科研进度看板

> **维护规则**：SessionStart hook 只注入顶部「当前状态」区；AI 在节点完成/决策后当场更新（保持 ≤20 行）；历史细节追加到下方里程碑区（只增不改），完整过程见 journal。

## 当前状态

- **阶段**：Study I 首条技术路径 gpfm-mamba 已实现并完成 fold0 单 seed 前置验证（训练+评估数值/图落盘）
- **活跃任务**：`.trellis/tasks/10-12-gpfm-mamba-edra-f0`（in_progress；5-fold 门禁等待用户）
- **最近节点**：2026-10-12 main-01 训练 100ep（val 0.802）+ 前置验证三指标落盘（GC r=0.282、PSD 低频相似/高频偏高、轨迹图待人工检验）
- **下一步**：用户判定三检验 + 5-fold 决定；收尾（spec 更新/3.3/3.4 归档）
- **待拍板**：指标 1/3 通过标准（占位）；是否进入 5-fold；数据口径已定 status_ok 46 例（划分固化 manifest）

## 里程碑（只增不改）

- 2026-10-10 · 研究框架文档定稿（用户提供）；Trellis 科研工作流装配完成；研究记忆初始化 · docs/research/research-direction.md
