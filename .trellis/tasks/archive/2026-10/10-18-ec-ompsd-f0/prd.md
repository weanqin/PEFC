# O×M-PSD 完整 EC 推断（v1-fmt θ*·fold0 test；镜像 ec-nmtime 流程）

## Goal

以 O（恒定偏置）×M-PSD（全带谱距离）组合执行完整 EC_pert 生产流程（10 例 80 窗、9 点含原点 O 尺度网格、slope/AUC/r* 三 readout、33 契约 npz、六项描述统计，只出矩阵不做对照——镜像 10-18-ec-nmtime-f0 口径）；完成后在最大扰动强度 r=R_MAX=4.0 下可视化 past+future 波形（含 PSD 读出面板）。搜索任务中该组合 Gate A 最强（z=6.64）而 Gate B 完美弥散（1.01）——与 N×M-time 成对照极端

## Requirements

- TBD

## Acceptance Criteria

- [ ] TBD

## Notes

- Keep `prd.md` focused on requirements, constraints, and acceptance criteria.
- Lightweight tasks can remain PRD-only.
- For complex tasks, add `design.md` for technical design and `implement.md` for execution planning before `task.py start`.


## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

> 任务创建时由 hook 自动播种。规划阶段必须填写以下四项；全部落定前不得进入 1.4 start 门禁（见 `.trellis/workflow.md` Phase 1.0b / 1.0c）。

- **分类结果**：`待确认` —— 可选值：`study1`（EEG 代理脑，框架 Study I）/ `study2`（EC 推断算法，框架 Study II）/ `待确认`（歧义或非科研任务；不得强行归类，须向用户确认）
- **依据**：<一句话，从任务描述判定的理由；待确认时写明歧义点>
- **新路径/方法检测（1.0c）**：`待检测` —— 可选值：`不涉及` / `已有路径：<对照 .trellis/research/context.md 技术路线表填写>` / `新路径：<slug> → research 已触发|待触发`
- **research 产物**：`无` —— 涉及新路径/方法时填 `research/<slug>-adaptation.md`（适配结论 / 风险 / 可执行方案）
