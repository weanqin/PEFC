# O×M-time@25 完整 EC 推断（输出变化仅计前 0.1s/25 点；v1-fmt θ*·fold0）

## Goal

O（DC 偏置）×M-time 变体：主标量 = |δ̄| 前 25 样本（0.1 s）均值（用户 2026-10-18 指定，替换原 32 点/128 ms 口径）；镜像 EC 生产流程（80 窗、O 尺度 9 点网格、slope/AUC/r* 三 readout、33 契约 npz、六项描述统计、只出矩阵）。与 N×M-time@32（弱而特异 2.11）和 O×M-PSD（强而弥散 0.99）构成三方对照；25 点窗更贴近 OU 先验 24 ms 记忆区，读出更偏边界先验通路

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
