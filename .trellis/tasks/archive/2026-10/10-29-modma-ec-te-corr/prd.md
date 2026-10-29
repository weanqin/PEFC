# MODMA 5 折 EC_pert×TE-EC 两级相关 + fold0 同口径补全

## Goal

补训 fold0（07-05 架构损失口径：std graph-mexhat flow-only、50ep/b64/p10/lr1e-3/seed42、复用旧库 fold0 VAE）+ v4 网格 EC（0.48×24/σ_ref0.3/q0.5）；随后对 refs/symbolic_transfer_entropy modma5fold 档计算 EC_pert×TE-EC：逐被试 Pearson/Spearman+p 值、每折组平均矩阵 Pearson/Spearman+p 值，表格汇总（fold1-4 EC 用 10-29 产物，TE∩EC 取交集）。用户 2026-10-30 完整指令为同意凭据。

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
