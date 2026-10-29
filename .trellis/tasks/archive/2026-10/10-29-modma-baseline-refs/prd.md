# MODMA 5 折 baseline(gc/gpd/te)注册入 refs

## Goal

把用户下载的旧库原版 MODMA baseline 矩阵（/root/autodl-tmp/modma-gc-ec.zip 解包：granger_bivariate_spectral / gpdc / symbolic_te × fold0-4，AIC8/1-125Hz/nf65/fs250/全长）按 refs 归档规范注册：三方法独立子目录、原文件名 fold 子目录、SHA256SUMS、meta.json（含被试映射与三方逐位验证凭据）、可重放 register_refs.py，并在 MANIFEST.md 追加三行。用户 2026-10-30 明确指令为同意凭据。

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
