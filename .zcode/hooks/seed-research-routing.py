#!/usr/bin/env python3
"""Seed the 子研究路由 (research routing) section into a newly created task's prd.md.

Wired as a `hooks.after_create` command in .trellis/config.yaml. task.py invokes
it with TASK_JSON_PATH (absolute path to the new task's task.json) and cwd=repo
root. Fail-open: any problem exits non-zero with a stderr note; the lifecycle
command only warns.

Idempotent: the section is appended only when the `trellis-research-routing`
marker is absent, so re-runs and manual re-creation never duplicate it.

The classification itself is NOT done here — a hook cannot read intent. The
hook only guarantees the routing block exists with 待确认/待检测 defaults; the
main agent fills it during Phase 1.0b/1.0c (see .trellis/workflow.md).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MARKER = "<!-- trellis-research-routing -->"

ROUTING_SECTION = f"""

## 子研究路由（1.0b / 1.0c）

{MARKER}

> 任务创建时由 hook 自动播种。规划阶段必须填写以下四项；全部落定前不得进入 1.4 start 门禁（见 `.trellis/workflow.md` Phase 1.0b / 1.0c）。

- **分类结果**：`待确认` —— 可选值：`study1`（EEG 代理脑，框架 Study I）/ `study2`（EC 推断算法，框架 Study II）/ `待确认`（歧义或非科研任务；不得强行归类，须向用户确认）
- **依据**：<一句话，从任务描述判定的理由；待确认时写明歧义点>
- **新路径/方法检测（1.0c）**：`待检测` —— 可选值：`不涉及` / `已有路径：<对照 .trellis/research/context.md 技术路线表填写>` / `新路径：<slug> → research 已触发|待触发`
- **research 产物**：`无` —— 涉及新路径/方法时填 `research/<slug>-adaptation.md`（适配结论 / 风险 / 可执行方案）
"""


def main() -> int:
    task_json_path = os.environ.get("TASK_JSON_PATH", "")
    if not task_json_path:
        print("seed-research-routing: TASK_JSON_PATH not set", file=sys.stderr)
        return 1

    task_dir = Path(task_json_path).resolve().parent
    if not task_dir.is_dir():
        print(f"seed-research-routing: task dir missing: {task_dir}", file=sys.stderr)
        return 1

    prd_path = task_dir / "prd.md"
    if prd_path.exists():
        if MARKER in prd_path.read_text(encoding="utf-8"):
            return 0  # already seeded
        with prd_path.open("a", encoding="utf-8") as f:
            f.write(ROUTING_SECTION)
    else:
        # prd.md should exist by the time after_create fires; create a minimal
        # carrier rather than losing the routing block.
        prd_path.write_text(f"# PRD\n{ROUTING_SECTION}", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
