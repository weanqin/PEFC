#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iron rules injection hook (project-local, ZCode).

Registered FIRST in the SessionStart array of .zcode/config.json (config
order = injection order), so the three hard gates are announced before the
workflow summary and the research memory. Fixed text, <= 15 lines (design
§2.4 of task 10-21-trellis-hard-gates). Output format matches
inject-research-context.py: hookSpecificOutput.additionalContext.

Failure is always a silent no-op — a missing rules block must never break
session startup.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import json
import os
import sys


def _should_skip() -> bool:
    if os.environ.get("TRELLIS_HOOKS") == "0":
        return True
    if os.environ.get("TRELLIS_DISABLE_HOOKS") == "1":
        return True
    if os.environ.get("ZCODE_NON_INTERACTIVE") == "1":
        return True
    return False


RULES_TEXT = (
    "【Trellis 硬门 · 铁律】\n"
    "1. 无 in_progress 任务不得写入代码路径——Edit/Write 等文件工具已被 hook 硬拦"
    "（经 Bash 重定向/heredoc/sed -i 写文件同样违反本条，只是技术旁路，不改变约定）；"
    "小改动也须先立轻量任务。\n"
    "2. task.py start 前必须走完 review → briefing（改动点/关键逻辑/潜在风险）→ "
    "用户明确同意；consent.md 缺失时 start 已被 hook 硬拦。\n"
    "3. 有未提交变更时 Stop 会被拦——先完成收尾三件套（2.2 check / 3.3 spec-update / "
    "3.4 commit）再停；非本任务脏路径不擅动，需暂停走 task.py finish"
    "（任务保持 in_progress，只清会话指针）。\n"
    "4. 误拦申诉：TRELLIS_HOOKS=0 只能由用户在启动环境设置（agent 无法为单次调用自行设置）；"
    "详细流程见 .trellis/workflow.md，按需 `get_context.py --mode phase --step X.Y`。"
)


def main() -> None:
    if _should_skip():
        sys.exit(0)
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": RULES_TEXT,
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
