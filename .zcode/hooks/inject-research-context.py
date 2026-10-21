#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Research Context Injection Hook (project-local, ZCode)

Injects this project's research memory at SessionStart so every new
conversation starts aligned with the research context:

  .trellis/research/context.md      -> only the leading "## 速览" section
                                       (full text when the section is absent)
  .trellis/research/progress.md     -> only the leading "## 当前状态" section
  .trellis/research/environment.md  -> only the leading "## 速览" section
                                       (full text when the section is absent)

Every injected block ends with a pointer line naming the full archive so the
agent can Read it on demand. The full-text fallback keeps pre-速览 files
behaving exactly as before (migration safety).

Project-local addition alongside the stock session-start.py; a missing or
partial research memory must never break session startup, so every failure
path is a silent no-op. Registered in .zcode/config.json (SessionStart).
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
import os
import sys
from pathlib import Path

# Files carrying this marker are unfilled initialization templates.
UNINITIALIZED_MARKER = "<!-- trellis-research: uninitialized -->"


def _should_skip() -> bool:
    if os.environ.get("TRELLIS_HOOKS") == "0":
        return True
    if os.environ.get("TRELLIS_DISABLE_HOOKS") == "1":
        return True
    if os.environ.get("ZCODE_NON_INTERACTIVE") == "1":
        return True
    return False


def _read(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""
    # Uninitialized template: carry the marker -> treat as absent so placeholder
    # bodies never leak into session context.
    if UNINITIALIZED_MARKER in text:
        return ""
    return text


def _extract_section(text: str, header_prefix: str) -> str:
    """Return the body of the section whose '## ' header starts with header_prefix,
    up to the next '## ' header. Empty string when the section is absent."""
    lines = text.splitlines()
    out: list[str] = []
    inside = False
    for line in lines:
        if not inside and line.startswith("## ") and line[3:].startswith(header_prefix):
            inside = True
            continue
        if inside and line.startswith("## "):
            break
        if inside:
            out.append(line)
    return "\n".join(out).strip()


def _overview_or_full(text: str, pointer: str) -> str:
    """Inject only the leading 「速览」 section, with a pointer to the archive.

    Falls back to the full text when the section is absent, so files not yet
    carrying a 速览 keep the historical full-injection behavior. The pointer
    line names the full archive for on-demand Read.
    """
    overview = _extract_section(text, "速览")
    body = "## 速览\n" + overview if overview else text
    return body + "\n(" + pointer + ")"


def _script_project_root() -> Path | None:
    """Project root inferred from where this hook script is installed.

    Hooks ship at <project-root>/.zcode/hooks/, so walking up from __file__
    anchors the project no matter where the session has drifted: in this
    client ${ZCODE_PROJECT_DIR} inside the command string expands to empty
    and payload cwd points outside the project after the agent cd's out.
    Returns None outside a Trellis checkout so callers fall back to env
    vars / cwd as before.
    """
    try:
        here = Path(__file__).resolve()
    except OSError:
        return None
    for parent in here.parents:
        if (parent / ".trellis").is_dir():
            return parent
    return None


def main() -> None:
    if _should_skip():
        sys.exit(0)

    try:
        hook_input = json.loads(sys.stdin.read())
        if not isinstance(hook_input, dict):
            hook_input = {}
    except (json.JSONDecodeError, ValueError):
        hook_input = {}

    project_dir = _script_project_root()
    if project_dir is None:
        for var in ("ZCODE_PROJECT_DIR", "CLAUDE_PROJECT_DIR"):
            val = os.environ.get(var)
            if val:
                project_dir = Path(val).resolve()
                break
    if project_dir is None:
        project_dir = Path(str(hook_input.get("cwd", "."))).resolve()

    research = project_dir / ".trellis" / "research"
    blocks: list[str] = []

    context = _read(research / "context.md")
    if context:
        blocks.append(
            "<research-context>\n"
            + _overview_or_full(
                context, "完整档案: .trellis/research/context.md；写/跑代码前按需 Read"
            )
            + "\n</research-context>"
        )

    progress_head = _extract_section(_read(research / "progress.md"), "当前状态")
    if progress_head:
        blocks.append(
            "<research-progress>\n## 当前状态\n"
            + progress_head
            + "\n(完整看板: .trellis/research/progress.md；节点完成后当场更新)\n</research-progress>"
        )

    env = _read(research / "environment.md")
    if env:
        blocks.append(
            "<research-environment>\n"
            + _overview_or_full(
                env, "完整档案: .trellis/research/environment.md；写/跑代码前按需 Read"
            )
            + "\n</research-environment>"
        )

    if not blocks:
        # Research memory not initialized (or absent). Emit a guard hint instead
        # of staying silent: the next session must ASK the user for research
        # direction materials rather than inferring them from the repository.
        if not research.is_dir():
            sys.exit(0)
        text = (
            "<research-memory>\n"
            "状态：研究记忆未初始化（.trellis/research/ 下档案为占位模板）。\n"
            "规则：开展任何研究方向相关工作前，先请用户提供研究方向资料（路径或内容），"
            "据实填写 .trellis/research/{context,progress,environment}.md 并删除各文件顶部的\n"
            + UNINITIALIZED_MARKER + " 标记。\n"
            "在此之前，不得从仓库文件自行推断或填充研究方向内容；运行代码的环境未记录在案时，必须先问用户。\n"
            "</research-memory>"
        )
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": text,
                    }
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        sys.exit(0)

    text = (
        "<research-memory>\n"
        "科研工作记忆（hook 注入的长期档案，非聊天记录）。维护规则：事实/进度/环境变化时"
        "由 AI 当场更新对应源文件；写或运行任何代码前先按环境契约确认 env，未记录的场景必须先问用户。\n\n"
        + "\n\n".join(blocks)
        + "\n</research-memory>"
    )

    # ZCode reads hookSpecificOutput.additionalContext; emitting the top-level
    # additional_context key too would duplicate the payload (see session-start.py).
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": text,
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
