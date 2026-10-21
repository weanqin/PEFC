#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task.py start consent gate (project-local, ZCode).

PreToolUse hook on Bash, appended AFTER the stock shell-session hook in
.zcode/config.json. When the pending command runs `task.py start`, every
start instance's target task directory must carry a substantive consent.md
(1.4 gate evidence: briefing points + the user's explicit confirmation, with
timestamp). Missing or empty -> exit 2 with write-back guidance. Present ->
allow. `create` / `finish` / `archive` and every non-start command pass
through untouched, and a start the user runs in their own terminal never
sees this hook — it only guards the agent's Bash tool.

Task-ref resolution mirrors common/task_utils.resolve_task_dir semantics:

  - path-form ref (absolute, or containing "/" / starting with ".trellis")
    resolves directly against the repo root; a candidate outside
    .trellis/tasks/ or a non-existent directory fails OPEN (task.py reports
    the same-semantics error itself);
  - name-form ref goes exact-match first, then unique `*-<ref>` suffix match
    (find_task_by_name); ambiguous or no hit fails OPEN the same way.

Failure safety is the mirror image of the edit gate (design §2.2): start is
additionally guarded by task.py validation and session identity, and a
false block wedges a legitimate launch — so ANY anomaly fails OPEN with a
stderr warning. Exit codes are only ever 0/2. Registered in
.zcode/config.json (PreToolUse, matcher "Bash").
"""
from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

# Hook hosts send UTF-8 JSON regardless of the process locale.
_stdin_reconfigure = getattr(sys.stdin, "reconfigure", None)
if callable(_stdin_reconfigure):
    try:
        _stdin_reconfigure(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        pass


DIR_WORKFLOW = ".trellis"
DIR_TASKS = "tasks"
TOOL_INPUT_KEYS = ("tool_input", "toolInput")
START_SUBCOMMAND = "start"

GUIDANCE = (
    "1.4 门未满足：先派 trellis-review（复杂任务）→ 向用户呈现 briefing"
    "（改动点/关键逻辑/潜在风险/路径与环境/验证方案）→ 用户明确同意后把同意记录写入"
    " {TASK_DIR}/consent.md（含用户确认原话与时间）→ 重试 start。"
)


def _string_value(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _escapes_enabled() -> bool:
    return (
        os.environ.get("TRELLIS_HOOKS") == "0"
        or os.environ.get("TRELLIS_DISABLE_HOOKS") == "1"
    )


def _script_project_root() -> Path | None:
    """Project root inferred from where this hook script is installed."""
    try:
        here = Path(__file__).resolve()
    except OSError:
        return None
    for parent in here.parents:
        if (parent / DIR_WORKFLOW).is_dir():
            return parent
    return None


def _pending_command(hook_input: dict[str, Any]) -> str | None:
    """Read the pending shell command out of either hook payload shape."""
    command = _string_value(hook_input.get("command"))
    if command:
        return command
    for key in TOOL_INPUT_KEYS:
        tool_input = hook_input.get(key)
        if not isinstance(tool_input, dict):
            continue
        command = _string_value(tool_input.get("command"))
        if command:
            return command
    return None


def _extract_start_instances(command: str) -> list[str | None]:
    """Return the task_ref (or None) of every `task.py start` instance.

    Mirrors the stock `_extract_task_subcommands` scan, with one difference:
    the ref is taken by scanning FORWARD past `-`-prefixed tokens, so
    `task.py start --allow-empty-context <ref>` resolves <ref> instead of the
    flag. No positional arg -> None (task.py reports that itself).
    """
    try:
        tokens = shlex.split(command, posix=os.name != "nt")
    except ValueError:
        return []
    instances: list[str | None] = []
    for index, token in enumerate(tokens[:-1]):
        if Path(token.strip("\"'")).name != "task.py":
            continue
        if tokens[index + 1] != START_SUBCOMMAND:
            continue
        task_ref: str | None = None
        for candidate in tokens[index + 2:]:
            if candidate.startswith("-"):
                continue
            task_ref = candidate
            break
        instances.append(task_ref)
    return instances


def _find_task_by_name(name: str, tasks_dir: Path) -> Path | None:
    """Mirror common/task_utils.find_task_by_name (without its stderr chatter).

    Exact directory-name match first, then a unique `*-<name>` suffix match.
    Ambiguous or absent -> None (fail-open; task.py prints the same errors).
    """
    if not name or name in (".", ".."):
        return None
    exact = tasks_dir / name
    if exact.is_dir():
        return exact
    try:
        matches = sorted(
            d for d in tasks_dir.iterdir()
            if d.is_dir() and d.name.endswith(f"-{name}")
        )
    except OSError:
        return None
    if len(matches) == 1:
        return matches[0]
    return None


def _resolve_task_dir(task_ref: str, repo_root: Path) -> Path | None:
    """Mirror common/task_utils.resolve_task_dir containment semantics.

    Returns the task directory spelled through the repo's own tasks dir, or
    None when task.py itself would refuse the ref (outside .trellis/tasks/,
    the tasks dir itself, traversal, unresolvable). Existence is NOT checked
    here — the caller decides fail-open on a missing directory.
    """
    if not task_ref:
        return None
    normalized = task_ref.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]

    tasks_lexical = repo_root / DIR_WORKFLOW / DIR_TASKS

    if Path(task_ref).is_absolute():
        candidate = Path(task_ref)
    elif "/" in normalized or normalized.startswith(DIR_WORKFLOW):
        candidate = repo_root / Path(normalized)
    else:
        return _find_task_by_name(task_ref, tasks_lexical)

    try:
        resolved = candidate.resolve()
        tasks_resolved = tasks_lexical.resolve()
    except (OSError, RuntimeError):
        return None
    if resolved == tasks_resolved:
        return None
    if tasks_resolved not in resolved.parents:
        return None
    return tasks_lexical / resolved.relative_to(tasks_resolved)


def _has_substantive_content(consent_path: Path) -> bool:
    """True when consent.md exists and carries >=1 non-empty, non-heading line."""
    try:
        text = consent_path.read_text(encoding="utf-8")
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return True
    return False


def _warn(message: str) -> None:
    print(f"[trellis start 同意门] {message}", file=sys.stderr)


def _judge(hook_input: dict[str, Any]) -> int:
    command = _pending_command(hook_input)
    if not command:
        return 0
    instances = _extract_start_instances(command)
    if not instances:
        return 0

    root = _script_project_root()
    if root is None:
        _warn("无法从 hook 安装位置定位 .trellis 项目根，fail-open 放行。")
        return 0

    missing: list[tuple[str, Path]] = []
    for task_ref in instances:
        if task_ref is None:
            continue  # no positional arg: task.py errors on its own
        task_dir = _resolve_task_dir(task_ref, root)
        if task_dir is None or not task_dir.is_dir():
            continue  # unresolved: task.py reports the same-semantics error
        if _has_substantive_content(task_dir / "consent.md"):
            continue
        missing.append((task_ref, task_dir))
    if not missing:
        return 0

    listed = "\n".join(
        f"  - {ref} → {task_dir / 'consent.md'} 缺失或无实质内容"
        for ref, task_dir in missing
    )
    print(
        "[trellis start 同意门] 已拦截 task.py start（1.4 同意证据缺失）：\n"
        f"{listed}\n{GUIDANCE}",
        file=sys.stderr,
    )
    return 2


def main() -> int:
    if _escapes_enabled():
        return 0
    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
        _warn(f"stdin 不是合法 JSON（{type(exc).__name__}），fail-open 放行。")
        return 0
    if not isinstance(hook_input, dict):
        _warn("stdin 顶层不是 JSON 对象，fail-open 放行。")
        return 0

    try:
        return _judge(hook_input)
    except Exception as exc:  # noqa: BLE001 — fail-open catch-all, exit codes only 0/2
        _warn(f"判定异常 fail-open 放行（{type(exc).__name__}: {exc}）。")
        return 0


if __name__ == "__main__":
    sys.exit(main())
