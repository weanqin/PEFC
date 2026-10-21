#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrap-up gate (project-local, ZCode): Stop must not land mid-task with a
dirty tree.

Stop hook (no matcher). Decision order (design §2.3 of task
10-21-trellis-hard-gates):

  1. escape switches (TRELLIS_HOOKS=0 / TRELLIS_DISABLE_HOOKS=1) -> allow;
  2. resolve the active task through the SAME three-level identity chain as
     the edit gate (① payload resolve_context_key → ② env-aware resolver →
     ③ single fresh-session fallback, 48h mtime window, diagnostics buffer);
     no task or status != in_progress -> silent allow;
  3. `git -c core.quotepath=false status --porcelain` at the repo root;
     .trellis/.runtime/ is gitignored (via .trellis/.gitignore) so porcelain
     hides it already — a defensive prefix filter is kept anyway. Dirty
     paths outside runtime -> exit 2 with: (a) the first 10 dirty paths,
     (b) the Phase 3.4 two-group protocol (this session's AI changes -> a
     batched commit plan for user confirmation; unrecognized files -> listed
     separately for include/exclude, never silently committed), (c) the
     pause exit (`task.py finish` clears only this session's pointer, the
     task stays in_progress and resumable, and Stop is then allowed);
  4. nothing dirty after filtering -> allow.

Failure safety mirrors the start gate, not the edit gate: Stop is the
session's exit and a false block costs more than a missed reminder, so ANY
anomaly fails OPEN silently. Exit codes are only ever 0/2 (1 would be
logged as a hook error). The platform's own 3-continuation cap bounds any
loop. Registered in .zcode/config.json (Stop).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
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
DIR_RUNTIME = ".runtime"
DIR_SESSIONS = "sessions"
SESSION_FRESH_SECONDS = 48 * 3600
RUNTIME_PREFIX = ".trellis/.runtime/"
DIRTY_SUMMARY_LIMIT = 10
GIT_TIMEOUT_SECONDS = 10


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


def _host_platform_name() -> str | None:
    """Name the host from the config directory this hook was installed into."""
    for part in reversed(Path(sys.argv[0]).parts):
        if part.startswith(".") and part not in (".", "..") and len(part) > 1:
            return part[1:]
    return None


def _load_resolve_context_key(root: Path):
    scripts_dir = root / DIR_WORKFLOW / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from common.active_task import resolve_context_key  # type: ignore[import-not-found]

    return resolve_context_key


def _read_session_task_ref(session_path: Path) -> str | None:
    try:
        data = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return _string_value(data.get("current_task"))


def _session_task_ref(sessions_dir: Path, key: str | None) -> str | None:
    if not key:
        return None
    return _read_session_task_ref(sessions_dir / f"{key}.json")


def _resolve_active_task(root: Path, hook_input: dict[str, Any]) -> str | None:
    """Same three-level chain as gate-file-edits.py (shared contract)."""
    sessions_dir = root / DIR_WORKFLOW / DIR_RUNTIME / DIR_SESSIONS
    try:
        resolve_context_key = _load_resolve_context_key(root)
        platform = _host_platform_name()

        key = resolve_context_key(
            hook_input, platform=platform, allow_environment_context=False
        )
        ref = _session_task_ref(sessions_dir, key)
        if ref:
            return ref

        key = resolve_context_key(
            hook_input, platform=platform, allow_environment_context=True
        )
        ref = _session_task_ref(sessions_dir, key)
        if ref:
            return ref
    except Exception:
        pass

    now = time.time()
    fresh: list[Path] = []
    try:
        for path in sorted(sessions_dir.glob("*.json")):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if now - mtime <= SESSION_FRESH_SECONDS:
                fresh.append(path)
    except OSError:
        pass
    if len(fresh) == 1:
        return _read_session_task_ref(fresh[0])
    return None


def _task_status(root: Path, task_ref: str) -> str | None:
    task_dir = Path(task_ref)
    if not task_dir.is_absolute():
        task_dir = root / task_dir
    try:
        data = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return _string_value(data.get("status"))


def _dirty_paths(root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(root), "-c", "core.quotepath=false", "status", "--porcelain"],
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git status exited {proc.returncode}")
    paths: list[str] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        entry = line[3:] if len(line) > 3 else line.strip()
        if " -> " in entry:
            entry = entry.split(" -> ", 1)[1]
        entry = entry.strip().strip('"')
        if entry:
            paths.append(entry)
    # .trellis/.runtime/ is gitignored and never shown by porcelain; the
    # filter is belt-and-suspenders, not the primary mechanism.
    return [p for p in paths if not p.startswith(RUNTIME_PREFIX)]


def _denied_message(task_ref: str, dirty: list[str]) -> str:
    shown = dirty[:DIRTY_SUMMARY_LIMIT]
    more = "" if len(dirty) <= DIRTY_SUMMARY_LIMIT else f"\n  …（其余 {len(dirty) - DIRTY_SUMMARY_LIMIT} 条略）"
    listing = "\n".join(f"  {p}" for p in shown) + more
    return (
        f"[trellis 收尾门] 活动任务 {task_ref} 仍为 in_progress 且工作区有未提交变更，Stop 暂被拦截。\n"
        f"脏路径（{len(dirty)} 条，前 {min(len(dirty), DIRTY_SUMMARY_LIMIT)} 条）：\n{listing}\n"
        "收尾三件套（workflow 2.2 / 3.3 / 3.4）——\n"
        "  - 本会话 AI 改动：按逻辑分组起草提交计划，向用户一次征求确认后分批提交；\n"
        "  - 未识别脏文件（他任务遗留 / 用户手改）：单列出征求 include/exclude，不得擅入提交。\n"
        "出路：非本任务脏路径不要求本次处理；需暂停请 `task.py finish`"
        "（只清本会话指针，任务保持 in_progress 可恢复），之后 Stop 放行。"
    )


def _judge(hook_input: dict[str, Any]) -> int:
    root = _script_project_root()
    if root is None:
        return 0  # fail-open: nothing to guard on

    task_ref = _resolve_active_task(root, hook_input)
    if not task_ref:
        return 0
    status = _task_status(root, task_ref)
    if status != "in_progress":
        return 0

    dirty = _dirty_paths(root)
    if not dirty:
        return 0
    print(_denied_message(task_ref, dirty), file=sys.stderr)
    return 2


def main() -> int:
    if _escapes_enabled():
        return 0
    try:
        hook_input = json.loads(sys.stdin.read())
        if not isinstance(hook_input, dict):
            hook_input = {}
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        return 0  # fail-open: silent allow
    try:
        return _judge(hook_input)
    except Exception:  # noqa: BLE001 — fail-open catch-all, exit codes only 0/2
        return 0


if __name__ == "__main__":
    sys.exit(main())
