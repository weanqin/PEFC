#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edit gate (project-local, ZCode): no code writes without an in_progress task.

PreToolUse hook on Edit/Write/MultiEdit/NotebookEdit/ApplyPatch. Decision
order (design §2.1 of task 10-21-trellis-hard-gates):

  1. escape switches (TRELLIS_HOOKS=0 / TRELLIS_DISABLE_HOOKS=1, read from
     this hook process's own env — setting them inside the command string
     has no effect) -> allow;
  2. resolve the trellis root from this script's install location;
  3. normalize the target path repo-relative; the always-allow set is ONLY
     .trellis/tasks/**, .trellis/research/**, .trellis/workspace/**
     (planning artifacts / research memory / journals). Anything outside the
     repo or outside that set needs an in_progress task — including absolute
     targets like /root/autodl-tmp/**;
  4. resolve the active task through a three-level identity chain (each
     level's outcome recorded into a diagnostics buffer):
       ① payload identity via common.active_task.resolve_context_key
          (allow_environment_context=False);
       ② the same resolver with allow_environment_context=True (reads
          TRELLIS_CONTEXT_ID and the platform env table internally — this
          hook never inspects env var names itself);
       ③ single-fresh-session fallback: adopted only when exactly one
          session file under .trellis/.runtime/sessions/ has mtime <= 48h
          (covers sub-agents that cannot inherit the parent identity; stale
          pointers are excluded by the mtime window).
  5. status == "in_progress" -> allow; otherwise exit 2 with a stderr block
     naming the path, the state, the chain diagnostics and the remediation.

Failure safety is deliberately asymmetric: this gate's whole point is that a
missing task must not let code writes through, so ANY anomaly fails CLOSED
(exit 2 + exception summary + escape note). Exit codes are only ever 0/2 —
any other non-zero would be logged as a hook error by the platform instead
of a denial. Registered in .zcode/config.json (PreToolUse).
"""
from __future__ import annotations

import json
import os
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
# Tool-call payloads nest the target one level down; same casing spread as the
# stock shell-session hook.
TOOL_INPUT_KEYS = ("tool_input", "toolInput")
# file editors first, then NotebookEdit's key, then a generic fallback.
TARGET_PATH_KEYS = ("file_path", "filePath", "notebook_path", "notebookPath", "path")
# The always-allow set is exactly these three trees — planning artifacts,
# research memory, journals. NOT the whole .trellis/** (spec/, scripts/,
# workflow.md all need an in_progress task).
ALWAYS_ALLOW_PREFIXES = (
    ".trellis/tasks/",
    ".trellis/research/",
    ".trellis/workspace/",
)
SESSION_FRESH_SECONDS = 48 * 3600


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
    """Project root inferred from where this hook script is installed.

    Hooks ship at <project-root>/.zcode/hooks/, so walking up from __file__
    anchors the project no matter where the session has drifted (same
    convention as the stock hooks).
    """
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


def _extract_target_path(hook_input: dict[str, Any]) -> str | None:
    for key in TOOL_INPUT_KEYS:
        tool_input = hook_input.get(key)
        if not isinstance(tool_input, dict):
            continue
        for path_key in TARGET_PATH_KEYS:
            value = _string_value(tool_input.get(path_key))
            if value:
                return value
    return None


def _repo_relative(root: Path, target: str) -> str | None:
    """Repo-relative posix path for `target`, or None when it sits outside.

    Symlink-aware: a target that resolves inside the repo is mapped back even
    when its lexical spelling differs; a target outside both forms is None.
    """
    path = Path(target)
    if not path.is_absolute():
        path = root / path
    try:
        root_real = root.resolve()
    except OSError:
        return None
    normalized = Path(os.path.normpath(str(path)))
    try:
        return normalized.resolve().relative_to(root_real).as_posix()
    except (OSError, ValueError):
        pass
    try:
        return normalized.relative_to(root_real).as_posix()
    except ValueError:
        return None


def _in_allow_set(repo_rel: str) -> bool:
    if repo_rel in ("", "."):
        return False
    for prefix in ALWAYS_ALLOW_PREFIXES:
        stem = prefix.rstrip("/")
        if repo_rel == stem or repo_rel.startswith(prefix):
            return True
    return False


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


def _resolve_active_task(
    root: Path,
    hook_input: dict[str, Any],
    chain: list[str],
) -> str | None:
    """Three-level identity chain; returns the current_task ref or None.

    Every level's outcome is appended to `chain` for the denial diagnostics.
    """
    sessions_dir = root / DIR_WORKFLOW / DIR_RUNTIME / DIR_SESSIONS
    try:
        resolve_context_key = _load_resolve_context_key(root)
        platform = _host_platform_name()

        # ① payload identity only.
        key = resolve_context_key(
            hook_input, platform=platform, allow_environment_context=False
        )
        ref = _session_task_ref(sessions_dir, key)
        chain.append(f"①payload: key={key or '∅'} → " + (f"任务 {ref}" if ref else "无会话指针"))
        if ref:
            return ref

        # ② adds TRELLIS_CONTEXT_ID / platform env table / shell tickets,
        #    all inside the stock resolver — never checked by name here.
        key = resolve_context_key(
            hook_input, platform=platform, allow_environment_context=True
        )
        ref = _session_task_ref(sessions_dir, key)
        chain.append(f"②env: key={key or '∅'} → " + (f"任务 {ref}" if ref else "无会话指针"))
        if ref:
            return ref
    except Exception as exc:  # noqa: BLE001 — diagnostics, not a control flow
        chain.append(f"①②身份解析异常: {type(exc).__name__}: {exc}")

    # ③ single fresh session fallback (mtime <= 48h, exactly one).
    now = time.time()
    total = 0
    fresh: list[Path] = []
    try:
        for path in sorted(sessions_dir.glob("*.json")):
            total += 1
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if now - mtime <= SESSION_FRESH_SECONDS:
                fresh.append(path)
    except OSError:
        pass
    summary = f"③fallback: 会话文件 {total} 个 / 新鲜(≤48h) {len(fresh)} 个"
    if len(fresh) == 1:
        ref = _read_session_task_ref(fresh[0])
        chain.append(
            summary + " → 唯一新鲜 " + fresh[0].name + " → "
            + (f"任务 {ref}" if ref else "无 current_task")
        )
        return ref
    chain.append(summary + " → 非『恰好一个新鲜会话』，拒绝")
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


def _deny(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def _escape_note() -> str:
    return (
        "确认误拦时，由用户在启动环境设置 TRELLIS_HOOKS=0 重启会话"
        "（agent 无法为单次调用自行设置该开关）。"
    )


def _fail_closed_message(reason: str) -> str:
    return (
        f"[trellis 编辑门] 判定异常，按 fail-closed 拦截本次写入。原因：{reason}\n"
        "补救：" + _escape_note()
    )


def _denied_message(path: str, state_line: str, chain: list[str]) -> str:
    chain_text = "\n".join(f"  {line}" for line in chain)
    return (
        f"[trellis 编辑门] 已拦截对 {path} 的写入：{state_line}\n"
        f"身份链路诊断：\n{chain_text}\n"
        "补救：先分类本轮请求；小改动也须立轻量任务并走 1.4 review→briefing→用户同意门"
        "后再 task.py start。经 Bash 重定向/heredoc/sed -i 写文件同样违反约定"
        "（技术旁路不改变约定）。" + _escape_note()
    )


def _state_line(task_ref: str | None, status: str | None) -> str:
    if task_ref is None:
        return "当前无 in_progress 任务"
    if status is None:
        return f"活动任务 {task_ref} 的 task.json 不可读（按非 in_progress 处理）"
    if status == "planning":
        return f"活动任务 {task_ref} 处于 planning（尚未 task.py start）"
    return f"活动任务 {task_ref} 状态为 {status}（非 in_progress）"


def _judge(hook_input: dict[str, Any]) -> int:
    root = _script_project_root()
    if root is None:
        return _deny(_fail_closed_message("无法从 hook 安装位置定位 .trellis 项目根"))

    target = _extract_target_path(hook_input)
    if target is None:
        chain = [f"载荷未携带目标路径字段（已查 {', '.join(TARGET_PATH_KEYS)}）"]
        return _deny(
            _denied_message("(未能从 hook 载荷提取目标路径)", "无法判定目标，按 fail-closed 拦截", chain)
        )

    repo_rel = _repo_relative(root, target)
    if repo_rel is not None and _in_allow_set(repo_rel):
        return 0

    chain: list[str] = []
    task_ref = _resolve_active_task(root, hook_input, chain)
    status = _task_status(root, task_ref) if task_ref else None
    if task_ref and status == "in_progress":
        return 0
    return _deny(_denied_message(target, _state_line(task_ref, status), chain))


def main() -> int:
    if _escapes_enabled():
        return 0
    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
        return _deny(_fail_closed_message(f"hook stdin 不是合法 JSON（{type(exc).__name__}）"))
    if not isinstance(hook_input, dict):
        return _deny(_fail_closed_message("hook stdin 顶层不是 JSON 对象"))

    try:
        return _judge(hook_input)
    except Exception as exc:  # noqa: BLE001 — fail-closed catch-all, exit codes only 0/2
        return _deny(_fail_closed_message(f"{type(exc).__name__}: {exc}"))


if __name__ == "__main__":
    sys.exit(main())
