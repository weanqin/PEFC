#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stdin scenario matrix T1–T14 for task 10-21-trellis-hard-gates (S7, AC1).

Drives the four project-local hooks exactly as the platform would:
`printf <json> | python3 <hook>` via subprocess with a controlled env.

Test fixtures (all cleaned up in finally):
  - fake task dirs   .trellis/tasks/zz-test-*        (task.json with chosen status)
  - temp session files .trellis/.runtime/sessions/*.json (mtime via os.utime)
  - the REAL session file(s) are never touched content-wise; for "no active
    session" / "exactly one fresh" scenarios their mtime is temporarily
    pushed >48h into the past (os.utime) and restored afterwards.

Run: python3 .trellis/tasks/10-21-trellis-hard-gates/research/test-gates.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/root/npi")
HOOKS = ROOT / ".zcode" / "hooks"
TASKS = ROOT / ".trellis" / "tasks"
SESSIONS = ROOT / ".trellis" / ".runtime" / "sessions"

sys.path.insert(0, str(ROOT / ".trellis" / "scripts"))
from common.active_task import resolve_context_key  # noqa: E402

GATE_EDIT = HOOKS / "gate-file-edits.py"
GATE_START = HOOKS / "gate-task-start-consent.py"
GATE_WRAP = HOOKS / "gate-wrapup-check.py"
HOOK_IRON = HOOKS / "inject-iron-rules.py"
HOOK_RESEARCH = HOOKS / "inject-research-context.py"

STALE_EPOCH = time.time() - 96 * 3600  # > 48h window

PAYLOAD_KEY = resolve_context_key(
    {"session_id": "zz-test-payload-session"}, platform="zcode",
    allow_environment_context=False,
)
ENV_KEY = "zz-test-env-key"  # TRELLIS_CONTEXT_ID passes through _sanitize_key

PLAN_TASK = "zz-test-plan-task"
RUN_TASK = "zz-test-run-task"
CONSENT_GATE_TASK = "zz-test-consent-gate"
CONSENT_A = "zz-test-consent-a"
CONSENT_B = "zz-test-consent-b"

ENV_STRIP = (
    "TRELLIS_CONTEXT_ID", "CLAUDE_CODE_SESSION_ID", "CLAUDE_SESSION_ID",
    "CODEX_THREAD_ID", "DSH_SESSION_ID", "GEMINI_SESSION_ID", "QODER_SESSION_ID",
    "KIRO_SESSION_ID", "COPILOT_SESSION_ID", "COPILOT_SESSIONID", "SNOW_SESSION_ID",
    "CURSOR_CONVERSATION_ID", "CURSOR_CONVERSATIONID", "CURSOR_TRANSCRIPT_PATH",
    "GEMINI_TRANSCRIPT_PATH", "FACTORY_TRANSCRIPT_PATH", "DROID_TRANSCRIPT_PATH",
    "QODER_TRANSCRIPT_PATH", "CODEBUDDY_TRANSCRIPT_PATH",
)

created_tasks: list[str] = []
created_sessions: list[Path] = []
stashed_real: dict[Path, tuple[float, float]] = {}


def make_task(name: str, status: str) -> Path:
    d = TASKS / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "task.json").write_text(
        json.dumps({"id": name, "name": name, "status": status}), encoding="utf-8"
    )
    if name not in created_tasks:
        created_tasks.append(name)
    return d


def write_session(key: str, task_ref: str, mtime: float | None = None) -> Path:
    SESSIONS.mkdir(parents=True, exist_ok=True)
    p = SESSIONS / f"{key}.json"
    p.write_text(
        json.dumps({"platform": "zcode", "current_task": task_ref}), encoding="utf-8"
    )
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    created_sessions.append(p)
    return p


def stash_real_sessions() -> None:
    """Push every pre-existing session file's mtime >48h back (restored later)."""
    for p in SESSIONS.glob("*.json"):
        if p in created_sessions or p in stashed_real:
            continue
        st = p.stat()
        stashed_real[p] = (st.st_atime, st.st_mtime)
        os.utime(p, (STALE_EPOCH, STALE_EPOCH))


def restore_real_sessions() -> None:
    for p, (a, m) in stashed_real.items():
        try:
            os.utime(p, (a, m))
        except OSError:
            pass
    stashed_real.clear()


def run_hook(hook: Path, payload: dict | None = None, env_overrides: dict | None = None,
             raw: bytes | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    for var in ENV_STRIP:
        env.pop(var, None)
    if env_overrides:
        env.update(env_overrides)
    data = raw if raw is not None else json.dumps(payload or {}).encode("utf-8")
    return subprocess.run(
        [sys.executable, str(hook)], input=data, capture_output=True, env=env, timeout=30
    )


def edit_payload(path: str, tool_name: str = "Edit") -> dict:
    return {"tool_name": tool_name, "tool_input": {"file_path": path}}


results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"{mark}  {name}" + ("" if cond else f"  | {detail}"))


def expect(name: str, proc: subprocess.CompletedProcess, want_rc: int,
           stderr_contains: tuple[str, ...] = (), stdout_empty: bool = False) -> None:
    problems = []
    if proc.returncode != want_rc:
        problems.append(f"rc={proc.returncode} (want {want_rc})")
    err = proc.stderr.decode("utf-8", "replace")
    out = proc.stdout.decode("utf-8", "replace")
    for marker in stderr_contains:
        if marker not in err:
            problems.append(f"stderr missing {marker!r}")
    if stdout_empty and out.strip():
        problems.append(f"stdout not empty: {out[:80]!r}")
    check(name, not problems, "; ".join(problems) + f" || stderr[:200]={err[:200]!r}")


def main() -> int:
    try:
        # ---- fixtures -------------------------------------------------
        make_task(PLAN_TASK, "planning")
        make_task(RUN_TASK, "in_progress")
        make_task(CONSENT_GATE_TASK, "planning")   # consent.md added per-test
        make_task(CONSENT_A, "planning")
        make_task(CONSENT_A, "planning")
        (TASKS / CONSENT_A / "consent.md").write_text(
            "# consent\n\n- 时间：2026-10-21 测试\n- 用户原话：确认开始\n", encoding="utf-8"
        )
        make_task(CONSENT_B, "planning")

        payload_session = write_session(PAYLOAD_KEY, f".trellis/tasks/{PLAN_TASK}")
        env_session = write_session(ENV_KEY, f".trellis/tasks/{RUN_TASK}")
        del payload_session, env_session

        # ---- T1 无活动会话 Edit models/foo.py → 2 ---------------------
        stash_real_sessions()
        p = run_hook(GATE_EDIT, edit_payload("models/foo.py"))
        expect("T1", p, 2, ("models/foo.py", "①payload", "②env", "③fallback", "补救"))

        # ---- T2 allow-set → 0 -----------------------------------------
        p = run_hook(GATE_EDIT, edit_payload(".trellis/tasks/x/prd.md"))
        expect("T2", p, 0)

        # ---- T2b allow-set negative examples → 2 ----------------------
        for rel in (".trellis/spec/x.md", ".zcode/hooks/x.py", ".trellis/scripts/x.py"):
            p = run_hook(GATE_EDIT, edit_payload(rel))
            expect(f"T2b[{rel}]", p, 2)

        # ---- T2c repo-external absolute path → 2 ----------------------
        p = run_hook(GATE_EDIT, edit_payload("/root/autodl-tmp/x/out.png"))
        expect("T2c", p, 2)

        # ---- T5 invalid JSON → 2 (fail-closed + escape note) ----------
        p = run_hook(GATE_EDIT, raw=b"{not json")
        expect("T5", p, 2, ("TRELLIS_HOOKS=0",))

        # ---- T5b escape switch → 0 ------------------------------------
        p = run_hook(GATE_EDIT, edit_payload("models/foo.py"), env_overrides={"TRELLIS_HOOKS": "0"})
        expect("T5b", p, 0)

        # ---- T5c tool-name variants → 2 --------------------------------
        for tool in ("MultiEdit", "ApplyPatch"):
            p = run_hook(GATE_EDIT, edit_payload("models/foo.py", tool))
            expect(f"T5c[{tool}]", p, 2)
        restore_real_sessions()

        # ---- T3 session → planning task → 2 ----------------------------
        stash_real_sessions()
        p = run_hook(GATE_EDIT, {"session_id": "zz-test-payload-session",
                                 **edit_payload("models/foo.py", "Write")})
        expect("T3", p, 2, ("planning",))

        # ---- T4 session → in_progress task → 0 -------------------------
        write_session(PAYLOAD_KEY, f".trellis/tasks/{RUN_TASK}")
        p = run_hook(GATE_EDIT, {"session_id": "zz-test-payload-session",
                                 **edit_payload("models/foo.py")})
        expect("T4", p, 0)

        # ---- T4b env identity fallback → 0 -----------------------------
        p = run_hook(GATE_EDIT, edit_payload("models/foo.py"),
                     env_overrides={"TRELLIS_CONTEXT_ID": ENV_KEY})
        expect("T4b", p, 0)

        # The fallback scenarios need exactly-one-fresh counting: age the two
        # fixture sessions out of the 48h window first.
        for key in (PAYLOAD_KEY, ENV_KEY):
            os.utime(SESSIONS / f"{key}.json", (STALE_EPOCH, STALE_EPOCH))

        # ---- T4c single fresh session fallback → 0 ---------------------
        fresh = write_session("zz-test-fresh-fallback", f".trellis/tasks/{RUN_TASK}", mtime=time.time())
        p = run_hook(GATE_EDIT, edit_payload("models/foo.py"))
        expect("T4c", p, 0)
        fresh.unlink()
        created_sessions.remove(fresh)

        # ---- T4d stale single session fallback → 2 ---------------------
        write_session("zz-test-stale-fallback", f".trellis/tasks/{RUN_TASK}", mtime=STALE_EPOCH)
        p = run_hook(GATE_EDIT, edit_payload("models/foo.py"))
        expect("T4d", p, 2, ("③fallback",))
        restore_real_sessions()

        # ---- consent gate T6/T6b/T6c/T6d/T6e/T7/T7b/T8/T9 --------------
        start_cmd = "python3 ./.trellis/scripts/task.py start "
        p = run_hook(GATE_START, {"tool_input": {"command": start_cmd + CONSENT_GATE_TASK}})
        expect("T6", p, 2, ("consent.md", "1.4", "trellis-review", "briefing", "重试 start"))

        p = run_hook(GATE_START, {"tool_input": {
            "command": start_cmd + "--allow-empty-context " + CONSENT_GATE_TASK}})
        expect("T6b", p, 2, ("consent.md",))

        # T6c as literally designed: bare slug of the REAL task (consent exists
        # in this repo) → suffix resolution + allow. The block-side of the
        # suffix semantics is covered by T6c-block on the slug of the fake task.
        p = run_hook(GATE_START, {"tool_input": {"command": start_cmd + "trellis-hard-gates"}})
        expect("T6c[literal: real task, consent present]", p, 0)
        p = run_hook(GATE_START, {"tool_input": {"command": start_cmd + "consent-gate"}})
        expect("T6c-block[suffix → fake task, no consent]", p, 2, ("consent.md",))

        p = run_hook(GATE_START, {"tool_input": {
            "command": f"{start_cmd}{CONSENT_A} && {start_cmd}{CONSENT_B}"}})
        expect("T6d", p, 2, ("consent.md",))

        p = run_hook(GATE_START, {"tool_input": {
            "command": start_cmd + f".trellis/tasks/{CONSENT_GATE_TASK}"}})
        expect("T6e", p, 2, ("consent.md",))

        consent_file = TASKS / CONSENT_GATE_TASK / "consent.md"
        consent_file.write_text("# consent\n\n- 用户原话：确认开始（T7）\n- 时间：2026-10-21\n", encoding="utf-8")
        p = run_hook(GATE_START, {"tool_input": {"command": start_cmd + CONSENT_GATE_TASK}})
        expect("T7", p, 0)

        consent_file.write_text("# 只有标题行\n", encoding="utf-8")
        p = run_hook(GATE_START, {"tool_input": {"command": start_cmd + CONSENT_GATE_TASK}})
        expect("T7b[heading-only]", p, 2)
        consent_file.write_text("", encoding="utf-8")
        p = run_hook(GATE_START, {"tool_input": {"command": start_cmd + CONSENT_GATE_TASK}})
        expect("T7b[empty]", p, 2)
        consent_file.unlink()

        p = run_hook(GATE_START, {"tool_input": {"command": "python3 ./.trellis/scripts/task.py list"}})
        expect("T8", p, 0)

        p = run_hook(GATE_START, raw=b"{not json")
        expect("T9", p, 0)
        check("T9[stderr warning]", bool(p.stderr.strip()), "stderr empty, expected fail-open warning")

        # ---- wrap-up gate T10/T11 --------------------------------------
        stash_real_sessions()
        p = run_hook(GATE_WRAP, {"session_id": "zz-test-unknown-session"})
        expect("T10", p, 0, stdout_empty=True)
        check("T10[silent]", not p.stderr.strip(), f"stderr={p.stderr[:120]!r}")
        restore_real_sessions()

        p = run_hook(GATE_WRAP, {"session_id": "zz-test-payload-session"})
        expect("T11", p, 2, ("in_progress", "未识别", "include/exclude", "task.py finish"))

        # ---- T13 iron rules --------------------------------------------
        p = run_hook(HOOK_IRON, {})
        ok = p.returncode == 0
        parsed = None
        try:
            parsed = json.loads(p.stdout.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            ok = False
        valid = (
            ok and isinstance(parsed, dict)
            and parsed.get("hookSpecificOutput", {}).get("hookEventName") == "SessionStart"
            and isinstance(parsed["hookSpecificOutput"].get("additionalContext"), str)
        )
        if valid:
            rules_text = parsed["hookSpecificOutput"]["additionalContext"]
            lines = rules_text.count("\n") + 1
        else:
            rules_text = ""
            lines = -1
        check("T13", valid and 0 < lines <= 15,
              f"valid={valid} lines={lines} rc={p.returncode}")
        check("T13[Bash-bypass + finish-exit wording]",
              "Bash 重定向/heredoc/sed -i" in rules_text
              and "task.py finish" in rules_text if valid else False)

        # ---- T14 research hook overview-only ---------------------------
        p = run_hook(HOOK_RESEARCH, {})
        text = ""
        if p.returncode == 0:
            try:
                text = json.loads(p.stdout.decode("utf-8"))["hookSpecificOutput"]["additionalContext"]
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
                text = ""
        check("T14", p.returncode == 0
              and "## 速览" in text
              and "完整档案: .trellis/research/context.md" in text
              and "完整档案: .trellis/research/environment.md" in text
              and "既有结论（已验证）" not in text        # deep context body absent
              and "派发惯例" not in text,                  # deep environment body absent
              f"rc={p.returncode} bytes={len(text.encode('utf-8'))}")
        print(f"T14 injected bytes: {len(text.encode('utf-8'))}")
    finally:
        restore_real_sessions()
        for name in created_tasks:
            target = TASKS / name
            if target.exists():
                shutil_tree_cleanup(target)
        for p in created_sessions:
            try:
                p.unlink()
            except OSError:
                pass

    failed = [n for n, ok, _ in results if not ok]
    print(f"\n=== {len(results) - len(failed)}/{len(results)} passed ===")
    if failed:
        print("FAILED:", ", ".join(failed))
    return 1 if failed else 0


def shutil_tree_cleanup(path: Path) -> None:
    import shutil
    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
