"""`harness hook`: hook stdin JSON -> <CLAUDE_PLUGIN_DATA>/sessions/<session_id>.json.

Also denies Claude Code's native subagent dispatch tool (`Agent`, alias
`Task`) at PreToolUse, forcing subagent launches through
`harness_start_agent`/`harness_start_prompt` instead (see #52).

#62: tracks every run a `harness_start_agent`/`harness_start_prompt`/
`harness_send_message` call starts (PostToolUse) as an empty marker file
under `tracked-runs/<session_id>/<run_id>`, then at `Stop` blocks turn-end
(exit 2, stderr naming the pending run_ids) unless every run tracked for
this session has reached a terminal state (COMPLETED/FAILED/CANCELLED).
There is no opt-out. A companion PreToolUse guard denies
`harness_cleanup_run` on a still-non-terminal tracked run, closing the one
path (see plan "Premises verified" #5) that could otherwise delete a live
run's record and let a stale marker read as "gone" at Stop.

#64: the Stop branch no longer takes a single snapshot. It polls the real
`record.json` for every tracked run, every `_POLL_INTERVAL_SECONDS`, until
either all are terminal (exit 0) or `stop_wait_timeout()` (env
`HARNESS_STOP_WAIT_TIMEOUT_SECONDS`, default/ceiling 7200s) elapses (exit 2,
same as before). A timeout never cancels a run. A run already seen
non-terminal stays "pending" (sticky) across a transient read failure,
rather than a corrupt/racing read being read as "gone" mid-wait; a run that
cannot be read on its very first check is still excluded, matching #62.

Never blocks the parent's tool call outside those two intentional guards:
any other failure exits 0 having written nothing. The deny/block branches
run outside the fail-open try/except that guards the generic per-event
session-context write -- they must still fire even when that write itself
fails.

#68: `_pending_runs` used to only ever read `record.json` passively -- nothing
in the Stop loop ever drove a run's own reconciliation, so a CLEAN `claude -p`
child that had already written its terminal `result` event but kept its OS
process alive (lib_python_harness v0.0.10's post-completion grace-kill, gated
by `Harness.wait`'s `_FINALIZE_GRACE_S`) stayed RUNNING in `record.json`
forever, and Stop blocked for the full `stop_wait_timeout()` instead of the
grace period. `_drive_run(run_id)` now calls `harness().wait(run_id, 0)`
before every `record.json` read in `_pending_runs`, so the loop itself
advances the run instead of only observing it. It lazily imports `harness`
from `harness_plugin.runs` (only Stop pays for that import) and fails open
(`try/except Exception: pass`) -- the passive `record.json` read right after
it is the existing, unchanged fallback, so a `_drive_run` failure (e.g. a
record with no pid, as #62/#64's seeded test records have) degrades to
exactly the old behaviour rather than blocking Stop."""
from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from harness_plugin.host_context import (
    PROJECT_ENV,
    _safe_name,
    artifacts_root,
    sessions_dir,
    stop_wait_timeout,
    write_session_context,
)

_KEYS = ("session_id", "cwd", "permission_mode", "effort", "model", "transcript_path")

# The CLI's current tool name for native subagent dispatch is "Agent"; "Task"
# is its legacy alias (see plan #52 "Premises verified"). Both are denied.
_NATIVE_SUBAGENT_TOOLS = frozenset({"Agent", "Task"})

_DENY_REASON = (
    "Native subagent dispatch is disabled by agent-harness. Use harness_start_agent "
    "(a named agent) or harness_start_prompt (an ad-hoc prompt) instead."
)

# Tool-name suffixes (see _tool_suffix) that start a run and so must be
# tracked. Poll/wait/stop/list/cleanup/send_message-as-resume share the same
# PreToolUse/PostToolUse matcher and must NOT be tracked.
_START_TOOLS = frozenset({"harness_start_agent", "harness_start_prompt", "harness_send_message"})

_CLEANUP_TOOL = "harness_cleanup_run"

_TERMINAL = frozenset({"COMPLETED", "FAILED", "CANCELLED"})

# #64: how often the Stop branch re-reads every tracked run's record.json
# while waiting for it to go terminal. `harness wait --interval` is a CLI
# argument of a different process (nothing here to reuse); this one is fixed,
# not configurable -- only the overall deadline (stop_wait_timeout()) is.
_POLL_INTERVAL_SECONDS = 0.5

_STOP_WAIT_MESSAGE_FILENAME = "stop_wait_message.md"


def _deny(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )


def _tool_suffix(tool_name: object) -> str | None:
    """The bare tool name off a possibly-qualified MCP tool name (e.g.
    `mcp__harness__harness_start_agent` or
    `mcp__plugin_agent-harness_harness__harness_start_prompt` both yield
    `harness_start_agent`/`harness_start_prompt`)."""
    if not isinstance(tool_name, str) or not tool_name:
        return None
    return tool_name.rsplit("__", 1)[-1]


def _search_for_run_id(value: object) -> str | None:
    """Plan Approach: a dict with a str `run_id` returns that value (never
    `resumed_from`); a dict whose `text` is a str, or a bare str, is
    JSON-parsed and searched again; a list is searched item by item."""
    if isinstance(value, dict):
        run_id = value.get("run_id")
        if isinstance(run_id, str) and run_id:
            return run_id
        text = value.get("text")
        if isinstance(text, str):
            return _search_for_run_id(_try_json_loads(text))
        return None
    if isinstance(value, list):
        for item in value:
            run_id = _search_for_run_id(item)
            if run_id:
                return run_id
        return None
    if isinstance(value, str):
        return _search_for_run_id(_try_json_loads(value))
    return None


def _try_json_loads(text: str) -> object:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_run_id(tool_response: object) -> str | None:
    return _search_for_run_id(tool_response)


def _tracked_runs_dir() -> Path:
    return sessions_dir().parent / "tracked-runs"


def _track_started_run(session_id: object, run_id: object) -> None:
    session_name = _safe_name(session_id)
    run_name = _safe_name(run_id)
    if session_name is None or run_name is None:
        return
    directory = _tracked_runs_dir() / session_name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / run_name).touch()


def _tracked_run_ids(session_id: object) -> list[str]:
    session_name = _safe_name(session_id)
    if session_name is None:
        return []
    directory = _tracked_runs_dir() / session_name
    if not directory.is_dir():
        return []
    try:
        return [p.name for p in directory.iterdir()]
    except OSError:
        return []


def _run_state(run_id: str) -> str | None:
    """`<artifacts_root()>/<run_id>/record.json`'s `state.__runstate__`, or
    `None` on ANY failure (missing file, unreadable, not JSON, not the
    expected shape) -- a fail-open read, scoped to this one record so a
    malformed record for one tracked run can never swallow a different,
    genuinely-non-terminal one (test-critic round 2 note #1)."""
    try:
        raw = (artifacts_root() / run_id / "record.json").read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    state = data.get("state")
    if isinstance(state, dict):
        runstate = state.get("__runstate__")
        if isinstance(runstate, str):
            return runstate
    return None


def _drive_run(run_id: str) -> None:
    """#68: advance `run_id`'s own reconciliation (including
    lib_python_harness v0.0.10's post-completion grace-kill) before the
    passive `record.json` read that follows. `harness` is imported lazily
    from `harness_plugin.runs` so only the Stop event pays for it -- every
    other hook event never imports the lib at all. Fails open
    (`try/except Exception: pass`): a record with no pid (e.g. #62/#64's
    seeded test records) or any other error here must fall through to the
    unchanged passive read, never block Stop."""
    try:
        from harness_plugin.runs import harness

        harness().wait(run_id, 0)
    except Exception:
        pass


def _pending_runs(
    session_id: object, sticky: Mapping[str, str] | None = None
) -> list[tuple[str, str]]:
    """[(run_id, state)] for every run tracked under `session_id` that is not
    terminal, re-read fresh on every call (nothing here is cached across
    Stop invocations, so a run that later goes terminal is re-checked and
    correctly stops blocking).

    `sticky` (#64) is the caller's own previous return value, reshaped into a
    `{run_id: state}` mapping (see `main()`'s poll loop): when a read fails
    (missing/corrupt/racing record), a run already known non-terminal from an
    earlier pass keeps that last-known state instead of the read failure
    being treated as "gone" -- a single glitch mid-wait must not end the
    block early. A run whose very first read fails still has nothing in
    `sticky` yet, so it is excluded exactly as before (test-critic round 2
    note #1 / #62's fail-open behavior).

    #68: `_drive_run(run_id)` runs before each `record.json` read below, so
    this loop actually advances a run's own state (including the grace-kill
    for a lingering-but-result-written child) instead of only observing
    whatever an unrelated process last wrote."""
    sticky = sticky or {}
    pending: list[tuple[str, str]] = []
    for run_id in _tracked_run_ids(session_id):
        _drive_run(run_id)
        state = _run_state(run_id)
        if state is None:
            state = sticky.get(run_id)
            if state is None:
                continue
        if state not in _TERMINAL:
            pending.append((run_id, state))
    return pending


def _plugin_root() -> Path:
    """The plugin install root -- `hooks/` (and `bin/`) sit directly under
    it. Frozen (PyInstaller sets `sys.frozen`): `sys.executable` is
    `<root>/bin/harness[.exe]`, so `parents[1]` is `<root>`. Source checkout:
    this file is `<root>/src/harness_plugin/hooks/write_context.py`, so
    `parents[3]` is `<root>`."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parents[1]
    return Path(__file__).resolve().parents[3]


def _stop_block_message(pending: list[tuple[str, str]]) -> str:
    """Renders `hooks/stop_wait_message.md`'s single `{run_ids}` placeholder
    via `str.replace`. If the file cannot be read for any reason (missing,
    unreadable, unexpected plugin layout), stderr still gets the bare
    run_ids list -- the hook must never fail open just because its own
    message file is gone (#64 plan Approach)."""
    names = ", ".join(f"{run_id} ({state})" for run_id, state in pending)
    try:
        template = (_plugin_root() / "hooks" / _STOP_WAIT_MESSAGE_FILENAME).read_text(encoding="utf-8")
    except Exception:
        return names
    return template.replace("{run_ids}", names)


def _cleanup_deny_reason(data: dict) -> str | None:
    """Plan Approach: deny `harness_cleanup_run` when all three hold -- the
    tool suffix is `harness_cleanup_run`, `tool_input.run_id` is tracked for
    this session, and its record is readable and non-terminal. A read error
    (state None) means no deny."""
    tool_input = data.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    run_id = tool_input.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        return None
    if run_id not in _tracked_run_ids(data.get("session_id")):
        return None
    state = _run_state(run_id)
    if state is None or state in _TERMINAL:
        return None
    return f"run {run_id} is still {state}; wait for it or harness_stop_run it before cleaning up"


def main() -> int:
    try:
        raw = sys.stdin.buffer.read().decode("utf-8-sig")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return 0
    except Exception:
        return 0

    event = data.get("hook_event_name")
    tool_name = data.get("tool_name")

    if event == "PreToolUse" and tool_name in _NATIVE_SUBAGENT_TOOLS:
        _deny(_DENY_REASON)
        return 0

    if event == "PreToolUse" and _tool_suffix(tool_name) == _CLEANUP_TOOL:
        reason = _cleanup_deny_reason(data)
        if reason is not None:
            _deny(reason)
            return 0

    if event == "PostToolUse" and _tool_suffix(tool_name) in _START_TOOLS:
        try:
            session_id = data.get("session_id")
            run_id = _extract_run_id(data.get("tool_response"))
            if run_id:
                _track_started_run(session_id, run_id)
        except Exception:
            pass

    try:
        record = {k: data[k] for k in _KEYS if isinstance(data.get(k), str) and data[k]}
        if "session_id" in record:
            project = os.environ.get(PROJECT_ENV)
            if project:
                record["project_dir"] = project
            write_session_context(record)
    except Exception:
        pass

    if event == "Stop":
        session_id = data.get("session_id")
        deadline = time.monotonic() + stop_wait_timeout()
        pending: list[tuple[str, str]] = []
        while True:
            pending = _pending_runs(session_id, sticky=dict(pending))
            if not pending:
                return 0
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print(_stop_block_message(pending), file=sys.stderr)
                return 2
            time.sleep(min(_POLL_INTERVAL_SECONDS, remaining))

    return 0
