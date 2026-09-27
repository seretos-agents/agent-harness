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

Never blocks the parent's tool call outside those two intentional guards:
any other failure exits 0 having written nothing. The deny/block branches
run outside the fail-open try/except that guards the generic per-event
session-context write -- they must still fire even when that write itself
fails."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from harness_plugin.host_context import PROJECT_ENV, _safe_name, artifacts_root, sessions_dir, write_session_context

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


def _pending_runs(session_id: object) -> list[tuple[str, str]]:
    """[(run_id, state)] for every run tracked under `session_id` whose
    state was read successfully and is not terminal. A run whose state
    could not be read at all (missing/corrupt record) fails open -- it is
    silently excluded, never treated as pending -- per test-critic round 2
    note #1, re-read fresh on every call: nothing here is cached across
    Stop invocations, so a run that later goes terminal is re-checked and
    correctly stops blocking on the very next Stop."""
    pending: list[tuple[str, str]] = []
    for run_id in _tracked_run_ids(session_id):
        state = _run_state(run_id)
        if state is not None and state not in _TERMINAL:
            pending.append((run_id, state))
    return pending


def _stop_block_message(pending: list[tuple[str, str]]) -> str:
    names = ", ".join(f"{run_id} ({state})" for run_id, state in pending)
    return (
        f"agent-harness: run(s) {names} started in this session are not finished; "
        "call harness_wait_run / harness_poll_run until they are terminal, or "
        "harness_stop_run to cancel, before ending the turn."
    )


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
        pending = _pending_runs(data.get("session_id"))
        if pending:
            print(_stop_block_message(pending), file=sys.stderr)
            return 2

    return 0
