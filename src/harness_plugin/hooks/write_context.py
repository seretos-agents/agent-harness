"""`harness hook`: hook stdin JSON -> <CLAUDE_PLUGIN_DATA>/sessions/<session_id>.json.

Also denies Claude Code's native subagent dispatch tool (`Agent`, alias
`Task`) at PreToolUse, forcing subagent launches through
`harness_start_agent`/`harness_start_prompt` instead (see #52).

Never blocks the parent's tool call: any failure exits 0 having written nothing.
The deny branch runs outside that fail-open try/except -- it must still print
its denial even when the session-context write itself fails."""
from __future__ import annotations

import json
import os
import sys

from harness_plugin.host_context import PROJECT_ENV, write_session_context

_KEYS = ("session_id", "cwd", "permission_mode", "effort", "model", "transcript_path")

# The CLI's current tool name for native subagent dispatch is "Agent"; "Task"
# is its legacy alias (see plan #52 "Premises verified"). Both are denied.
_NATIVE_SUBAGENT_TOOLS = frozenset({"Agent", "Task"})

_DENY_REASON = (
    "Native subagent dispatch is disabled by agent-harness. Use harness_start_agent "
    "(a named agent) or harness_start_prompt (an ad-hoc prompt) instead."
)


def _deny_native_subagent() -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": _DENY_REASON,
                }
            }
        )
    )


def main() -> int:
    try:
        raw = sys.stdin.buffer.read().decode("utf-8-sig")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return 0
    except Exception:
        return 0

    if data.get("hook_event_name") == "PreToolUse" and data.get("tool_name") in _NATIVE_SUBAGENT_TOOLS:
        _deny_native_subagent()
        return 0

    try:
        record = {k: data[k] for k in _KEYS if isinstance(data.get(k), str) and data[k]}
        if "session_id" not in record:
            return 0
        project = os.environ.get(PROJECT_ENV)
        if project:
            record["project_dir"] = project
        write_session_context(record)
    except Exception:
        return 0
    return 0
