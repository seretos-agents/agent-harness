"""Minimal stand-in for the `claude` CLI, driven through HARNESS_CLAUDE_ARGV.

Answers `--version`; otherwise reads the prompt from stdin and emits the
stream-json events `lib_python_harness` parses (a terminal `result` event
carrying `result` and `session_id`). A prompt containing `SLEEP:<seconds>`
keeps the process alive first, so one fixture yields both a fast run and a
still-RUNNING run. With HARNESS_FAKE_ARGV_LOG set, each real invocation appends
one JSON line {"argv": [...], "cwd": ..., "launched_agent": ...} so tests can assert
which flags and working directory the CLI actually received, and (#51) whatever value
HARNESS_LAUNCHED_AGENT had in the child's own environment (None if unset). A prompt containing `NO_RESULT`
exits after the init event without any `result` event (a FAILED run).
A prompt containing `TICK:<count>:<interval>` emits `count` assistant events
`interval` seconds apart (flushed) before the terminal `result` event, so a
still-RUNNING run makes visible progress.
A prompt containing `TOOL:<name>` emits one assistant `tool_use` event for that tool right
after the init event (before any SLEEP), so a still-RUNNING run's last activity is that tool.
A prompt containing `ECHO:<word>` is answered with `<word>` instead of `OK`. The session
id is taken from `--session-id <id>` or, for a resumed run, `--resume <id>`.
A prompt containing `NO_INIT_NAMES` suppresses `INIT_ANNOUNCEMENTS` from the init event
(a bare `{"type": "system", "subtype": "init", ...}`), so a test can exercise
`harness_inspect_run`'s argv-derived `requested.*` fallback independently of what the
init event announces.
A prompt containing `ALT_INIT_SHAPE` emits the init event's mcp-server list under the
`mcpServers` alias key (instead of the primary `mcp_servers` spelling) and its tools list
as dict-shaped items (`{"name": ...}`) instead of plain strings, so a test can exercise
`harness_inspect_run`'s `_INIT_NAME_FIELDS` alias lookup and `_names()`'s dict branch,
neither of which `INIT_ANNOUNCEMENTS`'s plain-string/primary-key shape reaches.
A prompt containing `LINGER:<seconds>` keeps the process alive *after* it has written its
terminal `result` event -- the opposite of `SLEEP`, which stalls before any event at all.
This is #68's reproduction of a CLEAN `claude -p` child that has already told the harness
it is done but does not actually exit its OS process for a while: a test can start such a
run and exercise the grace-kill (`lib_python_harness.Harness.wait`'s `_FINALIZE_GRACE_S`)
that is supposed to reap it.
"""
import json
import os
import re
import sys
import time

# Fixed name lists the init event announces, keyed to match `_INIT_NAME_FIELDS`'s
# primary spelling (`mcp_servers`, `tools`, `skills`, `agents`) so a test can assert
# `harness_inspect_run`'s `announced.*` against this constant instead of a literal.
INIT_ANNOUNCEMENTS: dict[str, list[str]] = {
    "mcp_servers": ["alpha-server", "beta-server"],
    "tools": ["Read", "Bash", "Write"],
    "skills": ["skill-one", "skill-two", "skill-three"],
    "agents": ["agent-x", "agent-y"],
}

# Marker checked against the prompt (stdin) to suppress INIT_ANNOUNCEMENTS above.
NO_INIT_NAMES = "NO_INIT_NAMES"

# Marker checked against the prompt (stdin): switches the init event to the alias-key /
# dict-item shape below instead of INIT_ANNOUNCEMENTS's primary-key / plain-string shape.
ALT_INIT_SHAPE = "ALT_INIT_SHAPE"
ALT_INIT_MCP_SERVERS: list[str] = ["alias-server"]
ALT_INIT_TOOLS: list[str] = ["AliasTool"]


def main() -> int:
    argv = sys.argv[1:]
    if "--version" in argv:
        print("2.0.0 (Claude Code)")
        return 0

    log = os.environ.get("HARNESS_FAKE_ARGV_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "argv": argv,
                        "cwd": os.getcwd(),
                        "launched_agent": os.environ.get("HARNESS_LAUNCHED_AGENT"),
                    }
                )
                + "\n"
            )

    session_id = "fake-session"
    for flag in ("--session-id", "--resume"):
        if flag in argv:
            session_id = argv[argv.index(flag) + 1]

    prompt = sys.stdin.read()
    match = re.search(r"SLEEP:(\d+(?:\.\d+)?)", prompt)
    init_event = {"type": "system", "subtype": "init", "session_id": session_id}
    if ALT_INIT_SHAPE in prompt:
        init_event["mcpServers"] = ALT_INIT_MCP_SERVERS
        init_event["tools"] = [{"name": name} for name in ALT_INIT_TOOLS]
    elif NO_INIT_NAMES not in prompt:
        init_event.update(INIT_ANNOUNCEMENTS)
    print(json.dumps(init_event), flush=True)
    tool = re.search(r"TOOL:(\w+)", prompt)
    if tool:
        print(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": "toolu_1", "name": tool.group(1), "input": {}}
                        ],
                    },
                    "session_id": session_id,
                }
            ),
            flush=True,
        )
    if match:
        deadline = time.monotonic() + float(match.group(1))
        while time.monotonic() < deadline:
            time.sleep(0.1)

    tick = re.search(r"TICK:(\d+):(\d+(?:\.\d+)?)", prompt)
    if tick:
        for i in range(int(tick.group(1))):
            print(
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": f"tick {i}"}],
                        },
                        "session_id": session_id,
                    }
                ),
                flush=True,
            )
            time.sleep(float(tick.group(2)))

    echo = re.search(r"ECHO:(\w+)", prompt)
    answer = echo.group(1) if echo else "OK"

    if "NO_RESULT" in prompt:
        # Ends without a terminal `result` event: the run finishes FAILED.
        return 0

    print(
        json.dumps(
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                "session_id": session_id,
            }
        ),
        flush=True,
    )
    print(
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": answer,
                "session_id": session_id,
                "total_cost_usd": 0.0,
                "usage": {},
            }
        ),
        flush=True,
    )

    linger = re.search(r"LINGER:(\d+(?:\.\d+)?)", prompt)
    if linger:
        # The terminal `result` event above is already on disk (flushed); the
        # OS process itself just keeps running past it, reproducing #68.
        deadline = time.monotonic() + float(linger.group(1))
        while time.monotonic() < deadline:
            time.sleep(0.1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
