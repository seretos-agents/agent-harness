"""The `hook` subcommand: hook stdin -> <CLAUDE_PLUGIN_DATA>/sessions/<session_id>.json.

Run as a real subprocess (`python -m harness_plugin hook`), the way the plugin's
hooks.json runs the frozen binary."""
import json
import os
import subprocess
import sys

import pytest
from lib_python_harness import FileRunStore, RunState

PRE_TOOL_USE = {
    "session_id": "sess-abc",
    "transcript_path": "/tmp/transcripts/sess-abc.jsonl",
    "cwd": "/work/project/subdir",  # differs from CLAUDE_PROJECT_DIR on purpose
    "permission_mode": "acceptEdits",
    "effort": "high",
    "hook_event_name": "PreToolUse",
    "tool_name": "mcp__harness__harness_start_agent",
    "tool_input": {"agent": "demo"},
}


def run_hook(stdin_text, plugin_data, project_dir="/work/project", extra_env=None):
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "CLAUDE_PLUGIN_DATA",
            "CLAUDE_PROJECT_DIR",
            "CLAUDE_CODE_SESSION_ID",
            "HARNESS_ARTIFACTS_DIR",
        )
    }
    env["CLAUDE_PLUGIN_DATA"] = str(plugin_data)
    env["CLAUDE_PROJECT_DIR"] = project_dir
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "harness_plugin", "hook"],
        input=stdin_text,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def test_hook_writes_session_context(tmp_path):
    proc = run_hook(json.dumps(PRE_TOOL_USE), tmp_path)
    assert proc.returncode == 0, proc.stderr
    written = tmp_path / "sessions" / "sess-abc.json"
    assert written.is_file(), f"no session file; stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert json.loads(written.read_text(encoding="utf-8")) == {
        "session_id": "sess-abc",
        "cwd": "/work/project/subdir",
        "project_dir": "/work/project",
        "permission_mode": "acceptEdits",
        "effort": "high",
        "transcript_path": "/tmp/transcripts/sess-abc.jsonl",
    }


def test_session_start_payload_without_permission_mode_is_written(tmp_path):
    payload = {
        "session_id": "sess-start",
        "cwd": "/work/project",
        "hook_event_name": "SessionStart",
        "source": "startup",
    }
    proc = run_hook(json.dumps(payload), tmp_path)
    assert proc.returncode == 0, proc.stderr
    data = json.loads((tmp_path / "sessions" / "sess-start.json").read_text(encoding="utf-8"))
    assert data["session_id"] == "sess-start"
    assert data["cwd"] == "/work/project"
    assert "permission_mode" not in data


@pytest.mark.parametrize(
    "stdin_text",
    ["{not json", "", json.dumps({"cwd": "/work/project", "permission_mode": "plan"})],
    ids=["malformed", "empty", "no-session-id"],
)
def test_malformed_stdin_exits_zero_without_writing(tmp_path, stdin_text):
    # Control: the hook exists and does write for a good payload.
    control = run_hook(json.dumps(PRE_TOOL_USE), tmp_path / "control")
    assert control.returncode == 0, control.stderr
    assert (tmp_path / "control" / "sessions" / "sess-abc.json").is_file()

    bad_data = tmp_path / "bad"
    proc = run_hook(stdin_text, bad_data)
    assert proc.returncode == 0, proc.stderr
    sessions = bad_data / "sessions"
    assert not sessions.exists() or list(sessions.iterdir()) == []


def test_unwritable_sessions_dir_exits_zero(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("occupied", encoding="utf-8")  # mkdir under a file must fail
    proc = run_hook(json.dumps(PRE_TOOL_USE), blocker)
    assert proc.returncode == 0, proc.stderr


# --- #52: native Agent/Task subagent dispatch is denied at PreToolUse -------

_NATIVE_SUBAGENT_TOOLS = ("Agent", "Task")


def _native_pre_tool_use(tool_name):
    payload = dict(PRE_TOOL_USE)
    payload["tool_name"] = tool_name
    return payload


@pytest.mark.parametrize("tool_name", _NATIVE_SUBAGENT_TOOLS)
def test_native_subagent_tool_is_denied(tmp_path, tool_name):
    """R1: a PreToolUse payload naming the native `Agent` tool (or its legacy
    `Task` alias) is denied via the CLI's real hookSpecificOutput deny shape.

    Expected RED reason: stdout is empty today (the hook has no deny branch),
    so json.loads("") raises JSONDecodeError."""
    proc = run_hook(json.dumps(_native_pre_tool_use(tool_name)), tmp_path)
    assert proc.returncode == 0, proc.stderr
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(f"stdout did not parse as JSON: {exc}; stdout={proc.stdout!r} stderr={proc.stderr!r}")
    hook_output = parsed["hookSpecificOutput"]
    assert hook_output["hookEventName"] == "PreToolUse"
    assert hook_output["permissionDecision"] == "deny"
    reason = hook_output["permissionDecisionReason"]
    assert "harness_start_agent" in reason
    assert "harness_start_prompt" in reason


@pytest.mark.parametrize("tool_name", _NATIVE_SUBAGENT_TOOLS)
def test_deny_survives_unwritable_sessions_dir(tmp_path, tool_name):
    """Additional edge-case coverage for R1: even when the session-context
    write itself cannot happen (sessions dir blocked by a file), the deny
    output is still produced -- the deny print must not depend on the write's
    own try/except succeeding.

    Expected RED reason: same as test_native_subagent_tool_is_denied -- no
    deny branch exists yet."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("occupied", encoding="utf-8")
    proc = run_hook(json.dumps(_native_pre_tool_use(tool_name)), blocker)
    assert proc.returncode == 0, proc.stderr
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(f"stdout did not parse as JSON: {exc}; stdout={proc.stdout!r} stderr={proc.stderr!r}")
    assert parsed["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_harness_tool_not_denied(tmp_path):
    """R2 (non-regression guard): the existing `mcp__harness__harness_start_agent`
    PreToolUse payload produces no permissionDecision on stdout, and the
    session file is still written.

    Expected RED reason: none -- this guards against an over-broad deny
    branch and may already pass against unfixed code."""
    proc = run_hook(json.dumps(PRE_TOOL_USE), tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "deny" not in proc.stdout, proc.stdout
    written = tmp_path / "sessions" / "sess-abc.json"
    assert written.is_file(), f"no session file; stdout={proc.stdout!r} stderr={proc.stderr!r}"


def test_harness_prompt_tool_not_denied(tmp_path):
    """R2: a plugin-qualified harness tool name (`harness_start_prompt`) is
    likewise not denied and still writes the session file.

    Expected RED reason: none -- may already pass."""
    payload = dict(PRE_TOOL_USE)
    payload["tool_name"] = "mcp__plugin_agent-harness_harness__harness_start_prompt"
    proc = run_hook(json.dumps(payload), tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "deny" not in proc.stdout, proc.stdout
    written = tmp_path / "sessions" / "sess-abc.json"
    assert written.is_file(), f"no session file; stdout={proc.stdout!r} stderr={proc.stderr!r}"


# --- #62: Stop blocks while a run started in this session is non-terminal --------
#
# Setup shared by R1/R2/R3 below (plan "Test / verification strategy"): `run_hook`'s
# `extra_env` carries HARNESS_ARTIFACTS_DIR so the hook subprocess and this test's own
# `FileRunStore` seed agree on where record.json lives; records are seeded through the
# real `FileRunStore`, never hand-written JSON, so `state`'s `{"__runstate__": ...}`
# on-disk shape (lib_python_harness store.py `_serialize`) is exactly what a real run
# would have written.

_START_TOOL_NAMES = [
    "mcp__harness__harness_start_agent",
    "mcp__plugin_agent-harness_harness__harness_start_prompt",
    "mcp__harness__harness_send_message",
]

_TOOL_RESPONSE_SHAPE_IDS = ["text-block", "text-block-list", "raw-dict"]

_TRACKED_STATES = [RunState.RUNNING, RunState.CREATED]


def _tool_response_shape(shape_id, run_id):
    if shape_id == "text-block":
        return {"type": "text", "text": json.dumps({"run_id": run_id})}
    if shape_id == "text-block-list":
        return [{"type": "text", "text": json.dumps({"run_id": run_id})}]
    if shape_id == "raw-dict":
        return {"run_id": run_id}
    raise ValueError(shape_id)


def _post_tool_use_payload(session_id, tool_name, tool_response, cwd="/work/project"):
    return {
        "session_id": session_id,
        "cwd": cwd,
        "hook_event_name": "PostToolUse",
        "tool_name": tool_name,
        "tool_input": {},
        "tool_response": tool_response,
    }


def _stop_payload(session_id, cwd="/work/project"):
    return {
        "session_id": session_id,
        "cwd": cwd,
        "hook_event_name": "Stop",
    }


def _pre_tool_use_cleanup_payload(session_id, run_id, cwd="/work/project"):
    return {
        "session_id": session_id,
        "cwd": cwd,
        "hook_event_name": "PreToolUse",
        "tool_name": "mcp__harness__harness_cleanup_run",
        "tool_input": {"run_id": run_id},
    }


def _seed_record(artifacts_dir, run_id, state):
    FileRunStore(str(artifacts_dir)).put(run_id, {"run_id": run_id, "state": state})


def _track(plugin_data, extra_env, session_id, tool_name, run_id):
    """Run a real PostToolUse hook call so `run_id` is genuinely tracked for
    `session_id` -- never a hand-written marker file, so this exercises the
    same PostToolUse code path a real tool call would."""
    proc = run_hook(
        json.dumps(_post_tool_use_payload(session_id, tool_name, {"run_id": run_id})),
        plugin_data,
        extra_env=extra_env,
    )
    assert proc.returncode == 0, proc.stderr
    return proc


# --- R1: Stop blocks while a tracked run is non-terminal -------------------------


@pytest.mark.parametrize("state", _TRACKED_STATES, ids=[s.name for s in _TRACKED_STATES])
@pytest.mark.parametrize("shape_id", _TOOL_RESPONSE_SHAPE_IDS)
@pytest.mark.parametrize("tool_name", _START_TOOL_NAMES)
def test_stop_blocks_on_tracked_running_run(tmp_path, tool_name, shape_id, state):
    """R1 driving test: a PostToolUse for a start tool tracks the run_id
    extracted from `tool_response` (whichever of the three shapes it arrives
    in), and a Stop for the same session then exits 2 while that run is
    RUNNING/CREATED, naming the run_id and the two ways out in stderr.

    Expected RED reason: there is no Stop branch today -- Stop always exits
    0 (the generic per-event session write is the only thing that runs), so
    `assert stop.returncode == 2` fails with 0 == 2."""
    plugin_data = tmp_path / "plugin-data"
    artifacts_dir = tmp_path / "artifacts"
    extra_env = {"HARNESS_ARTIFACTS_DIR": str(artifacts_dir)}
    session_id = "sess-r1"
    run_id = "run-r1"

    _seed_record(artifacts_dir, run_id, state)
    post = run_hook(
        json.dumps(_post_tool_use_payload(session_id, tool_name, _tool_response_shape(shape_id, run_id))),
        plugin_data,
        extra_env=extra_env,
    )
    assert post.returncode == 0, post.stderr

    stop = run_hook(json.dumps(_stop_payload(session_id)), plugin_data, extra_env=extra_env)
    assert stop.returncode == 2, (
        f"expected Stop to block on a {state.name} tracked run; "
        f"stdout={stop.stdout!r} stderr={stop.stderr!r}"
    )
    assert run_id in stop.stderr
    assert "harness_wait_run" in stop.stderr
    assert "harness_stop_run" in stop.stderr


def test_resumed_from_is_not_tracked(tmp_path):
    """R1 additional coverage: only the `run_id` key of `tool_response` is
    read, never `resumed_from` (plan Approach). A version that also tracked
    `resumed_from` would block here even though nothing was ever started in
    this session.

    Expected RED reason: none directly against today's fully-absent Stop
    branch (Stop already exits 0 unconditionally) -- this guards against a
    future _extract_run_id that is too eager, once Stop exists."""
    plugin_data = tmp_path / "plugin-data"
    artifacts_dir = tmp_path / "artifacts"
    extra_env = {"HARNESS_ARTIFACTS_DIR": str(artifacts_dir)}
    session_id = "sess-resumed"
    run_id = "run-resumed"
    _seed_record(artifacts_dir, run_id, RunState.RUNNING)

    post = run_hook(
        json.dumps(
            _post_tool_use_payload(
                session_id, "mcp__harness__harness_send_message", {"resumed_from": run_id}
            )
        ),
        plugin_data,
        extra_env=extra_env,
    )
    assert post.returncode == 0, post.stderr

    stop = run_hook(json.dumps(_stop_payload(session_id)), plugin_data, extra_env=extra_env)
    assert stop.returncode == 0, (
        f"resumed_from must not be tracked as a run_id; stdout={stop.stdout!r} stderr={stop.stderr!r}"
    )


def test_stop_names_only_pending_run_when_mixed(tmp_path):
    """R1 additional coverage: two tracked runs in one session, one already
    COMPLETED and one still RUNNING -- Stop blocks and names only the
    RUNNING one.

    Expected RED reason: same as the driving test -- Stop always exits 0
    today, so `assert stop.returncode == 2` fails."""
    plugin_data = tmp_path / "plugin-data"
    artifacts_dir = tmp_path / "artifacts"
    extra_env = {"HARNESS_ARTIFACTS_DIR": str(artifacts_dir)}
    session_id = "sess-mixed"
    done_id = "run-done"
    pending_id = "run-pending"
    _seed_record(artifacts_dir, done_id, RunState.COMPLETED)
    _seed_record(artifacts_dir, pending_id, RunState.RUNNING)

    _track(plugin_data, extra_env, session_id, "mcp__harness__harness_start_agent", done_id)
    _track(plugin_data, extra_env, session_id, "mcp__harness__harness_start_agent", pending_id)

    stop = run_hook(json.dumps(_stop_payload(session_id)), plugin_data, extra_env=extra_env)
    assert stop.returncode == 2, f"stdout={stop.stdout!r} stderr={stop.stderr!r}"
    assert pending_id in stop.stderr
    assert done_id not in stop.stderr


def test_stop_blocks_despite_other_malformed_record(tmp_path):
    """R1 additional coverage (plan-critic round 2 note): the per-record
    fail-open must not swallow a different, genuinely-RUNNING tracked run.
    One tracked run's record.json is not even a JSON object; a second
    tracked run in the same session is genuinely RUNNING. A version that
    wraps the *whole* collection loop in one broad try/except (instead of
    one per record) would return 0 here, hiding the RUNNING run.

    Expected RED reason: Stop always exits 0 today, so
    `assert stop.returncode == 2` fails."""
    plugin_data = tmp_path / "plugin-data"
    artifacts_dir = tmp_path / "artifacts"
    extra_env = {"HARNESS_ARTIFACTS_DIR": str(artifacts_dir)}
    session_id = "sess-malformed"
    bad_id = "run-malformed"
    good_id = "run-good"

    _seed_record(artifacts_dir, good_id, RunState.RUNNING)
    bad_record_dir = artifacts_dir / bad_id
    bad_record_dir.mkdir(parents=True)
    (bad_record_dir / "record.json").write_text("not json at all", encoding="utf-8")

    _track(plugin_data, extra_env, session_id, "mcp__harness__harness_start_agent", bad_id)
    _track(plugin_data, extra_env, session_id, "mcp__harness__harness_start_agent", good_id)

    stop = run_hook(json.dumps(_stop_payload(session_id)), plugin_data, extra_env=extra_env)
    assert stop.returncode == 2, (
        f"a malformed record for one tracked run must not hide a different "
        f"genuinely-RUNNING one; stdout={stop.stdout!r} stderr={stop.stderr!r}"
    )
    assert good_id in stop.stderr


def test_stop_blocks_despite_unwritable_sessions_subdir(tmp_path):
    """R1 additional coverage (pattern: test_deny_survives_unwritable_sessions_dir):
    the Stop block must not depend on the generic per-event session-context
    write succeeding. Here the `sessions/` subdirectory itself (not the
    whole CLAUDE_PLUGIN_DATA, which would also block tracked-runs/ and so
    prevent tracking in the first place) is occupied by a file, so the
    plain write_session_context() call fails and exits via that branch's
    own try/except -- the Stop-specific check-and-block logic is a separate
    code path and must still run and block.

    Expected RED reason: same as the driving test."""
    plugin_data = tmp_path / "plugin-data"
    plugin_data.mkdir()
    (plugin_data / "sessions").write_text("occupied", encoding="utf-8")
    artifacts_dir = tmp_path / "artifacts"
    extra_env = {"HARNESS_ARTIFACTS_DIR": str(artifacts_dir)}
    session_id = "sess-blocked-sessions"
    run_id = "run-blocked-sessions"
    _seed_record(artifacts_dir, run_id, RunState.RUNNING)

    _track(plugin_data, extra_env, session_id, "mcp__harness__harness_start_agent", run_id)

    stop = run_hook(json.dumps(_stop_payload(session_id)), plugin_data, extra_env=extra_env)
    assert stop.returncode == 2, f"stdout={stop.stdout!r} stderr={stop.stderr!r}"
    assert run_id in stop.stderr


# --- R2: Stop passes once every tracked run is terminal --------------------------


@pytest.mark.parametrize(
    "terminal_state",
    [RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED],
    ids=["COMPLETED", "FAILED", "CANCELLED"],
)
def test_stop_passes_after_runs_terminal(tmp_path, terminal_state):
    """R2 driving test: a tracked run blocks Stop while RUNNING, then Stop
    passes once the same run's record reaches a terminal state.

    Expected RED reason: the first `assert stop.returncode == 2` fails on
    current code (Stop always exits 0)."""
    plugin_data = tmp_path / "plugin-data"
    artifacts_dir = tmp_path / "artifacts"
    extra_env = {"HARNESS_ARTIFACTS_DIR": str(artifacts_dir)}
    session_id = "sess-r2"
    run_id = "run-r2"

    _seed_record(artifacts_dir, run_id, RunState.RUNNING)
    _track(plugin_data, extra_env, session_id, "mcp__harness__harness_start_agent", run_id)

    stop = run_hook(json.dumps(_stop_payload(session_id)), plugin_data, extra_env=extra_env)
    assert stop.returncode == 2, f"stdout={stop.stdout!r} stderr={stop.stderr!r}"

    _seed_record(artifacts_dir, run_id, terminal_state)
    stop2 = run_hook(json.dumps(_stop_payload(session_id)), plugin_data, extra_env=extra_env)
    assert stop2.returncode == 0, (
        f"Stop must pass once the tracked run is {terminal_state.name}; "
        f"stdout={stop2.stdout!r} stderr={stop2.stderr!r}"
    )


def test_stop_passes_with_no_tracked_runs(tmp_path):
    """R2 additional coverage: nothing was ever tracked for this session --
    Stop passes. May already pass today (Stop is currently unconditional)."""
    plugin_data = tmp_path / "plugin-data"
    stop = run_hook(json.dumps(_stop_payload("sess-empty")), plugin_data)
    assert stop.returncode == 0, stop.stderr


def test_stop_passes_with_missing_record(tmp_path):
    """R2 additional coverage: a tracked run whose record.json was never
    written (e.g. already cleaned up) fails open. May already pass today."""
    plugin_data = tmp_path / "plugin-data"
    artifacts_dir = tmp_path / "artifacts"
    extra_env = {"HARNESS_ARTIFACTS_DIR": str(artifacts_dir)}
    session_id = "sess-missing-record"
    run_id = "run-missing-record"
    _track(plugin_data, extra_env, session_id, "mcp__harness__harness_start_agent", run_id)

    stop = run_hook(json.dumps(_stop_payload(session_id)), plugin_data, extra_env=extra_env)
    assert stop.returncode == 0, f"stdout={stop.stdout!r} stderr={stop.stderr!r}"


def test_stop_ignores_other_sessions_runs(tmp_path):
    """R2 additional coverage: a run tracked under session A does not block
    Stop for session B. May already pass today."""
    plugin_data = tmp_path / "plugin-data"
    artifacts_dir = tmp_path / "artifacts"
    extra_env = {"HARNESS_ARTIFACTS_DIR": str(artifacts_dir)}
    run_id = "run-session-a"
    _seed_record(artifacts_dir, run_id, RunState.RUNNING)
    _track(plugin_data, extra_env, "sess-a", "mcp__harness__harness_start_agent", run_id)

    stop = run_hook(json.dumps(_stop_payload("sess-b")), plugin_data, extra_env=extra_env)
    assert stop.returncode == 0, f"stdout={stop.stdout!r} stderr={stop.stderr!r}"


def test_poll_run_is_not_tracked(tmp_path):
    """R2 additional coverage: PostToolUse for harness_poll_run (not a start
    tool) tracks nothing, even though its tool_response might carry a
    run_id. May already pass today."""
    plugin_data = tmp_path / "plugin-data"
    artifacts_dir = tmp_path / "artifacts"
    extra_env = {"HARNESS_ARTIFACTS_DIR": str(artifacts_dir)}
    session_id = "sess-poll"
    run_id = "run-polled"
    _seed_record(artifacts_dir, run_id, RunState.RUNNING)
    _track(plugin_data, extra_env, session_id, "mcp__harness__harness_poll_run", run_id)

    stop = run_hook(json.dumps(_stop_payload(session_id)), plugin_data, extra_env=extra_env)
    assert stop.returncode == 0, f"stdout={stop.stdout!r} stderr={stop.stderr!r}"


def test_malformed_tool_response_writes_no_marker(tmp_path):
    """R2 additional coverage: a start-tool PostToolUse whose tool_response
    is not any recognized shape (e.g. a bare int) writes no marker. May
    already pass today."""
    plugin_data = tmp_path / "plugin-data"
    artifacts_dir = tmp_path / "artifacts"
    extra_env = {"HARNESS_ARTIFACTS_DIR": str(artifacts_dir)}
    session_id = "sess-malformed-response"
    post = run_hook(
        json.dumps(_post_tool_use_payload(session_id, "mcp__harness__harness_start_agent", 12345)),
        plugin_data,
        extra_env=extra_env,
    )
    assert post.returncode == 0, post.stderr

    stop = run_hook(json.dumps(_stop_payload(session_id)), plugin_data, extra_env=extra_env)
    assert stop.returncode == 0, f"stdout={stop.stdout!r} stderr={stop.stderr!r}"


# --- R3: harness_cleanup_run cannot be used to escape the Stop block -------------


@pytest.mark.parametrize("state", _TRACKED_STATES, ids=[s.name for s in _TRACKED_STATES])
def test_cleanup_denied_for_live_tracked_run(tmp_path, state):
    """R3 driving test: harness_cleanup_run is denied at PreToolUse for a run
    tracked in this session whose record is still RUNNING/CREATED --
    closing the cleanup-as-escape-hatch the plan-critic found (cleanup
    itself never checks state; plan "Premises verified" #5).

    Expected RED reason: there is no deny branch for harness_cleanup_run
    today, so stdout is empty and `json.loads("")` raises JSONDecodeError."""
    plugin_data = tmp_path / "plugin-data"
    artifacts_dir = tmp_path / "artifacts"
    extra_env = {"HARNESS_ARTIFACTS_DIR": str(artifacts_dir)}
    session_id = "sess-r3"
    run_id = "run-r3"
    _seed_record(artifacts_dir, run_id, state)
    _track(plugin_data, extra_env, session_id, "mcp__harness__harness_start_agent", run_id)

    proc = run_hook(
        json.dumps(_pre_tool_use_cleanup_payload(session_id, run_id)),
        plugin_data,
        extra_env=extra_env,
    )
    assert proc.returncode == 0, proc.stderr
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(f"stdout did not parse as JSON: {exc}; stdout={proc.stdout!r} stderr={proc.stderr!r}")
    hook_output = parsed["hookSpecificOutput"]
    assert hook_output["hookEventName"] == "PreToolUse"
    assert hook_output["permissionDecision"] == "deny"
    reason = hook_output["permissionDecisionReason"]
    assert run_id in reason
    assert "harness_stop_run" in reason


def test_cleanup_not_denied_for_terminal_record(tmp_path):
    """R3 additional coverage: a tracked run whose record has already
    reached a terminal state is not denied. May already pass today."""
    plugin_data = tmp_path / "plugin-data"
    artifacts_dir = tmp_path / "artifacts"
    extra_env = {"HARNESS_ARTIFACTS_DIR": str(artifacts_dir)}
    session_id = "sess-r3-terminal"
    run_id = "run-r3-terminal"
    _seed_record(artifacts_dir, run_id, RunState.COMPLETED)
    _track(plugin_data, extra_env, session_id, "mcp__harness__harness_start_agent", run_id)

    proc = run_hook(
        json.dumps(_pre_tool_use_cleanup_payload(session_id, run_id)),
        plugin_data,
        extra_env=extra_env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "deny" not in proc.stdout, proc.stdout


def test_cleanup_not_denied_for_untracked_run(tmp_path):
    """R3 additional coverage: harness_cleanup_run for a run_id never
    tracked in this session (even though its record exists and is RUNNING)
    is not denied. May already pass today."""
    plugin_data = tmp_path / "plugin-data"
    artifacts_dir = tmp_path / "artifacts"
    extra_env = {"HARNESS_ARTIFACTS_DIR": str(artifacts_dir)}
    run_id = "run-r3-untracked"
    _seed_record(artifacts_dir, run_id, RunState.RUNNING)

    proc = run_hook(
        json.dumps(_pre_tool_use_cleanup_payload("sess-r3-untracked", run_id)),
        plugin_data,
        extra_env=extra_env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "deny" not in proc.stdout, proc.stdout


def test_cleanup_not_denied_for_other_sessions_run(tmp_path):
    """R3 additional coverage: a run tracked under session A is not denied
    when harness_cleanup_run is called from session B. May already pass
    today."""
    plugin_data = tmp_path / "plugin-data"
    artifacts_dir = tmp_path / "artifacts"
    extra_env = {"HARNESS_ARTIFACTS_DIR": str(artifacts_dir)}
    run_id = "run-r3-other-session"
    _seed_record(artifacts_dir, run_id, RunState.RUNNING)
    _track(plugin_data, extra_env, "sess-a-r3", "mcp__harness__harness_start_agent", run_id)

    proc = run_hook(
        json.dumps(_pre_tool_use_cleanup_payload("sess-b-r3", run_id)),
        plugin_data,
        extra_env=extra_env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "deny" not in proc.stdout, proc.stdout
