"""The `hook` subcommand: hook stdin -> <CLAUDE_PLUGIN_DATA>/sessions/<session_id>.json.

Run as a real subprocess (`python -m harness_plugin hook`), the way the plugin's
hooks.json runs the frozen binary."""
import contextlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
from lib_python_harness import FileRunStore, RunState

REPO = Path(__file__).resolve().parents[1]

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


def _hook_env(plugin_data, project_dir="/work/project", extra_env=None):
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "CLAUDE_PLUGIN_DATA",
            "CLAUDE_PROJECT_DIR",
            "CLAUDE_CODE_SESSION_ID",
            "HARNESS_ARTIFACTS_DIR",
            "HARNESS_STOP_WAIT_TIMEOUT_SECONDS",
        )
    }
    env["CLAUDE_PLUGIN_DATA"] = str(plugin_data)
    env["CLAUDE_PROJECT_DIR"] = project_dir
    # #64: a real internal wait defaults to 7200s -- keep every test above
    # (and any test below that does not care about the wait itself) a single
    # snapshot, matching #62's behavior, unless a test opts in via extra_env.
    env["HARNESS_STOP_WAIT_TIMEOUT_SECONDS"] = "0"
    if extra_env:
        env.update(extra_env)
    return env


def run_hook(stdin_text, plugin_data, project_dir="/work/project", extra_env=None):
    return subprocess.run(
        [sys.executable, "-m", "harness_plugin", "hook"],
        input=stdin_text,
        capture_output=True,
        text=True,
        env=_hook_env(plugin_data, project_dir, extra_env),
        timeout=60,
    )


def _spawn_hook(stdin_text, plugin_data, project_dir="/work/project", extra_env=None):
    """Like `run_hook`, but returns a live `Popen` so a test can interact
    with the subprocess while it is still running -- the #64 poll loop lives
    inside a single Stop invocation, not across several hook calls."""
    return subprocess.Popen(
        [sys.executable, "-m", "harness_plugin", "hook"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_hook_env(plugin_data, project_dir, extra_env),
    )


@contextlib.contextmanager
def _running_hook(stdin_text, plugin_data, extra_env=None):
    """`_spawn_hook`, writing and closing stdin immediately (the real hook
    reads all of stdin before doing anything), with guaranteed cleanup even
    when an assertion fails while the subprocess is still alive."""
    proc = _spawn_hook(stdin_text, plugin_data, extra_env=extra_env)
    proc.stdin.write(stdin_text)
    proc.stdin.close()
    # CPython's POSIX `Popen._communicate` unconditionally calls
    # `self.stdin.flush()` on its first invocation whenever `self.stdin` is
    # still a truthy attribute -- even though we already closed it above --
    # and only suppresses `BrokenPipeError`, not the `ValueError: I/O
    # operation on closed file.` that flushing an already-closed stream
    # raises. That made every test below's own `proc.communicate(...)` call
    # blow up on Linux (never on Windows, whose `_communicate` instead
    # spawns reader/writer threads and only closes -- never flushes -- an
    # already-closed stdin, which is a harmless no-op). Clearing the
    # attribute once we're done with it tells `communicate()` there is no
    # stdin pipe left to manage, matching the state a caller who never
    # touched stdin directly would be in.
    proc.stdin = None
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
        with contextlib.suppress(Exception):
            proc.communicate(timeout=5)


def _wait_for_file(path, timeout=15):
    deadline = time.monotonic() + timeout
    while not path.is_file():
        assert time.monotonic() < deadline, f"{path} never appeared within {timeout}s"
        time.sleep(0.05)


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


# --- #64: Stop hook waits internally for non-terminal runs ------------------------
#
# `_hook_env` puts HARNESS_STOP_WAIT_TIMEOUT_SECONDS="0" in every test above (and
# in any test below that does not care about the wait itself), so #62's tests keep
# taking a single snapshot; these tests opt into a real wait via extra_env.
#
# `_running_hook`/`_wait_for_file` let a test interact with a still-running Stop
# subprocess: `write_context.main()`'s generic per-event session-context write
# (before the Stop branch) happens exactly once, at the very start of *this*
# process's `main()` call -- so deleting sessions/<sid>.json right after `_track()`
# and waiting for it to reappear is the signal that *this* Stop invocation has
# reached its own check, not an artifact of `_track()`'s own PostToolUse call.


@pytest.mark.parametrize(
    "terminal_state",
    [RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED],
    ids=["COMPLETED", "FAILED", "CANCELLED"],
)
def test_stop_waits_until_run_terminal(tmp_path, terminal_state):
    """R1 driving test: a real Stop subprocess with a RUNNING tracked run
    exits 0 once the real record turns terminal, with no tool call in
    between -- the poll loop must run inside this one invocation, and must
    notice the flip within a few poll intervals rather than sleeping to the
    20s deadline (tautology::F1: a snapshot-then-sleep-then-recheck hook
    that never re-reads would also exit 0 eventually here, just ~19s late).

    Expected RED reason: the current code exits 2 on its first snapshot
    (there is no loop), so `assert proc.returncode == 0` fails with 2 == 0."""
    plugin_data = tmp_path / "plugin-data"
    artifacts_dir = tmp_path / "artifacts"
    extra_env = {"HARNESS_ARTIFACTS_DIR": str(artifacts_dir), "HARNESS_STOP_WAIT_TIMEOUT_SECONDS": "20"}
    session_id = "sess-wait"
    run_id = "run-wait"

    _seed_record(artifacts_dir, run_id, RunState.RUNNING)
    _track(plugin_data, extra_env, session_id, "mcp__harness__harness_start_agent", run_id)

    session_file = plugin_data / "sessions" / f"{session_id}.json"
    assert session_file.is_file()
    session_file.unlink()

    with _running_hook(json.dumps(_stop_payload(session_id)), plugin_data, extra_env=extra_env) as proc:
        _wait_for_file(session_file)
        time.sleep(1)
        flip_time = time.monotonic()
        _seed_record(artifacts_dir, run_id, terminal_state)

        out, err = proc.communicate(timeout=40)
        elapsed_since_flip = time.monotonic() - flip_time
        assert proc.returncode == 0, f"stdout={out!r} stderr={err!r}"
        assert err == "", err
        # tautology::F1: bound how soon the exit follows the flip so only a
        # loop that keeps re-reading record.json (not one that snapshots
        # once, sleeps to the 20s deadline, then snapshots again) can pass.
        assert elapsed_since_flip < 5, (
            f"hook took {elapsed_since_flip:.2f}s to exit after the run turned "
            f"terminal (timeout was 20s) -- expected it to notice within a few "
            f"poll intervals, not sleep to the deadline; stdout={out!r} stderr={err!r}"
        )


def test_stop_waits_for_all_tracked_runs_before_terminal(tmp_path):
    """R1 additional coverage: with two tracked runs, Stop keeps waiting
    while either one is non-terminal, and only exits 0 once both are.

    Expected RED reason: same as the driving test -- the current code
    exits 2 on the very first snapshot, so `proc.poll()` is already `2`
    (not `None`) by the time this test checks it."""
    plugin_data = tmp_path / "plugin-data"
    artifacts_dir = tmp_path / "artifacts"
    extra_env = {"HARNESS_ARTIFACTS_DIR": str(artifacts_dir), "HARNESS_STOP_WAIT_TIMEOUT_SECONDS": "20"}
    session_id = "sess-wait-two"
    run_a, run_b = "run-wait-a", "run-wait-b"

    _seed_record(artifacts_dir, run_a, RunState.RUNNING)
    _seed_record(artifacts_dir, run_b, RunState.RUNNING)
    _track(plugin_data, extra_env, session_id, "mcp__harness__harness_start_agent", run_a)
    _track(plugin_data, extra_env, session_id, "mcp__harness__harness_start_agent", run_b)

    session_file = plugin_data / "sessions" / f"{session_id}.json"
    session_file.unlink()

    with _running_hook(json.dumps(_stop_payload(session_id)), plugin_data, extra_env=extra_env) as proc:
        _wait_for_file(session_file)
        time.sleep(1)

        _seed_record(artifacts_dir, run_a, RunState.COMPLETED)
        time.sleep(1)
        assert proc.poll() is None, "must still wait while run_b is RUNNING"

        _seed_record(artifacts_dir, run_b, RunState.COMPLETED)
        out, err = proc.communicate(timeout=40)
        assert proc.returncode == 0, f"stdout={out!r} stderr={err!r}"


def test_stop_wait_survives_unreadable_record(tmp_path):
    """R2 driving test: a run already seen as RUNNING stays pending while
    its record.json is transiently unreadable, instead of the glitch being
    read as "gone" and letting Stop exit 0 early (the sticky-state rule,
    plan Approach).

    The corrupt window is sized off the real `_POLL_INTERVAL_SECONDS`
    constant (tautology::F2) rather than a hand-picked literal, so the
    test stays deterministic regardless of the actual poll interval: a
    loop without the sticky fallback that happened to poll slower than a
    hardcoded window would otherwise miss the glitch and pass wrongly.

    Expected RED reason: `_POLL_INTERVAL_SECONDS` does not exist in
    `harness_plugin.hooks.write_context` yet, so this import raises
    ImportError. Once the poll loop exists but without the sticky
    fallback, the current code has already exited 2 on its first
    (successful) snapshot -- or a non-sticky loop would read the glitch as
    "gone" and exit 0 during the corrupt window -- either way
    `proc.poll() is None` is false."""
    from harness_plugin.hooks.write_context import _POLL_INTERVAL_SECONDS

    plugin_data = tmp_path / "plugin-data"
    artifacts_dir = tmp_path / "artifacts"
    extra_env = {"HARNESS_ARTIFACTS_DIR": str(artifacts_dir), "HARNESS_STOP_WAIT_TIMEOUT_SECONDS": "20"}
    session_id = "sess-glitch"
    run_id = "run-glitch"

    _seed_record(artifacts_dir, run_id, RunState.RUNNING)
    _track(plugin_data, extra_env, session_id, "mcp__harness__harness_start_agent", run_id)

    session_file = plugin_data / "sessions" / f"{session_id}.json"
    session_file.unlink()

    # At least 3 poll intervals of corruption, so a loop without the sticky
    # fallback is guaranteed a read attempt during the glitch regardless of
    # the constant's actual value.
    corrupt_duration = max(_POLL_INTERVAL_SECONDS * 3, 1.0)

    with _running_hook(json.dumps(_stop_payload(session_id)), plugin_data, extra_env=extra_env) as proc:
        _wait_for_file(session_file)
        time.sleep(1)

        (artifacts_dir / run_id / "record.json").write_text("{", encoding="utf-8")
        time.sleep(corrupt_duration)
        assert proc.poll() is None, (
            f"a transient unreadable record must not end the wait (corrupted "
            f"for {corrupt_duration:.2f}s, >= 3 poll intervals of "
            f"{_POLL_INTERVAL_SECONDS}s each)"
        )

        _seed_record(artifacts_dir, run_id, RunState.COMPLETED)
        out, err = proc.communicate(timeout=40)
        assert proc.returncode == 0, f"stdout={out!r} stderr={err!r}"


def test_stop_blocks_after_wait_timeout(tmp_path):
    """R3 driving test: with the timeout env var set below how long the run
    stays RUNNING, Stop exits 2 only once that timeout elapses, still
    naming the pending run_id -- a timeout never cancels the run (plan
    Approach).

    Expected RED reason: the current code returns immediately (no wait at
    all), so `elapsed >= 2.0` fails."""
    plugin_data = tmp_path / "plugin-data"
    artifacts_dir = tmp_path / "artifacts"
    timeout_seconds = 2.0
    extra_env = {
        "HARNESS_ARTIFACTS_DIR": str(artifacts_dir),
        "HARNESS_STOP_WAIT_TIMEOUT_SECONDS": str(timeout_seconds),
    }
    session_id = "sess-timeout"
    run_id = "run-timeout"

    _seed_record(artifacts_dir, run_id, RunState.RUNNING)
    _track(plugin_data, extra_env, session_id, "mcp__harness__harness_start_agent", run_id)

    start = time.monotonic()
    stop = run_hook(json.dumps(_stop_payload(session_id)), plugin_data, extra_env=extra_env)
    elapsed = time.monotonic() - start

    assert stop.returncode == 2, f"stdout={stop.stdout!r} stderr={stop.stderr!r}"
    assert run_id in stop.stderr
    assert elapsed >= timeout_seconds, f"elapsed={elapsed}"
    # tautology::F1: also bound how much *longer* than the configured env var
    # the hook may take, so this pins the implementation to actually reading
    # HARNESS_STOP_WAIT_TIMEOUT_SECONDS (`timeout_seconds` above) rather than
    # a hardcoded deadline that happens to also be >= 2.0 (e.g. the old 7200s
    # default, or a bug that ignored the env var and used a much longer
    # fixed wait). Margin covers poll-interval slop and process overhead.
    assert elapsed < timeout_seconds + 5, (
        f"hook took {elapsed:.2f}s to exit, expected close to the configured "
        f"{timeout_seconds}s timeout (plus poll/process overhead); "
        f"stdout={stop.stdout!r} stderr={stop.stderr!r}"
    )


@pytest.mark.parametrize(
    "env_value,expected",
    [
        (None, 7200.0),
        ("1.5", 1.5),
        ("0", 0.0),
        ("-3", 0.0),
        ("abc", 7200.0),
        ("nan", 7200.0),
        ("99999", 7200.0),
    ],
    ids=["unset", "one-point-five", "zero", "negative", "non-numeric", "nan", "above-default"],
)
def test_stop_wait_timeout_parsing(env_value, expected):
    """R4 driving test (in-process): host_context.stop_wait_timeout parses
    HARNESS_STOP_WAIT_TIMEOUT_SECONDS per plan Approach -- unset/non-numeric/
    non-finite gives the 7200s default, negative clamps to 0, and anything
    above the default clamps down to it.

    Expected RED reason: host_context has no stop_wait_timeout yet, so this
    import raises ImportError."""
    from harness_plugin.host_context import STOP_WAIT_TIMEOUT_ENV, stop_wait_timeout

    env = {} if env_value is None else {STOP_WAIT_TIMEOUT_ENV: env_value}
    assert stop_wait_timeout(env) == expected


def test_hooks_json_stop_timeout_exceeds_internal_wait():
    """R5 driving test: every hooks.json Stop entry's timeout sits
    comfortably above the longest possible internal wait, so Claude Code's
    own hook timeout can never cut a real wait short.

    The expected threshold is derived by calling the real
    `stop_wait_timeout()` clamp itself with a deliberately huge override
    (tautology::F5), not by hand-copying `STOP_WAIT_TIMEOUT_DEFAULT_SECONDS`
    plus a literal margin -- a future change to the clamp's own ceiling
    logic (not just to the default constant) is caught too, since a
    constant that disagreed with the applied clamp would otherwise let a
    hooks.json value below the real internal wait pass.

    Expected RED reason: host_context has no stop_wait_timeout /
    HARNESS_STOP_WAIT_TIMEOUT_SECONDS yet, so this import raises
    ImportError; once it exists, hooks.json's unchanged 10 fails against
    the real computed ceiling."""
    from harness_plugin.host_context import STOP_WAIT_TIMEOUT_ENV, stop_wait_timeout

    # The real clamp's own ceiling: an absurdly large override still clamps
    # down to whatever stop_wait_timeout() actually treats as its maximum --
    # the true longest possible internal wait, read from the real code path
    # rather than trusted to match a separately-copied constant.
    max_possible_wait = stop_wait_timeout({STOP_WAIT_TIMEOUT_ENV: "999999999999"})
    assert max_possible_wait > 0, "stop_wait_timeout produced a non-positive ceiling"

    margin_seconds = 60
    data = json.loads((REPO / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    stop_groups = data["hooks"]["Stop"]
    assert stop_groups, "no hooks.json Stop group"
    for group in stop_groups:
        for entry in group["hooks"]:
            assert entry["timeout"] >= max_possible_wait + margin_seconds, entry


def test_stop_message_rendered_from_file(tmp_path):
    """R6 driving test: the exit-2 message is actually loaded and rendered
    from hooks/stop_wait_message.md at runtime, not reproduced from #62's
    old hardcoded prose. Two distinct, uniquely-marked versions of the file
    are written in turn; a hook that genuinely reads the file must
    round-trip each marker verbatim (with {run_ids} filled in) into a
    blocked Stop's stderr, and switching the file's content must switch
    stderr's content. A bare "does the file contain {run_ids}" check
    (tautology::F4) would pass for any file with that token whether or not
    the hook ever opens it; reproducing this with the file's own real prose
    (tautology::F3) would also pass for an unmodified hook that never reads
    the file at all. The two distinct random markers below rule out both:
    no coincidental wording overlap is possible, and marker A must
    disappear once the file switches to marker B.

    Expected RED reason: the current hook still calls the hardcoded
    `_stop_block_message` and never opens hooks/stop_wait_message.md at
    all, so neither marker ever reaches stderr -- `assert marker_a in
    stderr_a` fails first."""
    message_path = REPO / "hooks" / "stop_wait_message.md"
    original = message_path.read_text(encoding="utf-8") if message_path.is_file() else None

    def _stderr_for_template(template_text):
        message_path.write_text(template_text, encoding="utf-8")
        plugin_data = tmp_path / f"plugin-data-{uuid.uuid4().hex}"
        artifacts_dir = tmp_path / f"artifacts-{uuid.uuid4().hex}"
        extra_env = {"HARNESS_ARTIFACTS_DIR": str(artifacts_dir)}
        session_id = "sess-message"
        run_id = "run-message"
        _seed_record(artifacts_dir, run_id, RunState.RUNNING)
        _track(plugin_data, extra_env, session_id, "mcp__harness__harness_start_agent", run_id)

        stop = run_hook(json.dumps(_stop_payload(session_id)), plugin_data, extra_env=extra_env)
        assert stop.returncode == 2, f"stdout={stop.stdout!r} stderr={stop.stderr!r}"
        assert run_id in stop.stderr
        # tautology::F2: `run_id in stop.stderr` alone would also pass a hook
        # that appended run_id somewhere unrelated to the {run_ids}
        # placeholder (e.g. always tacked on at the end, ignoring the
        # template's own layout). Locate the placeholder's fixed prefix in
        # the rendered output and require run_id to start exactly there --
        # i.e. genuinely substituted at the placeholder's position, not just
        # present somewhere in stderr.
        prefix, _, _ = template_text.partition("{run_ids}")
        prefix_at = stop.stderr.find(prefix)
        assert prefix_at != -1, (
            f"template's own fixed prefix before {{run_ids}} not found verbatim "
            f"in stderr; stderr={stop.stderr!r}"
        )
        after_prefix = stop.stderr[prefix_at + len(prefix):]
        assert after_prefix.startswith(run_id), (
            f"run_id must be substituted exactly at the {{run_ids}} placeholder's "
            f"position, not merely present elsewhere in stderr; "
            f"after_prefix={after_prefix!r}"
        )
        return stop.stderr

    try:
        marker_a = f"MARKER-A-{uuid.uuid4().hex}"
        template_a = f"{marker_a} pending run(s): {{run_ids}} -- {marker_a}-tail"
        stderr_a = _stderr_for_template(template_a)
        assert marker_a in stderr_a, (
            f"hook did not render hooks/stop_wait_message.md's own content "
            f"(marker A missing); stderr={stderr_a!r}"
        )

        marker_b = f"MARKER-B-{uuid.uuid4().hex}"
        template_b = f"{marker_b} a completely different wording: {{run_ids}} :: {marker_b}-end"
        stderr_b = _stderr_for_template(template_b)
        assert marker_b in stderr_b, (
            f"hook did not render the file's new content after it changed "
            f"(marker B missing); stderr={stderr_b!r}"
        )
        assert marker_a not in stderr_b, (
            "stderr still carries marker A after the file's content changed -- "
            f"the hook is not re-reading the file at runtime; stderr={stderr_b!r}"
        )
    finally:
        if original is None:
            message_path.unlink(missing_ok=True)
        else:
            message_path.write_text(original, encoding="utf-8")


# --- #65: fixed STOP_MARKER prefix + exact committed-file rendering --------------
#
# STOP_MARKER is imported by tests/test_live_stop_message.py -- the live scenarios'
# block detection there needs a guaranteed distinctive marker to find a Stop block in
# real `claude` stream-json stdout; the pre-#65 prefix ("agent-harness: run(s) ") is
# too generic (could coincidentally appear in ordinary model output) for that.

STOP_MARKER = "agent-harness Stop hook: waited for unfinished run(s)"


def test_stop_sends_committed_message_file(tmp_path):
    """R3 driving test: a real Stop subprocess, blocked on one tracked RUNNING run,
    renders the *actual committed* hooks/stop_wait_message.md (not a swapped-in
    template like test_stop_message_rendered_from_file uses) -- stderr is exactly
    that file's `{run_ids}` placeholder filled in with `run-x (RUNNING)`, and starts
    with the fixed STOP_MARKER prefix #65 introduces.

    Expected RED reason: the current committed file starts with "agent-harness:
    run(s) {run_ids} started in this session are not finished; call harness_wait_run
    / harness_poll_run until they are terminal, or harness_stop_run to cancel,
    before ending the turn." -- `stop.stderr.startswith(STOP_MARKER)` fails on
    unfixed code."""
    message_path = REPO / "hooks" / "stop_wait_message.md"
    template = message_path.read_text(encoding="utf-8")
    assert template.count("{run_ids}") == 1, (
        f"expected exactly one {{run_ids}} placeholder in {message_path}, "
        f"found {template.count('{run_ids}')}"
    )

    plugin_data = tmp_path / "plugin-data"
    artifacts_dir = tmp_path / "artifacts"
    extra_env = {"HARNESS_ARTIFACTS_DIR": str(artifacts_dir)}
    session_id = "sess-marker"
    run_id = "run-x"

    _seed_record(artifacts_dir, run_id, RunState.RUNNING)
    _track(plugin_data, extra_env, session_id, "mcp__harness__harness_start_agent", run_id)

    stop = run_hook(json.dumps(_stop_payload(session_id)), plugin_data, extra_env=extra_env)
    assert stop.returncode == 2, f"stdout={stop.stdout!r} stderr={stop.stderr!r}"

    expected = template.replace("{run_ids}", f"{run_id} (RUNNING)").strip()
    assert stop.stderr.strip() == expected, (
        f"stderr does not match the committed file rendered with run_ids filled in; "
        f"stderr={stop.stderr!r} expected={expected!r}"
    )
    assert stop.stderr.startswith(STOP_MARKER), (
        f"stderr does not start with the fixed STOP_MARKER prefix; stderr={stop.stderr!r}"
    )
