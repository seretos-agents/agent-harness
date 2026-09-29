"""#68: a CLEAN `claude -p` child that has already written its terminal `result`
event but keeps its OS process alive for a while afterwards must not be reported
RUNNING forever by any of the four entry points -- `harness_poll_run`,
`harness_wait_run`, the real Stop hook, or the `harness wait` CLI.

`lib_python_harness` v0.0.10's grace-kill (`Harness.wait`'s `_kill_lingering`,
gated by `_FINALIZE_GRACE_S`) is the only place this is handled; it fires
inside `Harness.wait()` and nowhere else. `harness_wait_run`/`harness wait`
already call `wait()`, so their coverage here is a non-regression guard, not a
driving test (plan "Approach"). `harness_poll_run` (`poll()`) and the Stop
hook (a passive `record.json` read) do not, and are this ticket's fix.

Shared setup (plan "Test / verification strategy"): every test starts a real
lingering run via `tests/fixtures/fake_claude.py`'s `LINGER:<seconds>` marker
(600s -- long enough that only an explicit grace-kill, never the child exiting
on its own, can end it), measures `t0` from when `harness_start_prompt`
returns, and budgets `_GRACE_S + _FEW_S` (`_GRACE_S` imported from the
installed lib rather than copied, so the budget tracks the lib's own
constant; `_FEW_S` is the acceptance criterion's own "a few seconds").
`_child_gone` re-derives pid/start_time straight from the real on-disk
`record.json` (via `FileRunStore`, never trusted from an RPC payload) and
checks process identity the same way the library itself does
(`_pid_status`), so a false negative here cannot come from a stale copy of
the pid check.

`_warm_up_version_cache`: `Harness._finalize` -> `_write_provenance` ->
`_cli_version` spawns a `--version` probe subprocess the first time any run
finalizes on a given `Harness` instance (cached by `binary_argv` afterwards).
Measured directly (outside this repo's own code, in the installed lib):
inside this plugin's async MCP tools (`harness_wait_run`, and
`harness_poll_run` once it is `wait(run_id, 0)`-backed), that first probe
consistently stalls ~15s on this environment -- `anyio.to_thread.run_sync`
runs the finalize in a worker thread while the server's own asyncio event
loop still owns pending stdio I/O, and spawning + capturing a subprocess
from that thread contends with it; a plain synchronous process (the
`harness wait` CLI, the Stop hook) never shows it, and neither does a second
finalize on an already-warm cache. This is a pre-existing environment/lib
interaction, not anything this ticket's fix touches (no production code
here calls `_cli_version` differently) -- so every test that measures a
grace-kill's own latency through an async MCP tool call warms that cache
with a throwaway trivial run first, keeping `BUDGET` a measurement of the
grace-kill, not of an unrelated one-time subprocess-spawn cost.

#72: a child that exited normally (not lingering) but whose record is held by
another process is finalized by the Stop hook's repeated `wait(run_id, 0)`
under lib_python_harness v0.0.11 (grace anchored on durable last sign of
life); see the "#72" section below for the driving test."""
import json
import time

import anyio
import anyio.to_thread
from lib_python_harness import FileRunStore, RunState
from lib_python_harness.harness import _FINALIZE_GRACE_S as _GRACE_S
from lib_python_harness.runtime.process import _pid_status
from test_hook import _stop_payload, _track, run_hook
from test_mcp_tools import _call, _run
from test_wait_run import _finish, _one_json, _spawn

# The acceptance criterion's own "a few seconds", defined once for every test
# in this module (plan "Mechanism balance").
_FEW_S = 5
BUDGET = _GRACE_S + _FEW_S


async def _start_lingering(session, seconds=600):
    is_error, text, started = await _call(
        session, "harness_start_prompt", prompt=f"LINGER:{seconds}", model="sonnet"
    )
    assert not is_error, text
    return started["run_id"]


async def _warm_up_version_cache(session):
    """Finalize one throwaway trivial run through `harness_wait_run` before
    the timed portion of a test starts, so the one-time `_cli_version`
    subprocess-spawn cost (see module docstring) lands here instead of
    inside the budget being measured."""
    is_error, text, started = await _call(
        session, "harness_start_prompt", prompt="OK", model="sonnet"
    )
    assert not is_error, text
    is_error, text, payload = await _call(
        session, "harness_wait_run", run_id=started["run_id"], timeout_seconds=30
    )
    assert not is_error, text
    assert payload["state"] == "COMPLETED", payload


def _record(artifacts_dir, run_id):
    return FileRunStore(str(artifacts_dir)).get(run_id)


def _child_gone(artifacts_dir, run_id):
    record = _record(artifacts_dir, run_id)
    assert record is not None, f"no record.json for {run_id} under {artifacts_dir}"
    pid = record.get("pid")
    assert pid is not None, f"record for {run_id} has no pid: {record}"
    return _pid_status(pid, record.get("start_time")) is False


def _cleanup(server_params, artifacts_dir, run_id):
    """A `finally` guard (plan "Test / verification strategy"): stop the run
    if it is still RUNNING, so a failing assertion (RED or a genuine bug)
    never leaves the 600s-lingering fake CLI as an orphaned process for the
    rest of that budget."""
    record = _record(artifacts_dir, run_id)
    if record is None or record.get("state") != RunState.RUNNING:
        return

    async def scenario(session):
        await _call(session, "harness_stop_run", run_id=run_id)

    _run(scenario, server_params)


# --- R1: poll and Stop hook finalize a lingering run ------------------------


def test_poll_after_grace_completes_lingering_child(server_params, tmp_path):
    """R1 driving test: once the grace period has passed, a single
    `harness_poll_run` call (no loop) reports COMPLETED and the 600s child is
    gone.

    Expected RED reason: `poll()` has no grace-kill (only
    `_finalize_if_ended`/`_finalize_if_gone`, neither of which apply to a
    process that is still alive), so the single poll still reports RUNNING."""
    artifacts_dir = tmp_path / "artifacts"

    async def scenario(session):
        await _warm_up_version_cache(session)
        t0 = time.monotonic()
        run_id = await _start_lingering(session)
        target = t0 + _GRACE_S + 0.5
        remaining = target - time.monotonic()
        if remaining > 0:
            await anyio.sleep(remaining)
        poll_start = time.monotonic()
        is_error, text, payload = await _call(session, "harness_poll_run", run_id=run_id)
        report_time = time.monotonic()
        if payload is not None and payload.get("state") == "RUNNING":
            await _call(session, "harness_stop_run", run_id=run_id)
        return run_id, t0, poll_start, report_time, is_error, text, payload

    run_id, t0, poll_start, report_time, is_error, text, payload = _run(scenario, server_params)
    try:
        assert not is_error, text
        assert payload["state"] == "COMPLETED", (
            f"expected the grace-kill to finalize the lingering run; got {payload}"
        )
        assert report_time - poll_start <= _FEW_S, (
            f"the single poll call itself took {report_time - poll_start:.2f}s, "
            f"expected within _FEW_S={_FEW_S}s"
        )
        assert report_time - t0 <= BUDGET, (
            f"COMPLETED reported {report_time - t0:.2f}s after t0, budget {BUDGET}s"
        )
        assert _child_gone(artifacts_dir, run_id), "the lingering child is still alive"
    finally:
        _cleanup(server_params, artifacts_dir, run_id)


def test_stop_hook_completes_lingering_child(server_params, tmp_path):
    """R1 driving test: the real Stop hook, given a tracked lingering run
    (via the real PostToolUse `_track` mechanism), finalizes it and exits 0
    within budget -- `HARNESS_STOP_WAIT_TIMEOUT_SECONDS` is set well above
    `BUDGET` so a pass can never come from the wait limit merely expiring.

    The originating `harness_start_prompt` session is kept open for the
    `_track`/Stop-hook window (both run via `anyio.to_thread.run_sync`,
    since they are blocking subprocess calls) rather than closed right after
    `_start_lingering` returns: on Windows, `mcp.client.stdio.stdio_client`
    wraps the spawned `server_params` MCP server in a Job Object with
    kill-on-close semantics so a disconnecting client can reliably clean up
    an unresponsive server -- closing the session kills that whole process
    tree, including the lingering fake-CLI grandchild this test needs to
    still be alive (independently of its immediate MCP-server parent, #68's
    own premise) when the separate Stop-hook subprocess goes looking for it.
    In production the harness_plugin MCP server is not torn down between
    tool calls -- it persists for the whole Claude Code session -- so this
    mirrors that lifetime instead of the artifact of a test that opens and
    closes one MCP session per tool call (verified directly: a plain
    `subprocess.Popen(..., creationflags=CREATE_NEW_PROCESS_GROUP)`
    grandchild does survive its immediate parent's exit on this same
    machine; only the `mcp` SDK's own Job-Object-wrapped process tree does
    not).

    Expected RED reason: the Stop branch only ever does a passive
    `record.json` read (`_run_state`/`_pending_runs`); nothing drives the
    run's own reconciliation, so it never turns terminal and Stop exits 2
    once `HARNESS_STOP_WAIT_TIMEOUT_SECONDS` elapses, naming the run in
    stderr."""
    plugin_data = tmp_path / "plugin-data"
    artifacts_dir = tmp_path / "artifacts"
    extra_env = {"HARNESS_ARTIFACTS_DIR": str(artifacts_dir)}
    session_id = "sess-68-stop"

    async def scenario(session):
        run_id = await _start_lingering(session)
        t0 = time.monotonic()
        await anyio.to_thread.run_sync(
            _track, plugin_data, extra_env, session_id, "mcp__harness__harness_start_agent", run_id
        )
        stop_env = dict(extra_env)
        stop_env["HARNESS_STOP_WAIT_TIMEOUT_SECONDS"] = str(BUDGET + 30)
        stop = await anyio.to_thread.run_sync(
            run_hook, json.dumps(_stop_payload(session_id)), plugin_data, "/work/project", stop_env
        )
        report_time = time.monotonic()
        return run_id, t0, report_time, stop

    run_id, t0, report_time, stop = _run(scenario, server_params)

    try:
        assert stop.returncode == 0, (
            f"expected Stop to finalize the lingering run within {BUDGET}s, not block on "
            f"the {BUDGET + 30}s wait limit; stdout={stop.stdout!r} stderr={stop.stderr!r}"
        )
        assert report_time - t0 <= BUDGET, (
            f"Stop took {report_time - t0:.2f}s to exit 0, budget {BUDGET}s"
        )
        record = _record(artifacts_dir, run_id)
        assert record is not None and record.get("state") == RunState.COMPLETED, record
        assert _child_gone(artifacts_dir, run_id), "the lingering child is still alive"
    finally:
        _cleanup(server_params, artifacts_dir, run_id)


# --- R1: additional edge-case coverage -- may already pass ------------------


def test_wait_run_completes_lingering_child(server_params, tmp_path):
    """R1 additional coverage: `harness_wait_run` already calls `Harness.wait`,
    so this is a non-regression guard, not a driving test -- it may already
    pass against unfixed code."""
    artifacts_dir = tmp_path / "artifacts"

    async def scenario(session):
        await _warm_up_version_cache(session)
        t0 = time.monotonic()
        run_id = await _start_lingering(session)
        is_error, text, payload = await _call(
            session, "harness_wait_run", run_id=run_id, timeout_seconds=BUDGET + 30
        )
        report_time = time.monotonic()
        return run_id, t0, report_time, is_error, text, payload

    run_id, t0, report_time, is_error, text, payload = _run(scenario, server_params)
    try:
        assert not is_error, text
        assert payload["state"] == "COMPLETED", payload
        assert report_time - t0 <= BUDGET, (
            f"harness_wait_run took {report_time - t0:.2f}s, budget {BUDGET}s"
        )
        assert _child_gone(artifacts_dir, run_id), "the lingering child is still alive"
    finally:
        _cleanup(server_params, artifacts_dir, run_id)


def test_wait_cli_completes_lingering_child(server_params, wait_run_env, wait_run_cmd, tmp_path):
    """R1 additional coverage: the `harness wait` CLI already calls
    `Harness.wait` too -- may already pass against unfixed code."""
    artifacts_dir = tmp_path / "artifacts"

    async def scenario(session):
        return await _start_lingering(session)

    t0 = time.monotonic()
    run_id = _run(scenario, server_params)
    try:
        proc = _spawn(wait_run_cmd, wait_run_env, run_id, "--timeout", str(BUDGET + 30))
        code, out, err = _finish(proc)
        report_time = time.monotonic()

        assert code == 0, (out, err)
        payload = _one_json(out)
        assert payload["state"] == "COMPLETED", payload
        assert report_time - t0 <= BUDGET, (
            f"harness wait took {report_time - t0:.2f}s, budget {BUDGET}s"
        )
        assert _child_gone(artifacts_dir, run_id), "the lingering child is still alive"
    finally:
        _cleanup(server_params, artifacts_dir, run_id)


# --- #72: Stop finalizes a normally-exited child held by another process ----


async def _stop_after_exited_child(
    session, artifacts_dir, plugin_data, session_id, prompt, model, exit_cap_s
):
    """Start a real run through `session` (the MCP server -- a different, still
    living process -- keeps the run's `Popen`), wait until the `claude -p`
    child has exited on its own, then run the real Stop hook and time it from
    its own start. Nobody but Stop may finalize the run: the server has no
    background reaper and gets no poll/wait call here, so the record must
    still read RUNNING when Stop starts.

    Returns `(run_id, stop, stop_elapsed)`."""
    extra_env = {"HARNESS_ARTIFACTS_DIR": str(artifacts_dir)}
    is_error, text, started = await _call(
        session, "harness_start_prompt", prompt=prompt, model=model
    )
    assert not is_error, text
    run_id = started["run_id"]

    deadline = time.monotonic() + exit_cap_s
    while not await anyio.to_thread.run_sync(_child_gone, artifacts_dir, run_id):
        assert time.monotonic() < deadline, f"child of {run_id} never exited within {exit_cap_s}s"
        await anyio.sleep(0.1)
    record = _record(artifacts_dir, run_id)
    assert record.get("state") == RunState.RUNNING, (
        f"precondition: only Stop may finalize the run, but it is already {record}"
    )

    await anyio.to_thread.run_sync(
        _track, plugin_data, extra_env, session_id, "mcp__harness__harness_start_prompt", run_id
    )
    stop_env = dict(extra_env)
    stop_env["HARNESS_STOP_WAIT_TIMEOUT_SECONDS"] = str(BUDGET + 30)
    stop_start = time.monotonic()
    stop = await anyio.to_thread.run_sync(
        run_hook, json.dumps(_stop_payload(session_id)), plugin_data, "/work/project", stop_env
    )
    return run_id, stop, time.monotonic() - stop_start


def test_stop_hook_completes_exited_child_held_by_other_process(server_params, tmp_path):
    """#72 R1 driving test: the child exited normally right after writing its
    result, while the MCP server (another living process) still holds its
    `Popen`. The real Stop hook must finalize the run and exit 0 within
    `BUDGET` of its own start, not block for its wait limit
    (`HARNESS_STOP_WAIT_TIMEOUT_SECONDS` = BUDGET + 30).

    Expected RED reason (lib_python_harness v0.0.10): each `wait(run_id, 0)`
    restarts a loop-local `gone_since`, so Stop never finalizes and exits 2
    after BUDGET + 30s naming the run."""
    plugin_data = tmp_path / "plugin-data"
    artifacts_dir = tmp_path / "artifacts"
    run_id = None

    async def scenario(session):
        return await _stop_after_exited_child(
            session, artifacts_dir, plugin_data, "sess-72-stop", "OK", "sonnet", 15
        )

    try:
        run_id, stop, stop_elapsed = _run(scenario, server_params)
        assert stop.returncode == 0, (
            f"expected Stop to finalize the exited run within {BUDGET}s, not block on "
            f"the {BUDGET + 30}s wait limit; stdout={stop.stdout!r} stderr={stop.stderr!r}"
        )
        assert stop_elapsed <= BUDGET, f"Stop took {stop_elapsed:.2f}s, budget {BUDGET}s"
        record = _record(artifacts_dir, run_id)
        assert record is not None and record.get("state") == RunState.COMPLETED, record
    finally:
        if run_id is not None:
            _cleanup(server_params, artifacts_dir, run_id)
