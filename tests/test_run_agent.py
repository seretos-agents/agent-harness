"""`harness run-agent`: the blocking entry point behind native `Agent` calls.

Starts a run for a named agent, waits for it in the same process (no timeout, never
cancels) and prints one `run_to_dict` JSON line. Exit codes: 0 COMPLETED, 1 FAILED,
3 CANCELLED, 4 error (unknown agent, refused context, bad arguments); on 4 stdout is
empty and the message is on stderr."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import FAKE_CLAUDE, SESSION_ID
from lib_python_harness import FileRunStore, Harness

CLI_TIMEOUT = 90


def _env(wait_run_env, **extra):
    return {**wait_run_env, "CLAUDE_CODE_SESSION_ID": SESSION_ID, **extra}


def _spawn(cmd, env, *args):
    return subprocess.Popen(
        [*cmd, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
    )


def _finish(proc, timeout=CLI_TIMEOUT):
    out, err = proc.communicate(timeout=timeout)
    return proc.returncode, out, err


def _run_cli(cmd, env, *args, timeout=15):
    """Run to completion, bounded: reports a "blocked" sentinel instead of hanging when the
    subcommand does not exist (argv then falls through to the MCP server)."""
    proc = _spawn(cmd, env, *args)
    try:
        return _finish(proc, timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        return "blocked", out, err


def _one_json(out):
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 1, f"stdout must be exactly one JSON object, got {out!r}"
    return json.loads(lines[0])


def _argv_entries(env):
    with open(env["HARNESS_FAKE_ARGV_LOG"], encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def _only_run_id(env):
    root = Path(env["HARNESS_ARTIFACTS_DIR"])
    runs = [p.name for p in root.iterdir() if (p / "record.json").is_file()]
    assert len(runs) == 1, runs
    return runs[0]


def _stored_result(env, run_id):
    h = Harness(
        store=FileRunStore(env["HARNESS_ARTIFACTS_DIR"]),
        claude_argv=[sys.executable, str(FAKE_CLAUDE)],
    )
    return h.wait(run_id, timeout=10, poll_interval=0.2)


def test_run_agent_completed_returns_answer_and_run_id(
    session_context, wait_run_env, run_agent_cmd
):
    env = _env(wait_run_env)
    code, out, err = _run_cli(
        run_agent_cmd, env, "--subagent-type", "demo", "--prompt", "ECHO:alpha",
        "--description", "my label",
    )
    assert code == 0, (out, err)
    payload = _one_json(out)
    assert payload["state"] == "COMPLETED"
    assert payload["text"] == "alpha"
    record_path = Path(env["HARNESS_ARTIFACTS_DIR"]) / payload["run_id"] / "record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["state"] == "COMPLETED"
    # Additional coverage: --description becomes the run label.
    assert record.get("label") == "my label"


def test_run_agent_failed_exits_one(session_context, wait_run_env, run_agent_cmd):
    env = _env(wait_run_env)
    code, out, err = _run_cli(
        run_agent_cmd, env, "--subagent-type", "demo", "--prompt", "NO_RESULT"
    )
    assert code == 1, (out, err)
    payload = _one_json(out)
    assert payload["state"] == "FAILED"
    assert payload["text"] == _stored_result(env, payload["run_id"]).text


# Upstream limit, not a harness bug (same reason as test_wait_run.test_cancelled_run_exits_three).
@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "lib-python-harness v0.0.9+: Harness.stop() racing wait_for does not reliably "
        "yield CANCELLED on Windows; exit 3 is covered on Linux"
    ),
)
def test_run_agent_cancelled_exits_three(session_context, wait_run_env, run_agent_cmd):
    env = _env(wait_run_env)
    proc = _spawn(run_agent_cmd, env, "--subagent-type", "demo", "--prompt", "SLEEP:30")
    deadline = time.monotonic() + 20
    run_id = None
    while time.monotonic() < deadline and run_id is None:
        try:
            run_id = _only_run_id(env)
        except (FileNotFoundError, AssertionError):
            time.sleep(0.2)
    assert run_id, "run-agent never started a run"
    time.sleep(1)
    Harness(
        store=FileRunStore(env["HARNESS_ARTIFACTS_DIR"]),
        claude_argv=[sys.executable, str(FAKE_CLAUDE)],
    ).stop(run_id)
    code, out, err = _finish(proc)
    assert code == 3, (out, err)
    payload = _one_json(out)
    assert payload["state"] == "CANCELLED"
    assert payload["run_id"] == run_id
    # The text field must carry the run's own text, not be dropped or invented.
    assert payload["text"] == _stored_result(env, run_id).text


def test_run_agent_unknown_agent_lists_known(session_context, wait_run_env, run_agent_cmd):
    env = _env(wait_run_env)
    code, out, err = _run_cli(run_agent_cmd, env, "--subagent-type", "nope", "--prompt", "hi")
    assert code == 4, (out, err)
    assert out.strip() == ""
    assert "unknown agent 'nope'" in err
    assert "known agents:" in err
    assert "demo" in err


def test_run_agent_missing_prompt_exits_four(session_context, wait_run_env, run_agent_cmd):
    env = _env(wait_run_env)
    code, out, err = _run_cli(run_agent_cmd, env, "--subagent-type", "demo")
    assert code == 4, (out, err)
    assert out.strip() == ""


@pytest.mark.parametrize("name", ["general-purpose", "Explore", "Plan"])
def test_run_agent_maps_builtin_types(
    name, session_context, shipped_agents_install, wait_run_env, run_agent_cmd
):
    env = _env(wait_run_env)
    code, out, err = _run_cli(
        run_agent_cmd, env, "--subagent-type", name, "--prompt", "ECHO:beta"
    )
    assert code == 0, (out, err)
    assert [e["launched_agent"] for e in _argv_entries(env)] == [f"agent-harness:{name}"]


def test_run_agent_project_agent_bare_name(
    session_context, shipped_agents_install, wait_run_env, run_agent_cmd
):
    env = _env(wait_run_env)
    code, out, err = _run_cli(
        run_agent_cmd, env, "--subagent-type", "demo", "--prompt", "ECHO:beta"
    )
    assert code == 0, (out, err)
    assert [e["launched_agent"] for e in _argv_entries(env)] == ["demo"]


def test_run_agent_model_and_parent_context(session_context, wait_run_env, run_agent_cmd):
    env = _env(wait_run_env)
    code, out, err = _run_cli(
        run_agent_cmd, env, "--subagent-type", "demo", "--prompt", "ECHO:a", "--model", "sonnet"
    )
    assert code == 0, (out, err)
    argv = _argv_entries(env)[-1]["argv"]
    assert _flag(argv, "--model") == "sonnet"
    assert _flag(argv, "--permission-mode") == "acceptEdits"
    assert _flag(argv, "--effort") == "high"
    payload = _one_json(out)
    assert (payload["model"], payload["permission_mode"], payload["effort"]) == (
        "sonnet",
        "acceptEdits",
        "high",
    )

    # Without --model the session's own model is used.
    code, out, err = _run_cli(
        run_agent_cmd, env, "--subagent-type", "demo", "--prompt", "ECHO:b"
    )
    assert code == 0, (out, err)
    assert _flag(_argv_entries(env)[-1]["argv"], "--model") == "opus"


def test_run_agent_no_session_context_exits_four(wait_run_env, run_agent_cmd):
    env = {**wait_run_env, "CLAUDE_CODE_SESSION_ID": "no-such-session"}
    code, out, err = _run_cli(run_agent_cmd, env, "--subagent-type", "demo", "--prompt", "hi")
    assert code == 4, (out, err)
    assert out.strip() == ""
    assert "cannot determine the parent session's context" in err


def test_run_agent_long_run_returns(session_context, wait_run_env, run_agent_cmd):
    """A run longer than any hook-style 10 s limit still returns normally."""
    env = _env(wait_run_env)
    code, out, err = _run_cli(
        run_agent_cmd, env, "--subagent-type", "demo", "--prompt", "SLEEP:12", timeout=60
    )
    assert code == 0, (out, err)
    payload = _one_json(out)
    assert payload["state"] == "COMPLETED"
    assert payload["duration_s"] >= 11
