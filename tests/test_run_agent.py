"""`harness run-agent`: the blocking entry point behind native `Agent` calls.

Starts a run for a named agent, waits for it in the same process (no timeout, never
cancels) and prints one `run_to_dict` JSON line. Exit codes: 0 COMPLETED, 1 FAILED,
3 CANCELLED, 4 error (unknown agent, refused context, bad arguments); on 4 stdout is
empty and the message is on stderr."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import FAKE_CLAUDE, SESSION_ID, plant_session_file
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
    assert record["state"] == {"__runstate__": "COMPLETED"}
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


# --- #77: run-agent called from inside a harness child (grandchild runs) ---------------

CHILD_SESSION = "child-session"
_CAN_SPAWN = """\
agents:
  demo:
    canSpawn: {}
"""


def _child_project(tmp_path, project_dir) -> Path:
    child = tmp_path / "child-project"
    shutil.copytree(project_dir, child)
    (child / ".git").mkdir()
    return child


def _child_env(wait_run_env, caller, session=CHILD_SESSION, project=None):
    """What a harness child's module spawn sees: no CLAUDE_CODE_SESSION_ID (the lib scrubs
    it), the launch identity carriers, and the parent's CLAUDE_PROJECT_DIR leaking through."""
    env = {k: v for k, v in wait_run_env.items() if k != "CLAUDE_CODE_SESSION_ID"}
    env["HARNESS_LAUNCHED_AGENT"] = caller
    if session is not None:
        env["HARNESS_LAUNCHED_SESSION_ID"] = session
    if project is not None:
        env["CLAUDE_PROJECT_DIR"] = str(project)
    return env


def _run_child(cmd, env, cwd, *args):
    proc = subprocess.Popen(
        [*cmd, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=env, cwd=str(cwd),
    )
    return _finish(proc, 30)


def _child_setup(tmp_path, project_dir, session_context):
    """Parent file (acceptEdits/high, newer mtime) next to the child's own file
    (bypassPermissions/low, child_project)."""
    child_project = _child_project(tmp_path, project_dir)
    data = tmp_path / "plugin-data"
    parent_file = data / "sessions" / f"{SESSION_ID}.json"
    plant_session_file(
        data, CHILD_SESSION, cwd=str(child_project), project_dir=str(child_project),
        permission_mode="bypassPermissions", effort="low", model="haiku",
    )
    future = time.time() + 100
    os.utime(parent_file, (future, future))
    return child_project


def test_run_agent_child_uses_own_session_context(
    tmp_path, project_dir, session_context, wait_run_env, run_agent_cmd
):
    child_project = _child_setup(tmp_path, project_dir, session_context)
    env = _child_env(wait_run_env, "demo", project=project_dir)
    code, out, err = _run_child(
        run_agent_cmd, env, child_project, "--subagent-type", "effort-agent",
        "--prompt", "ECHO:alpha",
    )
    assert code == 0, (out, err)
    entry = _argv_entries(env)[-1]
    assert _flag(entry["argv"], "--permission-mode") == "bypassPermissions"
    assert Path(entry["cwd"]).resolve() == child_project.resolve()
    assert entry["launched_agent"] == "effort-agent"


def test_run_agent_child_missing_own_session_file_exits_four(
    tmp_path, project_dir, session_context, wait_run_env, run_agent_cmd
):
    child_project = _child_project(tmp_path, project_dir)
    env = _child_env(wait_run_env, "demo", session="no-such-child", project=project_dir)
    code, out, err = _run_child(
        run_agent_cmd, env, child_project, "--subagent-type", "demo", "--prompt", "ECHO:a"
    )
    assert code == 4, (out, err)
    assert out.strip() == ""
    root = Path(env["HARNESS_ARTIFACTS_DIR"])
    assert not root.is_dir() or not list(root.glob("*/record.json"))


def test_run_agent_child_can_spawn_false_refuses(
    tmp_path, project_dir, session_context, wait_run_env, run_agent_cmd
):
    child_project = _child_setup(tmp_path, project_dir, session_context)
    cfg = child_project / ".seretos"
    cfg.mkdir()
    (cfg / "harness.yml").write_text(_CAN_SPAWN.format("false"), encoding="utf-8")
    env = _child_env(wait_run_env, "demo", project=project_dir)
    code, out, err = _run_child(
        run_agent_cmd, env, child_project, "--subagent-type", "effort-agent",
        "--prompt", "ECHO:alpha",
    )
    assert code == 4, (out, err)
    assert out.strip() == ""
    assert "canSpawn" in err
    root = Path(env["HARNESS_ARTIFACTS_DIR"])
    assert not root.is_dir() or not list(root.glob("*/record.json"))
    assert not Path(env["HARNESS_FAKE_ARGV_LOG"]).exists()


@pytest.mark.parametrize(
    "yml",
    ["agents:\n  demo: {}\n", "defaults:\n  model: sonnet\n"],
    ids=["entry-without-canspawn", "defaults-only"],
)
def test_run_agent_child_can_spawn_unset_refuses(
    tmp_path, project_dir, session_context, wait_run_env, run_agent_cmd, yml
):
    """lib-python-harness v0.0.11 apply_config: any entry/defaults without canSpawn: true."""
    child_project = _child_setup(tmp_path, project_dir, session_context)
    cfg = child_project / ".seretos"
    cfg.mkdir()
    (cfg / "harness.yml").write_text(yml, encoding="utf-8")
    env = _child_env(wait_run_env, "demo", project=project_dir)
    code, out, err = _run_child(
        run_agent_cmd, env, child_project, "--subagent-type", "effort-agent",
        "--prompt", "ECHO:alpha",
    )
    assert code == 4, (out, err)
    assert "canSpawn" in err
    assert not Path(env["HARNESS_FAKE_ARGV_LOG"]).exists()


def test_run_agent_child_can_spawn_true_completes(
    tmp_path, project_dir, session_context, wait_run_env, run_agent_cmd
):
    child_project = _child_setup(tmp_path, project_dir, session_context)
    cfg = child_project / ".seretos"
    cfg.mkdir()
    (cfg / "harness.yml").write_text(_CAN_SPAWN.format("true"), encoding="utf-8")
    env = _child_env(wait_run_env, "demo", project=project_dir)
    code, out, err = _run_child(
        run_agent_cmd, env, child_project, "--subagent-type", "effort-agent",
        "--prompt", "ECHO:alpha",
    )
    assert code == 0, (out, err)
    payload = _one_json(out)
    assert payload["state"] == "COMPLETED"
    assert payload["text"] == "alpha"
    record = json.loads(
        (Path(env["HARNESS_ARTIFACTS_DIR"]) / payload["run_id"] / "record.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["state"] == {"__runstate__": "COMPLETED"}


def test_run_agent_injects_launched_session_id(
    session_context, wait_run_env, run_agent_cmd
):
    env = _env(wait_run_env)
    code, out, err = _run_cli(
        run_agent_cmd, env, "--subagent-type", "demo", "--prompt", "ECHO:alpha"
    )
    assert code == 0, (out, err)
    entry = _argv_entries(env)[-1]
    session_id = _flag(entry["argv"], "--session-id")
    assert session_id
    assert entry["launched_session"] == session_id
