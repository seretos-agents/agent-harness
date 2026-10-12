"""#82: the hooks module `hooks/agent_dispatch.ts` itself, run under Node against the real
`harness run-agent` / `send-message` CLIs (fake claude behind them).

A fake engine `$` (tests/agent_dispatch_driver.mjs) mirrors Claude Code's contract:
`$.process.spawn` takes ONE `{argv}` object and streams `{stream, text}` chunks, and a
`tool.call` answer is validated against the tool's output shape -- anything else is the
ticket's "result that does not match its output shape" failure. Errors must be `{deny}`.

#84: the module's child sees the engine's HOST env, not the hook env. The tests build the two
separately (`_host_env` / `_run_hook`), write the session file by running the real
`harness hook`, and the driver passes the child exactly the host env it is given.

Needs `node` >= 22.6 (type stripping); skipped without it, but a FAILURE when `CI` is set,
so CI cannot silently skip these."""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import FAKE_CLAUDE, REPO, SESSION_ID, _base_env
from lib_python_harness import FileRunStore, Harness
from test_run_agent import _argv_entries, _flag, _one_json, _only_run_id, _stored_result
from test_send_message import _origin

DRIVER = Path(__file__).parent / "agent_dispatch_driver.mjs"
MODULE = REPO / "hooks" / "agent_dispatch.ts"
BUILTIN_RESULT_NOTE = "agent-harness could not run Agent through the harness"
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
DRIVER_TIMEOUT = 120


def _node() -> str:
    node = shutil.which("node")
    problem = None
    if node is None:
        problem = "node is not on PATH"
    else:
        out = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
        major, minor = (int(x) for x in out.lstrip("v").split(".")[:2])
        if (major, minor) < (22, 6):
            problem = f"node {out} is older than 22.6 (type stripping)"
    if problem:
        if os.environ.get("CI"):
            pytest.fail(f"{problem}; CI must run the module tests")
        pytest.skip(problem)
    return node


@pytest.fixture
def module_mts(tmp_path) -> Path:
    """The shipped module copied to .mts so Node treats it as ESM TypeScript."""
    dest = tmp_path / "agent_dispatch.mts"
    shutil.copyfile(MODULE, dest)
    return dest


def _os_base() -> dict:
    """The OS environment minus every CLAUDE_*/HARNESS_* key (keeps PATH/SYSTEMROOT)."""
    return {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "HARNESS_"))}


def _host_env(tmp_path, **extra) -> dict:
    """What the engine's host process hands a `$.process.spawn` child (#84): the session id,
    HOME, the test knobs -- and NONE of CLAUDE_PLUGIN_DATA / CLAUDE_PLUGIN_ROOT / CLAUDE_PROJECT_DIR
    unless a test passes them. A None value in `extra` removes the key."""
    base = _base_env(tmp_path)
    home = base["HOME"]
    env = {
        **_os_base(),
        "HARNESS_CLAUDE_ARGV": base["HARNESS_CLAUDE_ARGV"],
        "HARNESS_ARTIFACTS_DIR": base["HARNESS_ARTIFACTS_DIR"],
        "HARNESS_FAKE_ARGV_LOG": base["HARNESS_FAKE_ARGV_LOG"],
        "CLAUDE_CONFIG_DIR": base["CLAUDE_CONFIG_DIR"],
        "HOME": home,
        "USERPROFILE": home,
        "CLAUDE_CODE_SESSION_ID": SESSION_ID,
    }
    env.update(extra)
    return {k: v for k, v in env.items() if v is not None}


def _hook_cmd() -> list:
    binary = os.environ.get("HARNESS_BIN")
    return [binary, "hook"] if binary else [sys.executable, "-m", "harness_plugin", "hook"]


def _run_hook(tmp_path, payload, project, data_name="plugin-data"):
    """The REAL `harness hook` under the HOOK env (per-plugin variables set), same HOME."""
    home = str(tmp_path / "home")
    env = {
        **_os_base(),
        "CLAUDE_PLUGIN_DATA": str(tmp_path / data_name),
        "CLAUDE_PROJECT_DIR": str(project),
        "HOME": home,
        "USERPROFILE": home,
    }
    proc = subprocess.run(
        _hook_cmd(), input=json.dumps(payload), capture_output=True, text=True, env=env, timeout=60
    )
    assert proc.returncode == 0, proc.stderr


def _prompt_payload(session_id, cwd, mode):
    return {
        "hook_event_name": "UserPromptSubmit", "session_id": session_id, "cwd": str(cwd),
        "permission_mode": mode, "model": "opus", "effort": "high",
    }


@pytest.fixture
def hooked_session(tmp_path, project_dir):
    """SessionStart then UserPromptSubmit for SESSION_ID through the real hook."""
    _run_hook(tmp_path, {"hook_event_name": "SessionStart", "session_id": SESSION_ID, "cwd": str(project_dir)}, project_dir)
    _run_hook(tmp_path, _prompt_payload(SESSION_ID, project_dir, "bypassPermissions"), project_dir)


def _driver_popen(module_mts, tool, event, env, prefix, session_id=SESSION_ID, cwd=None):
    """`env` is the HOST env handed to the module's spawn children; the node process itself
    runs under the bare OS base so nothing of the pytest env can reach them."""
    driver_env = {**_os_base(), "HARNESS_DRIVER_CMD": json.dumps(prefix), "HARNESS_DRIVER_HOST_ENV": json.dumps(env)}
    if session_id is not None:
        driver_env["HARNESS_DRIVER_SESSION_ID"] = session_id
    return subprocess.Popen(
        [_node(), "--experimental-strip-types", "--no-warnings", str(DRIVER), str(module_mts), tool, json.dumps(event)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=driver_env, cwd=cwd,
    )


def _answer(proc, timeout=DRIVER_TIMEOUT):
    out, err = proc.communicate(timeout=timeout)
    assert proc.returncode == 0, (out, err)
    return _one_json(out)


def _call(module_mts, tool, event, env, prefix, **kw):
    return _answer(_driver_popen(module_mts, tool, event, env, prefix, **kw))


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _assert_deny(ans):
    assert set(ans) == {"deny"}, ans
    assert isinstance(ans["deny"], str) and ans["deny"] != "", ans


def _assert_agent_output(ans):
    """Claude Code 2.1.294 native Agent `completed` output shape (or a deny)."""
    if "deny" in ans:
        return _assert_deny(ans)
    assert set(ans) == {"result"}, ans
    r = ans["result"]
    assert r.get("status") == "completed", ans
    assert isinstance(r.get("prompt"), str), ans
    assert isinstance(r.get("agentId"), str) and r["agentId"], ans
    content = r.get("content")
    assert isinstance(content, list) and content, ans
    assert all(c.get("type") == "text" and isinstance(c.get("text"), str) for c in content), ans
    for key in ("totalToolUseCount", "totalDurationMs", "totalTokens"):
        assert _is_num(r.get(key)), (key, ans)
    u = r.get("usage")
    assert isinstance(u, dict), ans
    assert _is_num(u.get("input_tokens")) and _is_num(u.get("output_tokens")), ans
    for key in ("cache_creation_input_tokens", "cache_read_input_tokens"):
        assert key in u and (u[key] is None or _is_num(u[key])), (key, ans)
    for key, subkeys in (
        ("server_tool_use", ("web_search_requests", "web_fetch_requests")),
        ("cache_creation", ("ephemeral_1h_input_tokens", "ephemeral_5m_input_tokens")),
    ):
        assert key in u, (key, ans)
        assert u[key] is None or all(_is_num(u[key].get(s)) for s in subkeys), (key, ans)
    assert "service_tier" in u and (u["service_tier"] is None or isinstance(u["service_tier"], str)), ans


def _assert_send_output(ans):
    if "deny" in ans:
        return _assert_deny(ans)
    assert set(ans) == {"result"}, ans
    r = ans["result"]
    assert r.get("success") is True and isinstance(r.get("message"), str), ans


def _agent_event(prompt="ECHO:alpha", **extra):
    return {"subagent_type": "demo", "prompt": prompt, "description": "d", **extra}


def _prefix(cmd):
    return cmd[:-1]


def test_agent_completed_matches_native_shape(hooked_session, tmp_path, run_agent_cmd, module_mts):
    env = _host_env(tmp_path)
    ans = _call(module_mts, "Agent", _agent_event(), env, _prefix(run_agent_cmd))
    _assert_agent_output(ans)
    assert "result" in ans, ans
    r = ans["result"]
    assert r["content"][0]["text"] == "alpha"
    assert r["prompt"] == "ECHO:alpha"
    assert r["agentId"] == _only_run_id(env)
    assert UUID.match(r["agentId"])
    assert r["totalDurationMs"] >= 0 and r["totalTokens"] >= 0


def test_agent_totals_are_computed_from_the_run_record(hooked_session, tmp_path, run_agent_cmd, module_mts):
    env = _host_env(tmp_path)
    ans = _call(module_mts, "Agent", _agent_event("ECHO:alpha SLEEP:1 USAGE"), env, _prefix(run_agent_cmd))
    _assert_agent_output(ans)
    r = ans["result"]
    stored = _stored_result(env, r["agentId"])
    # Exact values from the fake claude's KNOWN_USAGE: null (cache_read) counts as 0.
    assert r["usage"]["input_tokens"] == 11 and r["usage"]["output_tokens"] == 22
    assert r["usage"]["cache_creation_input_tokens"] == 330
    assert r["usage"]["cache_read_input_tokens"] is None
    assert r["totalTokens"] == 11 + 22 + 330
    assert r["totalDurationMs"] == round(stored.duration_s * 1000)
    assert r["totalDurationMs"] >= 1000
    assert r["totalToolUseCount"] == 0


def test_agent_alias_task_is_answered_too(hooked_session, tmp_path, run_agent_cmd, module_mts):
    env = _host_env(tmp_path)
    ans = _call(module_mts, "Task", _agent_event("ECHO:gamma"), env, _prefix(run_agent_cmd))
    _assert_agent_output(ans)
    assert ans["result"]["content"][0]["text"] == "gamma"


def test_agent_failed_is_deny(hooked_session, tmp_path, run_agent_cmd, module_mts):
    env = _host_env(tmp_path)
    ans = _call(module_mts, "Agent", _agent_event("NO_RESULT"), env, _prefix(run_agent_cmd))
    _assert_deny(ans)
    assert _only_run_id(env) in ans["deny"]
    assert "FAILED" in ans["deny"]


# Upstream limit, same reason as test_run_agent.test_run_agent_cancelled_exits_three. On
# win32 this case is therefore SKIPPED (R6's "PASSED, not SKIPPED" covers the other tests).
@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "lib-python-harness v0.0.9+: Harness.stop() racing wait_for does not reliably "
        "yield CANCELLED on Windows; covered on Linux"
    ),
)
def test_agent_cancelled_is_deny(hooked_session, tmp_path, run_agent_cmd, module_mts):
    env = _host_env(tmp_path)
    proc = _driver_popen(module_mts, "Agent", _agent_event("SLEEP:30"), env, _prefix(run_agent_cmd))
    deadline = time.monotonic() + 30
    run_id = None
    while time.monotonic() < deadline and run_id is None:
        try:
            run_id = _only_run_id(env)
        except (FileNotFoundError, AssertionError):
            time.sleep(0.2)
    assert run_id, "the module never started a run"
    time.sleep(1)
    Harness(
        store=FileRunStore(env["HARNESS_ARTIFACTS_DIR"]),
        claude_argv=[sys.executable, str(FAKE_CLAUDE)],
    ).stop(run_id)
    ans = _answer(proc)
    _assert_deny(ans)
    assert run_id in ans["deny"] and "CANCELLED" in ans["deny"]


def test_agent_unknown_agent_is_deny_with_note(hooked_session, tmp_path, run_agent_cmd, module_mts):
    env = _host_env(tmp_path)
    ev = {**_agent_event(), "subagent_type": "nope"}
    ans = _call(module_mts, "Agent", ev, env, _prefix(run_agent_cmd))
    _assert_deny(ans)
    assert BUILTIN_RESULT_NOTE in ans["deny"]
    assert "unknown agent" in ans["deny"]


def test_agent_missing_prompt_is_deny(hooked_session, tmp_path, run_agent_cmd, module_mts):
    env = _host_env(tmp_path)
    ans = _call(module_mts, "Agent", {"subagent_type": "demo"}, env, _prefix(run_agent_cmd))
    _assert_deny(ans)
    assert "no prompt" in ans["deny"]


def test_send_message_matches_native_shape(
    hooked_session, tmp_path, run_agent_cmd, send_message_cmd, module_mts
):
    env = _host_env(tmp_path)
    # Seed the origin run through the CLI so this test isolates the SendMessage answer.
    origin = _origin(run_agent_cmd, env, prompt="ECHO:a")
    ans = _call(
        module_mts, "SendMessage",
        {"to": origin, "message": "ECHO:b"},
        env, _prefix(send_message_cmd),
    )
    _assert_send_output(ans)
    assert ans == {"result": {"success": True, "message": "b"}}
    # The message is the resumed run's own answer, not a constant: a different echo differs.
    ans2 = _call(
        module_mts, "SendMessage",
        {"to": origin, "message": "ECHO:zulu"},
        env, _prefix(send_message_cmd),
    )
    assert ans2 == {"result": {"success": True, "message": "zulu"}}


def test_send_message_non_uuid_goes_to_next(hooked_session, tmp_path, send_message_cmd, module_mts):
    env = _host_env(tmp_path)
    ans = _call(module_mts, "SendMessage", {"to": "teammate", "message": "hi"}, env, _prefix(send_message_cmd))
    assert ans == {"passedToNext": True}


# --- #84: host env vs hook env ---------------------------------------------------------


def test_agent_host_env_without_plugin_data_completes(hooked_session, tmp_path, run_agent_cmd, module_mts):
    env = _host_env(tmp_path)
    assert not {"CLAUDE_PLUGIN_DATA", "CLAUDE_PLUGIN_ROOT", "CLAUDE_PROJECT_DIR"} & set(env)
    ans = _call(module_mts, "Agent", _agent_event(), env, _prefix(run_agent_cmd))
    _assert_agent_output(ans)
    assert "result" in ans, ans
    assert ans["result"]["content"][0]["text"] == "alpha"
    assert _stored_result(env, ans["result"]["agentId"]).state.name == "COMPLETED"


def test_agent_foreign_plugin_data_completes(hooked_session, tmp_path, run_agent_cmd, module_mts):
    foreign = tmp_path / "foreign-plugin-data"
    foreign.mkdir()
    env = _host_env(tmp_path, CLAUDE_PLUGIN_DATA=str(foreign))
    ans = _call(module_mts, "Agent", _agent_event(), env, _prefix(run_agent_cmd))
    _assert_agent_output(ans)
    assert "result" in ans, ans
    assert _stored_result(env, ans["result"]["agentId"]).state.name == "COMPLETED"


def test_agent_session_id_comes_from_module(hooked_session, tmp_path, run_agent_cmd, module_mts):
    """Even without CLAUDE_CODE_SESSION_ID in the host env, `$.session.id()` names the parent."""
    env = _host_env(tmp_path, CLAUDE_CODE_SESSION_ID=None)
    assert "CLAUDE_CODE_SESSION_ID" not in env
    ans = _call(module_mts, "Agent", _agent_event(), env, _prefix(run_agent_cmd), session_id=SESSION_ID)
    _assert_agent_output(ans)
    assert "result" in ans, ans
    assert _flag(_argv_entries(env)[-1]["argv"], "--permission-mode") == "bypassPermissions"


def test_agent_without_session_file_is_denied(tmp_path, run_agent_cmd, module_mts):
    env = _host_env(tmp_path)
    ans = _call(module_mts, "Agent", _agent_event(), env, _prefix(run_agent_cmd))
    _assert_deny(ans)
    assert "cannot determine the parent session's context" in ans["deny"]
    assert str(tmp_path / "home" / ".agent-harness" / "sessions") in ans["deny"]


def test_agent_in_harness_child_uses_own_session(tmp_path, project_dir, run_agent_cmd, module_mts):
    """#77 through the module: the launched id is authoritative; the parent's newer file and a
    leaked CLAUDE_PROJECT_DIR must not win."""
    child_project = tmp_path / "child-project"
    shutil.copytree(project_dir, child_project)
    (child_project / ".git").mkdir()
    (child_project / ".seretos").mkdir()
    (child_project / ".seretos" / "harness.yml").write_text(
        "agents:\n  demo:\n    canSpawn: true\n", encoding="utf-8"
    )
    _run_hook(tmp_path, _prompt_payload(SESSION_ID, project_dir, "acceptEdits"), project_dir)
    _run_hook(
        tmp_path, _prompt_payload("child-session", child_project, "bypassPermissions"), child_project,
        data_name="child-plugin-data",
    )
    parent_file = tmp_path / "home" / ".agent-harness" / "sessions" / f"{SESSION_ID}.json"
    assert parent_file.is_file(), "the hook wrote no home mirror of the parent session"
    future = time.time() + 100
    os.utime(parent_file, (future, future))
    env = _host_env(
        tmp_path, CLAUDE_CODE_SESSION_ID=None, HARNESS_LAUNCHED_SESSION_ID="child-session",
        HARNESS_LAUNCHED_AGENT="demo", CLAUDE_PROJECT_DIR=str(project_dir),
    )
    ans = _call(
        module_mts, "Agent", _agent_event(), env, _prefix(run_agent_cmd),
        session_id="child-session", cwd=str(child_project),
    )
    _assert_agent_output(ans)
    assert "result" in ans, ans
    assert _flag(_argv_entries(env)[-1]["argv"], "--permission-mode") == "bypassPermissions"
