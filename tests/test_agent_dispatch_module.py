"""#82: the hooks module `hooks/agent_dispatch.ts` itself, run under Node against the real
`harness run-agent` / `send-message` CLIs (fake claude behind them).

A fake engine `$` (tests/agent_dispatch_driver.mjs) mirrors Claude Code's contract:
`$.process.spawn` takes ONE `{argv}` object and streams `{stream, text}` chunks, and a
`tool.call` answer is validated against the tool's output shape -- anything else is the
ticket's "result that does not match its output shape" failure. Errors must be `{deny}`.

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
from conftest import FAKE_CLAUDE, REPO
from lib_python_harness import FileRunStore, Harness
from test_run_agent import _env, _one_json, _only_run_id
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


def _driver_popen(module_mts, tool, event, env, prefix):
    return subprocess.Popen(
        [_node(), "--experimental-strip-types", "--no-warnings", str(DRIVER), str(module_mts), tool, json.dumps(event)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**env, "HARNESS_DRIVER_CMD": json.dumps(prefix)},
    )


def _answer(proc, timeout=DRIVER_TIMEOUT):
    out, err = proc.communicate(timeout=timeout)
    assert proc.returncode == 0, (out, err)
    return _one_json(out)


def _call(module_mts, tool, event, env, prefix):
    return _answer(_driver_popen(module_mts, tool, event, env, prefix))


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


def test_agent_completed_matches_native_shape(session_context, wait_run_env, run_agent_cmd, module_mts):
    env = _env(wait_run_env)
    ans = _call(module_mts, "Agent", _agent_event(), env, _prefix(run_agent_cmd))
    _assert_agent_output(ans)
    assert "result" in ans, ans
    r = ans["result"]
    assert r["content"][0]["text"] == "alpha"
    assert r["prompt"] == "ECHO:alpha"
    assert r["agentId"] == _only_run_id(env)
    assert UUID.match(r["agentId"])
    assert r["totalDurationMs"] >= 0 and r["totalTokens"] >= 0


def test_agent_alias_task_is_answered_too(session_context, wait_run_env, run_agent_cmd, module_mts):
    env = _env(wait_run_env)
    ans = _call(module_mts, "Task", _agent_event("ECHO:gamma"), env, _prefix(run_agent_cmd))
    _assert_agent_output(ans)
    assert ans["result"]["content"][0]["text"] == "gamma"


def test_agent_failed_is_deny(session_context, wait_run_env, run_agent_cmd, module_mts):
    env = _env(wait_run_env)
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
def test_agent_cancelled_is_deny(session_context, wait_run_env, run_agent_cmd, module_mts):
    env = _env(wait_run_env)
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


def test_agent_unknown_agent_is_deny_with_note(session_context, wait_run_env, run_agent_cmd, module_mts):
    env = _env(wait_run_env)
    ev = {**_agent_event(), "subagent_type": "nope"}
    ans = _call(module_mts, "Agent", ev, env, _prefix(run_agent_cmd))
    _assert_deny(ans)
    assert BUILTIN_RESULT_NOTE in ans["deny"]
    assert "unknown agent" in ans["deny"]


def test_agent_missing_prompt_is_deny(session_context, wait_run_env, run_agent_cmd, module_mts):
    env = _env(wait_run_env)
    ans = _call(module_mts, "Agent", {"subagent_type": "demo"}, env, _prefix(run_agent_cmd))
    _assert_deny(ans)
    assert "no prompt" in ans["deny"]


def test_send_message_matches_native_shape(
    session_context, wait_run_env, run_agent_cmd, send_message_cmd, module_mts
):
    env = _env(wait_run_env)
    # Seed the origin run through the CLI so this test isolates the SendMessage answer.
    origin = _origin(run_agent_cmd, env, prompt="ECHO:a")
    ans = _call(
        module_mts, "SendMessage",
        {"to": origin, "message": "ECHO:b"},
        env, _prefix(send_message_cmd),
    )
    _assert_send_output(ans)
    assert ans == {"result": {"success": True, "message": "b"}}


def test_send_message_non_uuid_goes_to_next(session_context, wait_run_env, send_message_cmd, module_mts):
    env = _env(wait_run_env)
    ans = _call(module_mts, "SendMessage", {"to": "teammate", "message": "hi"}, env, _prefix(send_message_cmd))
    assert ans == {"passedToNext": True}
