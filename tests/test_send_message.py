"""#78: `harness send-message` resumes a finished run -- the CLI behind a native
`SendMessage` to a harness `agentId` (= run_id). Modelled on test_run_agent.py; the hooks
module's routing is proven only by the `live` tests (test_live_native_agent.py)."""
import json
import time
import uuid
from pathlib import Path

from conftest import SESSION_ID  # noqa: F401  (session_context fixture plants it)
from test_run_agent import (
    _argv_entries,
    _env,
    _flag,
    _one_json,
    _only_run_id,
    _run_cli,
    _spawn,
)


def _records_by_id(env):
    root = Path(env["HARNESS_ARTIFACTS_DIR"])
    out = {}
    if root.is_dir():
        for p in root.glob("*/record.json"):
            out[p.parent.name] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _origin(run_agent_cmd, env, prompt="ECHO:alpha", *extra):
    code, out, err = _run_cli(
        run_agent_cmd, env, "--subagent-type", "demo", "--prompt", prompt, *extra, timeout=60
    )
    assert code in (0, 1), (out, err)
    return _one_json(out)["run_id"]


def _send(cmd, env, to, message, timeout=30):
    return _run_cli(cmd, env, "--to", to, "--message", message, timeout=timeout)


def _resume_flags(env):
    return [_flag(e["argv"], "--resume") for e in _argv_entries(env)]


def test_send_message_resumes_finished_run(
    session_context, wait_run_env, run_agent_cmd, send_message_cmd
):
    env = _env(wait_run_env)
    origin = _origin(run_agent_cmd, env)
    origin_session = _records_by_id(env)[origin]["session_id"]
    code, out, err = _send(send_message_cmd, env, origin, "ECHO:beta")
    assert code == 0, (out, err)
    payload = _one_json(out)
    assert payload["text"] == "beta"
    assert payload["run_id"] != origin
    assert _records_by_id(env)[payload["run_id"]].get("resumed_from") == origin
    assert _resume_flags(env)[-1] == origin_session
    # Additional coverage: the reply object names the origin too.
    assert payload.get("resumed_from") == origin


def test_send_message_resumes_failed_run(
    session_context, wait_run_env, run_agent_cmd, send_message_cmd
):
    env = _env(wait_run_env)
    origin = _origin(run_agent_cmd, env, prompt="NO_RESULT")
    code, out, err = _send(send_message_cmd, env, origin, "ECHO:again")
    assert code == 0, (out, err)
    assert _one_json(out)["text"] == "again"


def test_send_message_empty_message_exits_four(
    session_context, wait_run_env, run_agent_cmd, send_message_cmd
):
    env = _env(wait_run_env)
    origin = _origin(run_agent_cmd, env)
    before = set(_records_by_id(env))
    code, out, err = _send(send_message_cmd, env, origin, "")
    assert code == 4, (out, err)
    assert out.strip() == ""
    assert set(_records_by_id(env)) == before


def test_send_message_continues_newest_in_chain(
    session_context, wait_run_env, run_agent_cmd, send_message_cmd
):
    env = _env(wait_run_env, HARNESS_FAKE_FORK_ON_RESUME="1")
    origin = _origin(run_agent_cmd, env)
    origin_session = _records_by_id(env)[origin]["session_id"]
    code, out, err = _send(send_message_cmd, env, origin, "ECHO:one")
    assert code == 0, (out, err)
    f1 = _one_json(out)["run_id"]
    f1_session = _records_by_id(env)[f1]["session_id"]
    assert f1_session != origin_session  # the fork switch is in effect

    code, out, err = _send(send_message_cmd, env, origin, "ECHO:two")
    assert code == 0, (out, err)
    f2 = _one_json(out)["run_id"]
    assert _resume_flags(env)[-1] == f1_session
    assert _records_by_id(env)[f2].get("resumed_from") == f1

    # Additional coverage: addressing F1 by its own id reaches the newest (F2) session.
    f2_session = _records_by_id(env)[f2]["session_id"]
    code, out, err = _send(send_message_cmd, env, f1, "ECHO:three")
    assert code == 0, (out, err)
    assert _resume_flags(env)[-1] == f2_session


def test_send_message_chains_do_not_cross(
    session_context, wait_run_env, run_agent_cmd, send_message_cmd
):
    env = _env(wait_run_env, HARNESS_FAKE_FORK_ON_RESUME="1")
    a = _origin(run_agent_cmd, env, prompt="ECHO:a")
    b = _origin(run_agent_cmd, env, prompt="ECHO:b")
    b_session = _records_by_id(env)[b]["session_id"]
    code, out, err = _send(send_message_cmd, env, a, "ECHO:x")
    assert code == 0, (out, err)
    code, out, err = _send(send_message_cmd, env, b, "ECHO:y")
    assert code == 0, (out, err)
    assert _resume_flags(env)[-1] == b_session
    assert _records_by_id(env)[_one_json(out)["run_id"]].get("resumed_from") == b


def test_send_message_running_or_unknown_exits_four(
    session_context, wait_run_env, run_agent_cmd, send_message_cmd
):
    env = _env(wait_run_env)
    code, out, err = _send(send_message_cmd, env, str(uuid.uuid4()), "ECHO:x", timeout=30)
    assert code == 4, (out, err)
    assert out.strip() == ""
    assert "unknown run_id" in err
    assert not _records_by_id(env)

    proc = _spawn(run_agent_cmd, env, "--subagent-type", "demo", "--prompt", "SLEEP:30")
    try:
        deadline = time.monotonic() + 20
        run_id = None
        while time.monotonic() < deadline and run_id is None:
            try:
                run_id = _only_run_id(env)
            except (FileNotFoundError, AssertionError):
                time.sleep(0.2)
        assert run_id, "run-agent never started a run"
        time.sleep(1)
        code, out, err = _send(send_message_cmd, env, run_id, "ECHO:x", timeout=30)
        assert code == 4, (out, err)
        assert out.strip() == ""
        assert "RUNNING" in err
        assert set(_records_by_id(env)) == {run_id}
    finally:
        proc.kill()
        proc.communicate()


def test_send_message_running_follower_exits_four(
    session_context, wait_run_env, run_agent_cmd, send_message_cmd
):
    env = _env(wait_run_env, HARNESS_FAKE_FORK_ON_RESUME="1")
    origin = _origin(run_agent_cmd, env)
    follower = _spawn(send_message_cmd, env, "--to", origin, "--message", "SLEEP:30")
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and len(_records_by_id(env)) < 2:
            time.sleep(0.2)
        assert len(_records_by_id(env)) == 2, "follower never started"
        time.sleep(1)
        code, out, err = _send(send_message_cmd, env, origin, "ECHO:x", timeout=30)
        assert code == 4, (out, err)
        assert "RUNNING" in err
        assert len(_records_by_id(env)) == 2
    finally:
        follower.kill()
        follower.communicate()


def test_send_message_replays_origin_isolation(
    session_context, wait_run_env, run_agent_cmd, send_message_cmd
):
    env = _env(wait_run_env)
    origin = _origin(run_agent_cmd, env, "ECHO:alpha", "--model", "sonnet")
    code, out, err = _send(send_message_cmd, env, origin, "ECHO:beta")
    assert code == 0, (out, err)
    first, second = _argv_entries(env)[-2:]
    for flag in ("--agent", "--agents", "--permission-mode", "--effort", "--model"):
        assert _flag(second["argv"], flag) == _flag(first["argv"], flag), flag
    assert _flag(second["argv"], "--model") == "sonnet"
    assert second["cwd"] == first["cwd"]
    assert second["launched_agent"] == first["launched_agent"] == "demo"
