"""#78 live: a native `SendMessage` to a harness `agentId` (= run_id) in a real `claude -p`
session is answered by the plugin's hooks module (`hooks/agent_dispatch.ts`) through
`harness send-message`; non-harness recipients stay native.

Deselected by default (`-m live`); same skip contract and helpers as
test_live_native_agent.py. Real-claude results are not claimed by the package that adds
these tests: they are the contract for the live run. The offline proof of the CLI is
test_send_message.py."""
import json
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import pytest
from lib_python_harness import FileRunStore, Harness
from test_live_claude import _tool_result_text
from test_live_native_agent import (
    _agent_calls,
    _build_plugin_dir,
    _claude,
    _flag,
    _records,
    _require_live,
    _result_for,
    _session_env,
    _stream,
)

pytestmark = pytest.mark.live

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _run_id_of(record):
    return record.get("run_id") or record.get("id")


def _send_message_calls(entries):
    return [
        b
        for e in entries
        for b in (e.get("message") or {}).get("content") or []
        if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "SendMessage"
    ]


def _followers(artifacts, of):
    return [r for r in _records(artifacts) if r.get("resumed_from") == of]


@pytest.mark.timeout(700)
def test_live_sendmessage_codeword_recall():
    """The Agent gets a codeword; SendMessage to its agentId asks for it back. The reply
    carries it, a second record has persisted resumed_from == agentId and its `--resume`
    is the origin record's session id. RED today: the native call errors, no 2nd record."""
    _require_live()
    root = Path(tempfile.mkdtemp(prefix="ah78-"))
    try:
        plugin = _build_plugin_dir(root)
        env, project, artifacts = _session_env(root)
        proc, _ = _claude(
            plugin, project, env,
            "Step 1: make one Agent tool call (subagent_type general-purpose) with the prompt "
            "'Remember the codeword ZEBRA42. Reply with exactly OK.' "
            "Step 2: call SendMessage with `to` set to the agentId from step 1's result and "
            "`message` 'What was the codeword? Reply with only the codeword.' "
            "Then reply with the answer. Do nothing else.",
            timeout=660,
        )
        entries = _stream(proc)
        sends = _send_message_calls(entries)
        assert sends, f"the model never called SendMessage:\n{proc.stdout!r}\n{proc.stderr!r}"
        block, _ = _result_for(entries, sends[0]["id"])
        assert block is not None and not block.get("is_error"), block
        assert "ZEBRA42" in _tool_result_text(block).upper()
        records = _records(artifacts)
        assert len(records) == 2, records
        origin = next(r for r in records if not r.get("resumed_from"))
        follower = next(r for r in records if r is not origin)
        assert follower.get("resumed_from") == _run_id_of(origin)
        assert _flag(follower["argv"], "--resume") == origin["session_id"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.timeout(900)
def test_live_sendmessage_second_message_recall():
    """A second SendMessage to the ORIGINAL agentId reaches the newest run's session, so
    it recalls a word given only in the first follow-up; a third record is chained to F1."""
    _require_live()
    root = Path(tempfile.mkdtemp(prefix="ah78-"))
    try:
        plugin = _build_plugin_dir(root)
        env, project, artifacts = _session_env(root)
        proc, _ = _claude(
            plugin, project, env,
            "Step 1: make one Agent tool call (subagent_type general-purpose) with the prompt "
            "'Reply with exactly OK.' "
            "Step 2: call SendMessage with `to` = the agentId from step 1 and `message` "
            "'Remember the second word MANGO77. Reply with exactly OK.' "
            "Step 3: call SendMessage again with `to` = the SAME ORIGINAL agentId from step 1 "
            "and `message` 'What was the second word? Reply with only that word.' "
            "Then reply with the answer. Do nothing else.",
            timeout=860,
        )
        entries = _stream(proc)
        sends = _send_message_calls(entries)
        assert len(sends) >= 2, f"expected two SendMessage calls:\n{proc.stdout!r}"
        block, _ = _result_for(entries, sends[1]["id"])
        assert block is not None and not block.get("is_error"), block
        assert "MANGO77" in _tool_result_text(block).upper()
        records = _records(artifacts)
        assert len(records) == 3, records
        origin = next(r for r in records if not r.get("resumed_from"))
        f1 = next(r for r in records if r.get("resumed_from") == _run_id_of(origin))
        assert any(r.get("resumed_from") == _run_id_of(f1) for r in records), records
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.timeout(900)
def test_live_sendmessage_native_passthrough():
    """One session sends to the harness agentId (routed; the genuine RED today), then to
    `main`, teammate `researcher` and `a1b2c3d4` (never routed to the harness): those carry
    no harness error text and exactly one follow-up record exists."""
    _require_live()
    root = Path(tempfile.mkdtemp(prefix="ah78-"))
    try:
        plugin = _build_plugin_dir(root)
        env, project, artifacts = _session_env(root)
        proc, _ = _claude(
            plugin, project, env,
            "Step 1: make one Agent tool call (subagent_type general-purpose) with the prompt "
            "'Reply with exactly OK.' "
            "Step 2: call SendMessage with `to` = the agentId from step 1, `message` 'ping'. "
            "Step 3: call SendMessage with `to` 'main', `message` 'ping'. "
            "Step 4: call SendMessage with `to` 'researcher', `message` 'ping'. "
            "Step 5: call SendMessage with `to` 'a1b2c3d4', `message` 'ping'. "
            "Make all calls even if some fail. Then reply DONE.",
            timeout=860,
        )
        entries = _stream(proc)
        sends = _send_message_calls(entries)
        assert len(sends) >= 4, f"expected four SendMessage calls:\n{proc.stdout!r}"
        first, _ = _result_for(entries, sends[0]["id"])
        assert first is not None and not first.get("is_error"), first
        for call in sends[1:4]:
            block, _ = _result_for(entries, call["id"])
            text = _tool_result_text(block) if block else ""
            assert "unknown run_id" not in text, text
            assert "could not run" not in text, text
        origin = next(r for r in _records(artifacts) if not r.get("resumed_from"))
        assert len(_followers(artifacts, _run_id_of(origin))) == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.timeout(400)
def test_live_native_agentid_not_uuid():
    """Premise check (may pass at once): in a session WITHOUT this plugin a native Agent
    call's agentId is not a canonical UUID, so UUID-shaped routing never captures a
    native subagent."""
    _require_live()
    root = Path(tempfile.mkdtemp(prefix="ah78-"))
    try:
        plugin = root / "empty-plugin"
        (plugin / ".claude-plugin").mkdir(parents=True)
        (plugin / ".claude-plugin" / "plugin.json").write_text(
            json.dumps({"name": "empty-plugin", "version": "0.0.0"}), encoding="utf-8"
        )
        env, project, _ = _session_env(root)
        proc, _ = _claude(
            plugin, project, env,
            "Use the Agent tool with subagent_type general-purpose and the prompt "
            "'Reply OK.' Do nothing else.",
            timeout=380,
        )
        entries = _stream(proc)
        calls = _agent_calls(entries)
        assert calls, f"the model never made a native Agent call:\n{proc.stdout!r}"
        block, entry = _result_for(entries, calls[0]["id"])
        assert block is not None, proc.stdout
        structured = entry.get("tool_use_result")
        agent_id = structured.get("agentId") if isinstance(structured, dict) else None
        if agent_id is None:
            found = re.findall(r"agentId\W{1,4}([A-Za-z0-9_-]+)", _tool_result_text(block))
            agent_id = found[0] if found else None
        assert agent_id, "no agentId in the native result"
        assert not _UUID_RE.match(str(agent_id)), agent_id
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.timeout(500)
def test_live_sendmessage_running_or_unknown_errors():
    """SendMessage to a RUNNING harness run and to a fresh uuid4 both error, naming
    `RUNNING` and `unknown run_id`."""
    _require_live()
    root = Path(tempfile.mkdtemp(prefix="ah78-"))
    try:
        plugin = _build_plugin_dir(root)
        env, project, artifacts = _session_env(root)
        fake = Path(__file__).parent / "fixtures" / "fake_claude.py"
        harness = Harness(store=FileRunStore(artifacts), claude_argv=[sys.executable, str(fake)])
        started = harness.start(prompt="SLEEP:300", cwd=str(project))
        run_id = str(started.run_id)
        unknown = str(uuid.uuid4())
        try:
            proc, _ = _claude(
                plugin, project, env,
                f"Call SendMessage with `to` '{run_id}' and `message` 'hi'. Then call "
                f"SendMessage with `to` '{unknown}' and `message` 'hi'. Make both calls even "
                "if they fail. Then reply DONE.",
                timeout=460,
            )
        finally:
            harness.stop(run_id)
        entries = _stream(proc)
        sends = _send_message_calls(entries)
        assert len(sends) >= 2, f"expected two SendMessage calls:\n{proc.stdout!r}"
        b1, _ = _result_for(entries, sends[0]["id"])
        b2, _ = _result_for(entries, sends[1]["id"])
        assert b1 is not None and b1.get("is_error") and "RUNNING" in _tool_result_text(b1)
        assert b2 is not None and b2.get("is_error") and "unknown run_id" in _tool_result_text(b2)
    finally:
        shutil.rmtree(root, ignore_errors=True)
