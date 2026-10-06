"""#76 live: a native `Agent` call in a real `claude -p` session is answered by the
plugin's hooks module (`hooks/agent_dispatch.ts`) through `harness run-agent`.

Deselected by default (`-m live`). Each test skips without `claude` on PATH, real
credentials, or `HARNESS_BIN` (the frozen binary the module spawns). The module itself
is proven here only: the PR suite proves `run-agent` offline (test_run_agent.py).
Minimum Claude Code: 2.1.291 (Function Hooks early access)."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from test_live_claude import _real_credentials_path, _tool_result_text

pytestmark = pytest.mark.live

REPO = Path(__file__).resolve().parents[1]
DENY_MARKER = "harness_start_agent"  # the #52 classic deny names this tool; a real answer never does


def _require_live():
    if shutil.which("claude") is None:
        pytest.skip("the real `claude` CLI is not on PATH")
    if not _real_credentials_path().is_file():
        pytest.skip(f"no real credentials at {_real_credentials_path()}")
    if not os.environ.get("HARNESS_BIN"):
        pytest.skip("HARNESS_BIN is not set (frozen binary the hooks module spawns)")


def _build_plugin_dir(root: Path, *, with_binaries: bool = True) -> Path:
    """A plugin dir shaped like the release tree: `.claude-plugin`, `hooks/` (hooks.json
    + the module), `agents/`, `skills/`, the committed `bin/harness` dispatcher and
    HARNESS_BIN under its platform name."""
    plugin = root / "plugin"
    for name in (".claude-plugin", "hooks", "agents", "skills"):
        if (REPO / name).is_dir():
            shutil.copytree(REPO / name, plugin / name)
    (plugin / "bin").mkdir(parents=True, exist_ok=True)
    if with_binaries:
        shutil.copy2(REPO / "bin" / "harness", plugin / "bin" / "harness")
        target = "harness.exe" if sys.platform == "win32" else "harness-linux"
        shutil.copy2(os.environ["HARNESS_BIN"], plugin / "bin" / target)
        if sys.platform != "win32":
            (plugin / "bin" / target).chmod(0o755)
            (plugin / "bin" / "harness").chmod(0o755)
    return plugin


def _session_env(root: Path) -> tuple[dict, Path, Path]:
    config_dir = root / "claude-config"
    project = root / "live-project"
    artifacts = root / "artifacts"
    for d in (config_dir, project):
        d.mkdir(parents=True, exist_ok=True)
    shutil.copy(_real_credentials_path(), config_dir / ".credentials.json")
    env = {
        **os.environ,
        "CLAUDE_CONFIG_DIR": str(config_dir),
        "HARNESS_ARTIFACTS_DIR": str(artifacts),
    }
    return env, project, artifacts


def _claude(plugin: Path, project: Path, env: dict, prompt: str, *, timeout: float):
    started = time.monotonic()
    proc = subprocess.run(
        [
            "claude", "-p", prompt,
            "--plugin-dir", str(plugin),
            "--model", "haiku",
            "--effort", "medium",
            "--permission-mode", "bypassPermissions",
            "--output-format", "stream-json",
            "--verbose",
        ],
        capture_output=True, text=True, timeout=timeout, env=env, cwd=str(project),
    )
    return proc, time.monotonic() - started


def _stream(proc) -> list[dict]:
    out = []
    for line in proc.stdout.splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _agent_calls(entries):
    """[(tool_use_id, tool_use_block)] for every native Agent/Task call in the stream."""
    calls = []
    for entry in entries:
        for block in (entry.get("message") or {}).get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") in ("Agent", "Task"):
                calls.append(block)
    return calls


def _result_for(entries, tool_use_id):
    for entry in entries:
        for block in (entry.get("message") or {}).get("content") or []:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and block.get("tool_use_id") == tool_use_id
            ):
                return block, entry
    return None, None


def _records(artifacts: Path) -> list[dict]:
    if not artifacts.is_dir():
        return []
    return [
        json.loads((p / "record.json").read_text(encoding="utf-8"))
        for p in sorted(artifacts.iterdir())
        if (p / "record.json").is_file()
    ]


def _flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


@pytest.mark.timeout(400)
def test_live_native_agent_frame():
    """L1: the native Agent call returns the child's answer in a non-error tool_result
    carrying the run id as `agentId`; the run is COMPLETED; the parent's permission mode
    and effort reach the child's argv."""
    _require_live()
    root = Path(tempfile.mkdtemp(prefix="ah76-"))
    try:
        plugin = _build_plugin_dir(root)
        env, project, artifacts = _session_env(root)
        proc, _ = _claude(
            plugin, project, env,
            "Use the Agent tool with subagent_type general-purpose and the prompt "
            "'Reply with exactly the single word PINEAPPLE and nothing else.' Do nothing else.",
            timeout=380,
        )
        entries = _stream(proc)
        calls = _agent_calls(entries)
        assert calls, f"the model never made a native Agent call:\n{proc.stdout!r}\n{proc.stderr!r}"
        block, entry = _result_for(entries, calls[0]["id"])
        assert block is not None, f"no tool_result for the Agent call:\n{proc.stdout!r}"
        text = _tool_result_text(block)
        assert not block.get("is_error"), text
        assert "PINEAPPLE" in text.upper(), text
        assert DENY_MARKER not in text, f"the #52 deny text came back: {text!r}"
        records = _records(artifacts)
        assert len(records) == 1, records
        record = records[0]
        assert record["state"] == "COMPLETED"
        run_id = record.get("run_id") or record.get("id")
        assert run_id, record
        # Bind the run id to the `agentId` field of THIS Agent tool result (not merely
        # "appears somewhere in the entry"): the structured result's agentId, or the
        # hand-back frame text's `agentId: <id>` line.
        structured = entry.get("tool_use_result")
        structured_id = structured.get("agentId") if isinstance(structured, dict) else None
        frame_ids = re.findall(r"agentId\W{1,4}([A-Za-z0-9_-]+)", text)
        assert str(run_id) == structured_id or str(run_id) in frame_ids, (
            f"agentId != run id {run_id!r}: structured={structured_id!r} frame={frame_ids!r}\n{text!r}"
        )
        assert _flag(record["argv"], "--permission-mode") == "bypassPermissions"
        assert _flag(record["argv"], "--effort") == "medium"
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.timeout(600)
def test_live_three_parallel_agents():
    """L2: three Agent calls in one message each run a 15 s child; they run in parallel."""
    _require_live()
    root = Path(tempfile.mkdtemp(prefix="ah76-"))
    try:
        plugin = _build_plugin_dir(root)
        env, project, artifacts = _session_env(root)
        proc, wall = _claude(
            plugin, project, env,
            "In ONE message, make three Agent tool calls in parallel (subagent_type "
            "general-purpose), each with the prompt 'Run the Bash command `sleep 15`, then "
            "reply DONE.' Do nothing else.",
            timeout=560,
        )
        entries = _stream(proc)
        calls = _agent_calls(entries)
        assert len(calls) == 3, f"expected three Agent calls, got {len(calls)}:\n{proc.stdout!r}"
        for call in calls:
            block, _ = _result_for(entries, call["id"])
            assert block is not None and not block.get("is_error"), block
        records = _records(artifacts)
        assert len(records) == 3, records
        total = sum(float(r.get("duration_s") or 0) for r in records)
        assert total > 0
        assert wall < 0.7 * total, f"not parallel: wall {wall:.0f}s vs summed run time {total:.0f}s"
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.timeout(400)
def test_live_hook_failure_never_native():
    """L3: with no binaries in the plugin dir the module cannot spawn: the Agent call
    ends as an error (the module's message or the #52 deny), and the call is never run
    natively -- no stream line is attributed to the Agent tool_use as its parent."""
    _require_live()
    root = Path(tempfile.mkdtemp(prefix="ah76-"))
    try:
        plugin = _build_plugin_dir(root, with_binaries=False)
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
        call_id = calls[0]["id"]
        block, _ = _result_for(entries, call_id)
        assert block is not None and block.get("is_error") is True, block
        text = _tool_result_text(block)
        assert "agent-harness could not run Agent through the harness" in text or DENY_MARKER in text, text
        assert not any(e.get("parent_tool_use_id") == call_id for e in entries), (
            "a native subagent ran under the Agent tool_use"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.skipif(
    os.environ.get("HARNESS_LIVE_LONG") != "1",
    reason="costs >= 1100 s; set HARNESS_LIVE_LONG=1 to run",
)
@pytest.mark.timeout(1800)
def test_live_long_runner_1100s():
    """L4: a child running two 560 s Bash sleeps completes through the hook, so the call
    is not bound by any hook-style limit."""
    _require_live()
    root = Path(tempfile.mkdtemp(prefix="ah76-"))
    try:
        plugin = _build_plugin_dir(root)
        env, project, artifacts = _session_env(root)
        proc, wall = _claude(
            plugin, project, env,
            "Use the Agent tool with subagent_type general-purpose and the prompt "
            "'Run the Bash command `sleep 560` twice in sequence (two separate Bash calls, "
            "timeout 600000 ms each), then reply DONE.' Do nothing else.",
            timeout=1750,
        )
        entries = _stream(proc)
        calls = _agent_calls(entries)
        assert calls, f"the model never made a native Agent call:\n{proc.stdout[-2000:]!r}"
        block, _ = _result_for(entries, calls[0]["id"])
        assert block is not None and not block.get("is_error"), block
        records = _records(artifacts)
        assert len(records) == 1 and records[0]["state"] == "COMPLETED", records
        assert wall >= 1100, wall
    finally:
        shutil.rmtree(root, ignore_errors=True)
