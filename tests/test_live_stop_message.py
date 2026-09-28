"""Live tests against the real `claude` CLI for #65: a subagent blocked by the Stop
hook must neither abort nor self-poll around #64's internal wait, and on each
reactivation must judge plausibility from event_count/last_event_at/last_activity
instead of polling in a loop.

Deselected by default; run with
`python -m pytest -m live tests/test_live_stop_message.py -p no:cacheprovider`
(needs the real `claude` CLI on PATH plus real credentials; skips otherwise).

Per plan #65 "Completion": these two scenarios are supplementary, driving-test
evidence -- pass, fail, or "not run" (no `claude` CLI / no credentials) never
blocks the package's completion. R3 in tests/test_hook.py is the PR-CI-gating
offline evidence.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_hook import STOP_MARKER
from test_live_claude import REPO, _real_credentials_path, _write_json

pytestmark = pytest.mark.live

FAKE_CLAUDE = Path(__file__).parent / "fixtures" / "fake_claude.py"

_TRIALS = 3

# #64's internal Stop-hook wait pinned short so a live trial sees multiple real
# block/reactivate cycles instead of the 7200s default (plan Approach: "live tests
# set 15").
_STOP_WAIT_TIMEOUT_SECONDS = 15


def _provision_stop_message_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    """Marketplace-install a fixture plugin carrying this repo's own hooks/hooks.json
    verbatim plus a `bin/harness` shim and an `mcpServers.harness` entry wired to the
    fake CLI (fixture pattern: test_live_claude.py's
    test_live_stop_hook_tracks_started_run / test_live_native_subagent_dispatch_denied
    -- inlining provisioning per test rather than refactoring that module's own copies
    is out of scope for #65, plan "Mechanism balance"). Returns
    `(config_dir, project_dir, artifacts_dir, run_env)`: `run_env` is what the `claude
    -p` subprocess itself must be given -- HARNESS_STOP_WAIT_TIMEOUT_SECONDS and
    HARNESS_ARTIFACTS_DIR both pinned, so the Stop hook subprocess (a child of that
    same `claude -p` process, inheriting its env, not the MCP server's own configured
    env) reads the same run records the MCP server itself writes via the fake CLI."""
    config_dir = tmp_path / "claude-config"
    project_dir = tmp_path / "live-project"
    project_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    shutil.copy(_real_credentials_path(), config_dir / ".credentials.json")

    artifacts_dir = tmp_path / "artifacts"

    marketplace_dir = tmp_path / "marketplace"
    fixture_dir = marketplace_dir / "harness-stop-message-fixture"
    (fixture_dir / "hooks").mkdir(parents=True)
    shutil.copy2(REPO / "hooks" / "hooks.json", fixture_dir / "hooks" / "hooks.json")

    bin_dir = fixture_dir / "bin"
    bin_dir.mkdir(parents=True)
    harness_sh = bin_dir / "harness"
    with open(harness_sh, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f'#!/bin/sh\nexec "{Path(sys.executable).as_posix()}" -m harness_plugin "$@"\n')
    if sys.platform != "win32":
        harness_sh.chmod(0o755)

    _write_json(
        fixture_dir / ".claude-plugin" / "plugin.json",
        {
            "name": "harness-stop-message-fixture",
            "hooks": "./hooks/hooks.json",
            "mcpServers": {
                "harness": {
                    "command": sys.executable,
                    "args": ["-m", "harness_plugin"],
                    "env": {
                        "HARNESS_ARTIFACTS_DIR": str(artifacts_dir),
                        "HARNESS_CLAUDE_ARGV": json.dumps([sys.executable, str(FAKE_CLAUDE)]),
                    },
                }
            },
        },
    )
    _write_json(
        marketplace_dir / ".claude-plugin" / "marketplace.json",
        {
            "name": "lt",
            "owner": {"name": "65 live fixture"},
            "metadata": {"version": "0.0.0", "description": "65 live fixture marketplace"},
            "plugins": [
                {
                    "name": "harness-stop-message-fixture",
                    "description": "65 live fixture: this repo's hooks.json + a fake-CLI-backed MCP server",
                    "source": "./harness-stop-message-fixture",
                    "category": "mcp",
                    "version": "0.0.0",
                },
            ],
        },
    )

    setup_env = {**os.environ, "CLAUDE_CONFIG_DIR": str(config_dir)}
    added = subprocess.run(
        ["claude", "plugin", "marketplace", "add", str(marketplace_dir)],
        capture_output=True, text=True, timeout=60, env=setup_env,
    )
    assert added.returncode == 0, f"marketplace add failed: {added.stdout} {added.stderr}"
    installed = subprocess.run(
        ["claude", "plugin", "install", "harness-stop-message-fixture@lt", "-y"],
        capture_output=True, text=True, timeout=60, env=setup_env,
    )
    assert installed.returncode == 0, f"install failed: {installed.stdout} {installed.stderr}"

    run_env = {**os.environ, "CLAUDE_CONFIG_DIR": str(config_dir)}
    run_env["HARNESS_STOP_WAIT_TIMEOUT_SECONDS"] = str(_STOP_WAIT_TIMEOUT_SECONDS)
    run_env["HARNESS_ARTIFACTS_DIR"] = str(artifacts_dir)
    return config_dir, project_dir, artifacts_dir, run_env


def _blind_prompt(directive: str) -> str:
    """The subagent's only instruction -- deliberately blind to the Stop-hook
    protocol. Whatever it does once blocked must come entirely from the hook's own
    stderr feedback (hooks/stop_wait_message.md), which is exactly what these
    scenarios exist to prove -- coaching the model here about polling/waiting would
    make a self-polling old-wording run pass this test for the wrong reason."""
    return (
        f"Use the harness_start_prompt tool with the prompt {directive!r} to start a "
        "job, then end your turn. Once you have the job's final result, reply with a "
        "short one-line summary of it that includes the run's run_id, and stop."
    )


def _run_stop_message_scenario(project_dir: Path, run_env: dict[str, str], directive: str):
    return subprocess.run(
        [
            "claude", "-p", _blind_prompt(directive),
            "--model", "haiku",
            "--permission-mode", "bypassPermissions",
            "--output-format", "stream-json",
            "--verbose",
        ],
        capture_output=True, text=True, timeout=280, env=run_env, cwd=str(project_dir),
    )


def _stream_lines(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if line.strip()]


def _block_line_indices(lines: list[str]) -> list[int]:
    """Line indices whose raw text carries the Stop hook's STOP_MARKER -- the single
    source of truth for "a block happened here" (plan Approach: "stream-json stdout
    entries containing the marker; zero blocks = hard failure with dump"; a
    transcript-JSONL fallback is the documented next step only if this source proves
    to not carry the hook's feedback -- not implemented here since it has not been
    needed)."""
    return [i for i, line in enumerate(lines) if STOP_MARKER in line]


def _assistant_content_blocks(lines: list[str], start: int, end: int) -> list[dict]:
    blocks: list[dict] = []
    for line in lines[start:end]:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        for block in message.get("content") or []:
            if isinstance(block, dict):
                blocks.append(block)
    return blocks


def _tool_use_names(lines: list[str], start: int, end: int) -> list[str]:
    """Bare tool names (suffix after the last "__" for an MCP tool) of every
    tool_use block in assistant messages within lines[start:end)."""
    names = []
    for block in _assistant_content_blocks(lines, start, end):
        if block.get("type") == "tool_use":
            name = block.get("name") or ""
            names.append(name.rsplit("__", 1)[-1] if "__" in name else name)
    return names


def _bash_commands(lines: list[str], start: int, end: int) -> list[str]:
    commands = []
    for block in _assistant_content_blocks(lines, start, end):
        if block.get("type") == "tool_use" and block.get("name") == "Bash":
            command = (block.get("input") or {}).get("command")
            if isinstance(command, str):
                commands.append(command)
    return commands


def _final_result_event(lines: list[str]) -> dict | None:
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and entry.get("type") == "result":
            return entry
    return None


def _started_run_ids(artifacts_dir: Path) -> set[str]:
    if not artifacts_dir.is_dir():
        return set()
    return {
        record_dir.name
        for record_dir in artifacts_dir.iterdir()
        if (record_dir / "record.json").is_file()
    }


def _record_state(artifacts_dir: Path, run_id: str) -> str | None:
    try:
        data = json.loads((artifacts_dir / run_id / "record.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    state = data.get("state") if isinstance(data, dict) else None
    return state.get("__runstate__") if isinstance(state, dict) else None


def _skip_unless_live_available() -> None:
    if shutil.which("claude") is None:
        pytest.skip("the real `claude` CLI is not on PATH")
    real_credentials = _real_credentials_path()
    if not real_credentials.is_file():
        pytest.skip(f"no real credentials at {real_credentials}; cannot run a live child")


# --- R1: a working run is never self-polled, never aborted -----------------------


@pytest.mark.timeout(300)
@pytest.mark.parametrize("trial", range(_TRIALS))
def test_live_blocked_working_run_is_not_self_polled(tmp_path, trial):
    """R1 driving test: a run that keeps making visible progress (TICK) blocks the
    Stop hook one or more times; after each block the subagent must call
    harness_poll_run exactly once and nothing else (no harness_wait_run, no
    harness_stop_run, no shell `harness wait`), and the session ends normally
    (exit 0, a non-error terminal result) once the run completes, with the real run
    record reaching COMPLETED.

    Expected RED reason: hooks/stop_wait_message.md today tells the model to "call
    harness_wait_run / harness_poll_run until they are terminal" and carries no
    STOP_MARKER prefix at all, so either zero blocks are ever matched in stdout (the
    marker this test's block detection depends on is entirely absent from the
    unfixed file), or -- if matched via some other means -- a segment shows more
    than one harness_poll_run call, or a harness_wait_run call, contradicting "one
    poll per block, nothing else"."""
    _skip_unless_live_available()

    config_dir, project_dir, artifacts_dir, run_env = _provision_stop_message_fixture(tmp_path)
    proc = _run_stop_message_scenario(project_dir, run_env, "TICK:8:5")
    lines = _stream_lines(proc.stdout)
    blocks = _block_line_indices(lines)
    assert blocks, (
        f"zero Stop-hook blocks matched (marker {STOP_MARKER!r} never seen in "
        f"stdout); full dump: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )

    for idx, block_at in enumerate(blocks):
        segment_end = blocks[idx + 1] if idx + 1 < len(blocks) else len(lines)
        names = _tool_use_names(lines, block_at, segment_end)
        polls = [n for n in names if n == "harness_poll_run"]
        waits = [n for n in names if n == "harness_wait_run"]
        stops = [n for n in names if n == "harness_stop_run"]
        assert len(polls) == 1, (
            f"segment {idx} called harness_poll_run {len(polls)} time(s), expected "
            f"exactly 1: names={names}"
        )
        assert not waits, f"segment {idx} self-polled via harness_wait_run: names={names}"
        assert not stops, f"segment {idx} cancelled a still-working run: names={names}"
        shell_waits = [c for c in _bash_commands(lines, block_at, segment_end) if "harness wait" in c]
        assert not shell_waits, f"segment {idx} shelled out to `harness wait`: {shell_waits}"

    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    final = _final_result_event(lines)
    assert final is not None, f"no terminal result event; stdout={proc.stdout!r}"
    assert not final.get("is_error"), final

    started = _started_run_ids(artifacts_dir)
    assert len(started) == 1, f"expected exactly one run started, got {started}"
    run_id = next(iter(started))
    assert _record_state(artifacts_dir, run_id) == "COMPLETED", (
        f"run {run_id}'s record never reached COMPLETED: "
        f"{_record_state(artifacts_dir, run_id)!r}"
    )


# --- R2: a silent run gets one wait of grace, then is escalated and cancelled ----


@pytest.mark.timeout(300)
@pytest.mark.parametrize("trial", range(_TRIALS))
def test_live_stalled_run_escalates_and_ends(tmp_path, trial):
    """R2 driving test: a run that goes silent right after one Bash tool_use (a
    call that can legitimately run long) gets one silent reading's grace -- no
    cancel on the first silent block -- and is only cancelled (harness_stop_run) on
    the second consecutive silent reading; the final reply names the run_id. Never
    more than one poll per block, never harness_wait_run.

    Expected RED reason: hooks/stop_wait_message.md has no stall/grace rule at all
    today (and carries no STOP_MARKER, so block detection itself may match zero
    blocks) -- a subagent following the old wording either self-polls indefinitely
    (harness_wait_run appears, or the run is still pending well past the 4-block
    ceiling this test allows) or never cancels at all, so
    `stop_calls_per_segment.count(True) == 1` fails."""
    _skip_unless_live_available()

    config_dir, project_dir, artifacts_dir, run_env = _provision_stop_message_fixture(tmp_path)
    proc = _run_stop_message_scenario(project_dir, run_env, "TOOL:Bash SLEEP:900")
    lines = _stream_lines(proc.stdout)
    blocks = _block_line_indices(lines)
    assert blocks, (
        f"zero Stop-hook blocks matched (marker {STOP_MARKER!r} never seen in "
        f"stdout); full dump: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert len(blocks) <= 4, (
        f"too many blocks ({len(blocks)}) -- expected escalation well before this "
        f"many silent readings: {blocks}"
    )

    stop_calls_per_segment = []
    for idx, block_at in enumerate(blocks):
        segment_end = blocks[idx + 1] if idx + 1 < len(blocks) else len(lines)
        names = _tool_use_names(lines, block_at, segment_end)
        polls = [n for n in names if n == "harness_poll_run"]
        waits = [n for n in names if n == "harness_wait_run"]
        stops = [n for n in names if n == "harness_stop_run"]
        assert len(polls) == 1, (
            f"segment {idx} called harness_poll_run {len(polls)} time(s), expected "
            f"exactly 1: names={names}"
        )
        assert not waits, f"segment {idx} self-polled via harness_wait_run: names={names}"
        assert len(stops) <= 1, f"segment {idx} called harness_stop_run more than once: names={names}"
        stop_calls_per_segment.append(bool(stops))

    assert stop_calls_per_segment.count(True) == 1, (
        f"expected exactly one harness_stop_run call across all segments, got "
        f"{stop_calls_per_segment.count(True)}: {stop_calls_per_segment}"
    )
    assert stop_calls_per_segment[-1], (
        f"harness_stop_run must happen in the last segment (the second consecutive "
        f"silent reading), not earlier: {stop_calls_per_segment}"
    )
    assert not stop_calls_per_segment[0], (
        f"harness_stop_run must not happen on the first silent reading (one wait of "
        f"grace, plan Approach): {stop_calls_per_segment}"
    )

    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    final = _final_result_event(lines)
    assert final is not None, f"no terminal result event; stdout={proc.stdout!r}"
    final_text = final.get("result") or ""

    started = _started_run_ids(artifacts_dir)
    assert len(started) == 1, f"expected exactly one run started, got {started}"
    run_id = next(iter(started))
    assert run_id in final_text, f"final reply does not name the run_id {run_id!r}: {final_text!r}"
    assert _record_state(artifacts_dir, run_id) == "CANCELLED", (
        f"run {run_id}'s record never reached CANCELLED: "
        f"{_record_state(artifacts_dir, run_id)!r}"
    )
