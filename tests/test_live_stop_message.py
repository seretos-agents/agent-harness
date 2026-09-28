"""Live tests against the real `claude` CLI for #65: a subagent blocked by the Stop
hook must neither abort nor self-poll around #64's internal wait, and on each
reactivation must judge plausibility from event_count/last_event_at/last_activity
instead of polling in a loop.

Deselected by default; run with
`python -m pytest -m live tests/test_live_stop_message.py -p no:cacheprovider`
(needs the real `claude` CLI on PATH plus real credentials; skips otherwise).

Per plan #65 "Completion" (round 5, in response to test-critic F1): R3 in
tests/test_hook.py is the PR-CI-gating offline evidence -- CI cannot run this
module (no real `claude` credentials there), so this module's *result*
(pass/fail) never gates the PR-CI check. But the run itself is not optional:
the implement-phase developer must actually execute
`python -m pytest -m live tests/test_live_stop_message.py -p no:cacheprovider`
against the real shipped hooks/stop_wait_message.md at least once per scenario
(R1, R2) and paste the real output (pass/fail, timings) into the change report
and PR body -- developer-verified evidence that a wrong wording cannot fake,
closing the loop a static string-equality check like R3's cannot close on its
own. A run that could not happen at all (no `claude` CLI / no credentials) is
a reported blocker, never a silently skipped step.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_live_claude import REPO, _real_credentials_path, _tool_result_text, _write_json

pytestmark = pytest.mark.live

FAKE_CLAUDE = Path(__file__).parent / "fixtures" / "fake_claude.py"

# Moved here from tests/test_hook.py (plan #65 gen2 "Affected files") -- this module
# is the marker's only consumer now that R3's offline exact-match pinning test (which
# used to import it from test_hook.py) is dropped. Stream-json block detection below
# needs a guaranteed distinctive marker to find a Stop block in real `claude`
# stream-json stdout; the pre-#65 prefix ("agent-harness: run(s) ") was too generic
# (could coincidentally appear in ordinary model output) for that.
STOP_MARKER = "agent-harness Stop hook: waited for unfinished run(s)"

_TRIALS = 3

# #64's internal Stop-hook wait pinned short so a live trial sees multiple real
# block/reactivate cycles instead of the 7200s default (plan Approach: "live tests
# set 15").
_STOP_WAIT_TIMEOUT_SECONDS = 15

# tautology::F2 (test-critic round 5, major): the original "TICK:8:5" directive (40s
# of tick-phase sleep) reliably produced only ~2 blocks against a 15s internal wait
# (write_context.py's Stop handler blocks for exactly stop_wait_timeout() once the run
# is still pending, so each cycle costs ~15s of internal wait plus one real model
# turn's latency) -- the run tends to reach COMPLETED during the third hook wait
# rather than surviving it, so R1 (below) never reached a third reading. That let a
# wording which ignores event_count/last_event_at entirely and just counts
# reactivations to a fixed N (e.g. "cancel on the third reactivation") pass R1 for
# free, because the run finished before a third reading ever happened -- R1's
# `assert not stops` per segment was never exercised against that trap. Widened to
# 20 ticks so the tick phase alone spans ~100s, comfortably more than 3 full
# (wait + real-turn-latency) cycles even under slow model API latency, so the ≥3
# block floor below (R1_MIN_BLOCKS) is reliably reachable rather than a coin flip.
_WORKING_RUN_DIRECTIVE = "TICK:20:5"

# tautology::F2 (test-critic round 5, major): see _WORKING_RUN_DIRECTIVE above --
# R1 must reach at least a third reading of a still-advancing run, or a wording that
# ignores event_count/last_event_at and just counts reactivations to a fixed N could
# pass by having the run finish before the trap N is ever reached.
R1_MIN_BLOCKS = 3


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
    # tautology::F5: earlier wording told the model to include the run's run_id in
    # its own summary, which made `run_id in final_text` (R2, below) hold no matter
    # what hooks/stop_wait_message.md says about reporting -- the check was primed
    # by the prompt, not by the hook's escalation-report instruction. The prompt
    # asks only for "a summary", so R2's `run_id in final_text` now has a real
    # source: it can only be satisfied by the model following the hook message's
    # own "end with a report naming the run_id ..." instruction (plan Approach).
    #
    # R1 retry (#65 round-2 FAIL): captured stdout from the failing trials showed
    # the model calling harness_start_prompt and then, in that SAME turn, calling
    # harness_wait_run itself -- it never ended its turn at all, so the Stop hook
    # never even got a chance to fire once. That is a gap in this prompt's own
    # sequencing instruction, not the Stop-hook protocol this test exists to prove
    # blind to: "then end your turn" left room for a model to read "end your turn"
    # as "eventually, once you've got something worth reporting" rather than
    # "immediately, with nothing else in between". Tightened below to remove that
    # room -- "call harness_start_prompt exactly once, then stop, nothing else, no
    # other tool call" -- while still saying nothing at all about what to do once
    # blocked (no mention of polling, waiting, or the hook), so it stays blind to
    # the actual behaviour under test.
    return (
        f"Call the harness_start_prompt tool exactly once, with the prompt "
        f"{directive!r}, to start a job. The instant that single tool call "
        "returns, end your turn immediately: do not call any other tool of any "
        "kind (not harness_start_prompt again, not any tool to check on the job "
        "or its status), and do not write any other text first. Just stop, with "
        "nothing else in this turn. You will be given a further turn later; only "
        "then decide what, if anything, to do next. Once you eventually learn "
        "the job has a final result, reply with a short one-line summary of it, "
        "and stop."
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


def _tool_use_inputs(lines: list[str], start: int, end: int, bare_name: str) -> list[dict]:
    """`input` payloads of every tool_use block within lines[start:end) whose bare
    name (suffix after the last "__" for an MCP tool) matches `bare_name` --
    independent evidence of *what a real tool call actually did* (e.g. which
    run_id harness_stop_run was invoked with), as opposed to what the model's own
    free-text summary claims (tautology::F3)."""
    inputs = []
    for block in _assistant_content_blocks(lines, start, end):
        if block.get("type") != "tool_use":
            continue
        name = block.get("name") or ""
        bare = name.rsplit("__", 1)[-1] if "__" in name else name
        if bare == bare_name:
            inputs.append(block.get("input") or {})
    return inputs


def _segment_bounds(blocks: list[int], idx: int, total_len: int) -> tuple[int, int]:
    """The `[start, end)` line range of segment `idx` -- from its own STOP_MARKER
    line up to (but not including) the next one, or the end of the stream for the
    last segment."""
    start = blocks[idx]
    end = blocks[idx + 1] if idx + 1 < len(blocks) else total_len
    return start, end


def _tool_result_blocks(lines: list[str], start: int, end: int) -> list[dict]:
    return [b for b in _assistant_content_blocks(lines, start, end) if b.get("type") == "tool_result"]


def _poll_result_payload(lines: list[str], start: int, end: int) -> dict | None:
    """The single harness_poll_run call's own tool_result payload within
    lines[start:end) -- state / event_count / last_activity, read via regex from
    the real MCP tool_result's raw text content (never `json.loads`, since the
    exact shape a host wraps a structured dict's text representation in is not
    pinned here) rather than inferred from the model's own account of what the
    poll said -- the same independent-evidence-from-the-real-tool-call pattern
    `_tool_use_inputs` already uses elsewhere in this module (tautology::F3).
    Returns None if no harness_poll_run tool_use (or no matching tool_result) is
    found in this range."""
    poll_use_id = None
    for block in _assistant_content_blocks(lines, start, end):
        if block.get("type") != "tool_use":
            continue
        name = block.get("name") or ""
        bare = name.rsplit("__", 1)[-1] if "__" in name else name
        if bare == "harness_poll_run":
            poll_use_id = block.get("id")
            break
    if poll_use_id is None:
        return None
    for block in _tool_result_blocks(lines, start, end):
        if block.get("tool_use_id") != poll_use_id:
            continue
        text = _tool_result_text(block)
        state_match = re.search(r'"state"\s*:\s*"([A-Z_]+)"', text)
        count_match = re.search(r'"event_count"\s*:\s*(-?\d+)', text)
        activity_match = re.search(r'"last_activity"\s*:\s*("(?:[^"\\]|\\.)*"|null)', text)
        last_activity = None
        if activity_match and activity_match.group(1) != "null":
            last_activity = json.loads(activity_match.group(1))
        return {
            "state": state_match.group(1) if state_match else None,
            "event_count": int(count_match.group(1)) if count_match else None,
            "last_activity": last_activity,
        }
    return None


def _bash_commands(lines: list[str], start: int, end: int) -> list[str]:
    commands = []
    for block in _assistant_content_blocks(lines, start, end):
        if block.get("type") == "tool_use" and block.get("name") == "Bash":
            command = (block.get("input") or {}).get("command")
            if isinstance(command, str):
                commands.append(command)
    return commands


# tautology::F4: R1's shell-wait check originally matched only the literal
# substring "harness wait" -- wording that dodges that exact phrase while still
# leading the model to shell out to a polling/wait workaround (a `harness poll`/
# `harness status`-style CLI call, or a `sleep`-then-recheck loop in Bash) passed
# every R1 assertion. Broaden to the family of workarounds the plan's Approach
# rules out ("no polling loop", "no further calls" once a run is judged working).
_POLLING_WORKAROUND_PATTERNS = [
    re.compile(r"harness\s+wait", re.IGNORECASE),
    re.compile(r"harness\s+poll", re.IGNORECASE),
    re.compile(r"harness\s+status", re.IGNORECASE),
    re.compile(r"\bsleep\b", re.IGNORECASE),
]


def _polling_workaround_commands(commands: list[str]) -> list[str]:
    return [c for c in commands if any(p.search(c) for p in _POLLING_WORKAROUND_PATTERNS)]


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
    Stop hook at least three times; after each block the subagent must call
    harness_poll_run exactly once and nothing else (no harness_wait_run, no
    harness_stop_run, no shell `harness wait`), and the session ends normally
    (exit 0, a non-error terminal result) once the run completes, with the real run
    record reaching COMPLETED.

    tautology::F2 (test-critic round 5, major): a floor of "at least 1 block" let a
    wording that ignores event_count/last_event_at and just counts reactivations to
    a fixed N (e.g. "cancel on the third reactivation, no matter what") pass this
    test for free whenever the TICK run happened to finish within two hook waits --
    the third reading, where that trap would fire and this test's `assert not
    stops` would catch it, never happened. _WORKING_RUN_DIRECTIVE is tuned (see its
    definition above) so the run reliably needs at least R1_MIN_BLOCKS=3 hook-wait
    cycles to finish, forcing a genuinely repeated "RUNNING + advanced -> keep
    waiting, no cancel" judgement rather than just one.

    Expected RED reason: hooks/stop_wait_message.md today tells the model to "call
    harness_wait_run / harness_poll_run until they are terminal" and carries no
    STOP_MARKER prefix at all, so either zero blocks are ever matched in stdout (the
    marker this test's block detection depends on is entirely absent from the
    unfixed file), or -- if matched via some other means -- a segment shows more
    than one harness_poll_run call, or a harness_wait_run call, contradicting "one
    poll per block, nothing else"."""
    _skip_unless_live_available()

    config_dir, project_dir, artifacts_dir, run_env = _provision_stop_message_fixture(tmp_path)
    proc = _run_stop_message_scenario(project_dir, run_env, _WORKING_RUN_DIRECTIVE)
    lines = _stream_lines(proc.stdout)
    blocks = _block_line_indices(lines)
    assert blocks, (
        f"zero Stop-hook blocks matched (marker {STOP_MARKER!r} never seen in "
        f"stdout); full dump: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert len(blocks) >= R1_MIN_BLOCKS, (
        f"only {len(blocks)} block(s) matched, expected at least {R1_MIN_BLOCKS} -- "
        f"the run finished before a third reading of a still-advancing run could "
        f"happen, so a wording that ignores event_count/last_event_at and just "
        f"counts reactivations to a fixed N was never actually exercised against "
        f"its trap (test-critic round 5, tautology::F2); consider widening "
        f"_WORKING_RUN_DIRECTIVE further if this recurs. stdout={proc.stdout!r}"
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
        # tautology::F4: broadened past the literal "harness wait" substring so a
        # wording that dodges only that exact phrase -- while still leading the
        # model to shell out to a `harness poll`/`harness status`-style workaround,
        # or a `sleep`-then-recheck loop -- does not pass R1's no-self-polling
        # requirement for free.
        shell_workarounds = _polling_workaround_commands(_bash_commands(lines, block_at, segment_end))
        assert not shell_workarounds, (
            f"segment {idx} shelled out to a polling/wait workaround: {shell_workarounds}"
        )

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


_STALLED_DIRECTIVE = "TOOL:Bash SLEEP:900"


@pytest.mark.timeout(300)
@pytest.mark.parametrize("trial", range(_TRIALS))
def test_live_stalled_run_escalates_and_ends(tmp_path, trial):
    """R2 driving test: a run that goes silent right after one Bash tool_use (a
    call that can legitimately run long) gets one silent reading's grace -- no
    cancel on the first silent block -- and is only cancelled (harness_stop_run) on
    the second consecutive silent reading; the final reply names the run_id and the
    last_activity value the escalation poll actually read. Never more than one poll
    per block, never harness_wait_run.

    Plan #65 gen2 "R2 assertion fixes" (responding to test-critic round 6, F2/F3):

    - The report-clause check no longer greps for the word "stall" (a paraphrase of
      the message's own wording, easily satisfied by coincidence or by a differently
      worded message). It instead requires the final reply to name the run_id AND
      the `last_activity` value read off the escalation poll's own tool_result --
      value that appears in neither `_blind_prompt`'s prompt text nor the static
      hooks/stop_wait_message.md file (checked below), so it can only end up in the
      report if the model actually read it off a real poll result and followed the
      message's instruction to report it.

    - The block-count inference (`len(blocks) == 3`) was originally replaced by
      explicit position checks: the stop is in the last segment; the stop segment's
      poll and the immediately preceding segment's poll are both RUNNING with equal
      event_count (the second consecutive silent reading). ("No marker line follows
      it" was dropped as a separate check: test-critic gen2 round 1, tautology::F1
      found it always vacuously true once "stop is in the last segment" holds -- see
      the comment at Check 2's old site, below.)

    Plan-critic gen2 round 1 (misread::F1, blocking): that position check alone
    ("two equal-event_count RUNNING polls in a row before the stop") cannot tell
    the promised grace wait apart from cancelling at the very first silent reading
    -- for this SLEEP:900 scenario, event_count never changes again once the one
    Bash tool_use event has landed, so *every* poll from the first block onward
    reads equal to the one before it, whether the message gives one wait of grace
    or none at all. The genuinely distinguishing signal is therefore not "are the
    last two polls equal" (always true here) but "how many segments came before the
    stop": a message that cancels on the very first silent reading only ever
    reaches segment index 1 before the run goes terminal, while the grace-then-
    escalate rule this plan specifies needs a first silent reading that is *not*
    cancelled (segment 1) before the second one that is (segment 2) -- i.e. the
    stop must sit at segment index 2.

    Test-critic gen2 round 1 (tautology::F2, major): the original fix for
    misread::F1 only enforced `stop_idx >= 2`, a floor. That does not rule out a
    wording that grants two or more waits of grace and cancels on the third (or
    later) consecutive silent reading -- such a wording would satisfy every other
    check here too, since event_count never changes again in this scenario once the
    silence starts. Check 3 below now asserts `stop_idx == 2` exactly, which --
    combined with Check 1 pinning the stop to the last segment -- pins the total
    segment count to exactly 3: back to the same exact timing the original
    `len(blocks) == 3` inference expressed, but derived from position checks rather
    than a bare count.

    Expected RED reason: hooks/stop_wait_message.md has no stall/grace rule at all
    today (and carries no STOP_MARKER, so block detection itself may match zero
    blocks) -- a subagent following the old wording either self-polls indefinitely
    (harness_wait_run appears) or never cancels at all, so
    `stop_calls_per_segment.count(True) == 1` fails."""
    _skip_unless_live_available()

    config_dir, project_dir, artifacts_dir, run_env = _provision_stop_message_fixture(tmp_path)
    proc = _run_stop_message_scenario(project_dir, run_env, _STALLED_DIRECTIVE)
    lines = _stream_lines(proc.stdout)
    blocks = _block_line_indices(lines)
    assert blocks, (
        f"zero Stop-hook blocks matched (marker {STOP_MARKER!r} never seen in "
        f"stdout); full dump: stdout={proc.stdout!r} stderr={proc.stderr!r}"
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
    stop_idx = next(i for i, had_stop in enumerate(stop_calls_per_segment) if had_stop)

    # Check 1: "the stop is in the last segment".
    assert stop_idx == len(blocks) - 1, (
        f"harness_stop_run was called in segment {stop_idx}, not the last of "
        f"{len(blocks)} segments -- escalation must be the final act, nothing may "
        f"follow it; stop_calls_per_segment={stop_calls_per_segment}"
    )

    # Check 2 removed (test-critic gen2 round 1, tautology::F1, critical): this used
    # to scan `lines[stop_end:]` for a further STOP_MARKER, claiming to be an
    # independent proof that the session did not keep getting blocked after the
    # cancel. But Check 1 above already establishes stop_idx == len(blocks) - 1 (the
    # stop segment is the *last* recorded segment), and `_segment_bounds` always
    # returns `end = len(lines)` for the last segment -- so `lines[stop_end:]` is
    # always `[]` and `not any(...)` over an empty sequence can never be False. The
    # assertion could not fail under any implementation; it proved nothing beyond
    # what Check 1 already establishes (same category as gen1 round 3's F4 finding,
    # resolved the same way there: drop the redundant assertion rather than dress it
    # up as independent evidence). "The session actually ended, not merely stopped
    # emitting Stop-hook blocks" is covered by the real, non-tautological signals
    # already asserted later in this test: `proc.returncode == 0` and the terminal
    # `result` event (both below) -- those, not a scan of an always-empty slice, are
    # the genuine evidence that the process exited rather than being killed by an
    # external timeout.
    stop_start, stop_end = _segment_bounds(blocks, stop_idx, len(lines))

    # Check 3 (plan-critic gen2 round 1, misread::F1; tightened per test-critic gen2
    # round 1, tautology::F2, major): the stop must sit at *exactly* segment index
    # 2, not merely at index >= 2. A floor alone cannot rule out a wording that
    # grants two or more waits of grace and only cancels on the third (or later)
    # consecutive silent reading -- every poll in this SLEEP:900 scenario reads the
    # same event_count from the first block onward, so such an over-graceful
    # wording would still satisfy every other check here (equal event_count,
    # run_id, exit code, CANCELLED) while violating the plan's exact rule: "no
    # cancel at the first silent reading; harness_stop_run at the second". Combined
    # with Check 1 (stop_idx == len(blocks) - 1, i.e. the stop is always the last
    # segment), pinning stop_idx == 2 also pins len(blocks) == 3: one advancing
    # reading (segment 0: the Bash tool_use event lands before the first poll), one
    # silent reading given grace (segment 1: no stop), and the second consecutive
    # silent reading that escalates (segment 2, the last segment). stop_idx < 2
    # means no grace was given at all; stop_idx > 2 (equivalently, more than 3
    # segments) means more than one wait of grace was granted before cancelling.
    assert stop_idx == 2, (
        f"harness_stop_run was called in segment {stop_idx} (0-indexed) across "
        f"{len(blocks)} total segments, not segment 2 -- the plan's rule is exactly "
        f"one silent reading of grace (segment 1) then escalate on the second "
        f"consecutive silent reading (segment 2); stop_idx < 2 means no grace was "
        f"given at all, and stop_idx > 2 means more than one wait of grace was "
        f"granted before cancelling (test-critic gen2 round 1, tautology::F2); "
        f"stop_calls_per_segment={stop_calls_per_segment}"
    )

    # Check 4: "the stop segment's poll and the previous segment's poll are both
    # RUNNING with equal event_count" (plan Approach) -- the second consecutive
    # silent reading that triggers escalation.
    prev_start, prev_end = _segment_bounds(blocks, stop_idx - 1, len(lines))
    poll_stop = _poll_result_payload(lines, stop_start, stop_end)
    poll_prev = _poll_result_payload(lines, prev_start, prev_end)
    assert poll_stop is not None and poll_prev is not None, (
        f"could not read the harness_poll_run tool_result payload for the "
        f"escalation segment or its immediate predecessor; "
        f"poll_stop={poll_stop!r} poll_prev={poll_prev!r}"
    )
    assert poll_stop["state"] == "RUNNING" and poll_prev["state"] == "RUNNING", (
        f"expected both the escalation poll and its predecessor to read the run as "
        f"RUNNING (never terminal before the deliberate harness_stop_run cancel); "
        f"poll_stop={poll_stop!r} poll_prev={poll_prev!r}"
    )
    assert poll_stop["event_count"] is not None and poll_stop["event_count"] == poll_prev["event_count"], (
        f"expected the escalation segment's poll and the immediately preceding "
        f"segment's poll to show the same event_count (unchanged = silent); "
        f"poll_stop={poll_stop!r} poll_prev={poll_prev!r}"
    )

    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    final = _final_result_event(lines)
    assert final is not None, f"no terminal result event; stdout={proc.stdout!r}"
    final_text = final.get("result") or ""

    started = _started_run_ids(artifacts_dir)
    assert len(started) == 1, f"expected exactly one run started, got {started}"
    run_id = next(iter(started))

    # tautology::F3 (test-critic round 2): `run_id in final_text` alone is
    # self-fulfilling if the prompt itself asks the model to echo the run_id.
    # Ground the check in independent evidence too: the harness_stop_run tool call
    # captured from the real escalation segment must itself have been invoked with
    # the actually-started run_id -- evidence from what the model's tool call did,
    # not from what it chose to say afterwards.
    stop_inputs = _tool_use_inputs(lines, stop_start, len(lines), "harness_stop_run")
    assert stop_inputs, (
        f"no harness_stop_run tool_use captured in the escalation segment; "
        f"lines={lines[stop_start:]!r}"
    )
    assert stop_inputs[0].get("run_id") == run_id, (
        f"harness_stop_run was not called with the actually-started run_id "
        f"{run_id!r} (independent evidence from the real tool call, not the "
        f"model's self-reported summary text); stop_inputs={stop_inputs!r}"
    )
    assert run_id in final_text, f"final reply does not name the run_id {run_id!r}: {final_text!r}"

    # Plan #65 gen2 "R2 assertion fixes" / plan-critic gen2 round 1 (untestable::F2):
    # the report-clause check is grounded in the escalation poll's own last_activity
    # value, but only if that value could not plausibly have reached the final
    # reply by any other route than the model reading it off the poll result and
    # following the message's own report instruction. Verify both exclusions
    # directly rather than asserting them as an unverified premise.
    last_activity = poll_stop["last_activity"]
    assert last_activity, f"escalation poll's tool_result carried no last_activity; poll_stop={poll_stop!r}"
    prompt_text = _blind_prompt(_STALLED_DIRECTIVE)
    assert last_activity not in prompt_text, (
        f"last_activity {last_activity!r} must not already appear in the "
        f"subagent's own prompt, or its appearance in the final report would prove "
        f"nothing about whether the model read it from the poll result; "
        f"prompt={prompt_text!r}"
    )
    message_text = (REPO / "hooks" / "stop_wait_message.md").read_text(encoding="utf-8")
    assert last_activity not in message_text, (
        f"last_activity {last_activity!r} must not already be present in the "
        f"static hook message file itself, or its appearance in the report would "
        f"prove nothing about whether the model read it from the poll result; "
        f"message file={message_text!r}"
    )
    assert last_activity in final_text, (
        f"final reply does not name the last_activity value {last_activity!r} that "
        f"the escalation poll's own tool_result carried -- only the hook message's "
        f"own report instruction (naming run_id and last_activity) could "
        f"plausibly have prompted this, since last_activity appears neither in the "
        f"prompt nor in hooks/stop_wait_message.md itself; final_text={final_text!r}"
    )

    assert _record_state(artifacts_dir, run_id) == "CANCELLED", (
        f"run {run_id}'s record never reached CANCELLED: "
        f"{_record_state(artifacts_dir, run_id)!r}"
    )
