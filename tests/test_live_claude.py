"""Live test against the real `claude` CLI. Deselected by default; run with
`python -m pytest -m live`."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import anyio
import pytest
from conftest import SESSION_ID
from test_mcp_tools import TERMINAL, _call, _poll_until_terminal, _run, _session

pytestmark = pytest.mark.live

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.timeout(300)  # exceeds the repo's global 60s default: real 240s poll budget below
def test_live_start_agent_prompt_becomes_user_message(live_server_params, tmp_path):
    if shutil.which("claude") is None:
        pytest.skip("the real `claude` CLI is not on PATH")
    agents = tmp_path / "live-project" / ".claude" / "agents"
    agents.mkdir(parents=True)
    # The body only makes sense as instructions about the user message, so the output
    # differs depending on where `prompt` lands: user message -> "cba".
    (agents / "harness-echo.md").write_text(
        "---\nname: harness-echo\ndescription: Reverses the user message\nmodel: haiku\n---\n"
        "Reply with the user message reversed, character by character, and nothing else.\n",
        encoding="utf-8",
    )

    async def scenario(session):
        is_error, text, started = await _call(
            session,
            "harness_start_agent",
            agent="harness-echo",
            cwd=str(tmp_path / "live-project"),
            model="haiku",
            prompt="abc",
        )
        assert not is_error, text
        return await _poll_until_terminal(session, started["run_id"], budget=240.0)

    final = _run(scenario, live_server_params)
    assert final["state"] == "COMPLETED"
    assert "cba" in final["text"].lower()


def test_live_start_plugin_agent_colon_qualified(live_server_params, tmp_path):
    """R4: the colon-qualified plugin-agent start (R1/R2's fake-CLI mechanism) against
    the real `claude` CLI, with `tools:` proven as an enforced allowlist rather than
    just a JSON-payload field the CLI happens to ignore.

    Skipped unless a human has provisioned `HARNESS_LIVE_PLUGIN_CONFIG_DIR` to point at
    a real `CLAUDE_CONFIG_DIR` that holds:
      - `plugins/installed_plugins.json`, registering a `harness-live-fixture@<any>`
        plugin install;
      - that install's `agents/echo.md`, with frontmatter `model: haiku` and
        `tools: Read`, whose body asks the model to reply with the user message
        reversed, character by character, and nothing else;
      - `settings.json` with `enabledPlugins: {"harness-live-fixture@<any>": true}`;
      - real credentials (this dir's own `.credentials.json` / equivalent) -- the
        isolated `CLAUDE_CONFIG_DIR` `live_server_params` otherwise builds carries
        none, and this test writes no config dir of its own.

    Expected failure modes for a human running this provisioned, against a CLI that
    regresses: the real CLI rejects the `:` in the agent name (run ends FAILED, or
    never reaches "cba"), or the tools allowlist leaks (the non-`mcp__` entries of
    `announced["tools"]` grow beyond `{"Read"}`).
    """
    if shutil.which("claude") is None:
        pytest.skip("the real `claude` CLI is not on PATH")
    live_config_dir = os.environ.get("HARNESS_LIVE_PLUGIN_CONFIG_DIR")
    if not live_config_dir:
        pytest.skip("HARNESS_LIVE_PLUGIN_CONFIG_DIR is not set")

    env = dict(live_server_params.env)
    env["CLAUDE_CONFIG_DIR"] = live_config_dir
    params = live_server_params.model_copy(update={"env": env})

    async def scenario(session):
        listed = await _call(session, "harness_list_agents", cwd=str(tmp_path))
        is_error, text, started = await _call(
            session,
            "harness_start_agent",
            agent="harness-live-fixture:echo",
            cwd=str(tmp_path),
            model="haiku",
            prompt="abc",
        )
        assert not is_error, text
        final = await _poll_until_terminal(session, started["run_id"], budget=240.0)
        inspected = await _call(session, "harness_inspect_run", run_id=started["run_id"])
        return listed, final, inspected

    listed, final, inspected = _run(scenario, params)

    is_error, text, payload = listed
    assert not is_error, text
    by_name = {a["qualified_name"]: a for a in payload["agents"]}
    assert by_name["harness-live-fixture:echo"]["source_scope"] == "plugin"

    assert final["state"] == "COMPLETED"
    assert "cba" in final["text"].lower()

    is_error2, text2, inspected_payload = inspected
    assert not is_error2, text2
    assert inspected_payload["requested"]["tools"] == ["Read"]
    announced_tools = {
        name for name in inspected_payload["announced"]["tools"] if not name.startswith("mcp__")
    }
    assert announced_tools == {"Read"}, (
        "the tools allowlist must be enforced -- the default tool set must not leak"
    )


# --- parent MCP server set announced at first turn, live (#38 R1) -----------------

SLOW_STUB = Path(__file__).parent / "fixtures" / "slow_mcp_stub.py"
_STUB_DELAY_S = "5"


def _run_live(scenario, params, timeout_s=280.0):
    """Like test_mcp_tools._run, but with a caller-chosen deadline instead of that
    helper's hardcoded 60s -- a dispatch across five MCP servers (four 5-second-slow
    stubs plus a real nested harness_plugin process) genuinely takes longer than a
    single quick "Say OK." dispatch does."""

    async def main():
        with anyio.fail_after(timeout_s):
            async with _session(params) as session:
                return await scenario(session)

    return anyio.run(main)


def _real_credentials_path() -> Path:
    base = os.environ.get("CLAUDE_CONFIG_DIR")
    root = Path(base) if base else (Path.home() / ".claude")
    return root / ".credentials.json"


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _stub_server(name: str, log_path: Path) -> dict:
    return {"command": sys.executable, "args": [str(SLOW_STUB), name, _STUB_DELAY_S, str(log_path)]}


@pytest.mark.timeout(600)  # exceeds the repo's global 60s default: up to 3 real-dispatch attempts
def test_live_parent_mcp_servers_at_first_turn():
    """R1 -- the ticket's binding symptom criterion, live: a plugin agent started
    via harness_start_agent with no .seretos/harness.yml entry sees every MCP
    server the parent session has configured (project .mcp.json, user/local,
    enabled plugins' servers, harness included) in its first-turn deferred-tool
    announcement, none still pending, and can call a tool on each.

    Every one of the five sources (user, local, project, the harness-live-fixture
    plugin's own `pslow`, and the agent-harness plugin's own `harness`) is a
    5-second-slow-to-connect stub (or, for `harness`, the real -- and not
    especially fast to import -- harness_plugin server), with
    CLAUDE_CODE_MCP_STARTUP_WAIT_MS/MCP_TIMEOUT left unset, so there is no
    first-turn deadline unless an explicit --mcp-config gives one (plan P2).

    Expected RED reason on current code: the plugin passes no explicit
    --mcp-config for a no-entry launch, so the CLI applies no first-turn wait for
    any of the five stubs; `_first_turn_announcement`'s `pending` set is non-empty
    at the model's first turn (uslow/lslow/jslow/the pslow-plugin form and/or the
    harness dispatch server still connecting) -- falsifying nothing about P2, just
    reproducing the ticket's own symptom on unfixed code.
    """
    if shutil.which("claude") is None:
        pytest.skip("the real `claude` CLI is not on PATH")
    real_credentials = _real_credentials_path()
    if not real_credentials.is_file():
        pytest.skip(f"no real credentials at {real_credentials}; cannot run a live child")

    for var in ("CLAUDE_CODE_MCP_STARTUP_WAIT_MS", "MCP_TIMEOUT"):
        if var in os.environ:
            pytest.skip(f"{var} is set in this environment; R1 requires it unset")

    # A short root, not pytest's own tmp_path fixture: tmp_path nests several
    # directory levels deep (pytest-of-<user>/pytest-<N>/<test-name>-<N>/...), and
    # the real claude CLI encodes a run's *entire absolute cwd path* into the
    # transcript directory name under <CLAUDE_CONFIG_DIR>/projects/ (colons and
    # separators replaced with "-"). Combined with pytest's own nesting, the
    # resulting transcript path reliably exceeds Windows' 260-char MAX_PATH,
    # which silently breaks exact-name glob() matching (observed directly: glob()
    # matched the same file fine via a wildcard suffix or via rglob(), but never
    # via its own full literal name, once the full path crossed ~260 chars) --
    # not a timing race, a path-length ceiling. A short root keeps every path in
    # this test comfortably under that ceiling.
    tmp_path = Path(tempfile.mkdtemp(prefix="ah38-"))
    config_dir = tmp_path / "claude-config"
    project_dir = tmp_path / "live-project"
    project_dir.mkdir(parents=True)
    logs = tmp_path / "logs"
    logs.mkdir()

    config_dir.mkdir(parents=True)
    shutil.copy(real_credentials, config_dir / ".credentials.json")

    # -- user (uslow) + local (lslow), both in <CLAUDE_CONFIG_DIR>/.claude.json ----
    # hasTrustDialogAccepted: local-scope mcpServers is normally added via an
    # interactive `claude mcp add -s local`, which implies the project is already
    # trusted; a hand-crafted entry with no trust flag was verified live to be
    # silently skipped (lslow never appeared anywhere in the transcript, unlike
    # every other source, even though the file entry itself was correct).
    _write_json(
        config_dir / ".claude.json",
        {
            "mcpServers": {"uslow": _stub_server("uslow", logs / "uslow.log")},
            "projects": {
                str(project_dir): {
                    "mcpServers": {"lslow": _stub_server("lslow", logs / "lslow.log")},
                    "hasTrustDialogAccepted": True,
                }
            },
        },
    )

    # -- project (jslow), approved via settings.local.json ------------------------
    _write_json(
        project_dir / ".mcp.json", {"mcpServers": {"jslow": _stub_server("jslow", logs / "jslow.log")}}
    )
    _write_json(
        project_dir / ".claude" / "settings.local.json", {"enableAllProjectMcpServers": True}
    )

    # -- plugin harness-live-fixture@lt: agents/slowcheck.md + mcpServers.pslow ---
    # -- plugin agent-harness@lt: harness = this repo's own harness_plugin server -
    # Both are installed through the real `claude plugin marketplace add` / `install`
    # commands rather than hand-crafted installed_plugins.json/known_marketplaces.json:
    # the real CLI validates plugin registration against a marketplace entry (its
    # `source` recorded in settings.json's extraKnownMarketplaces and
    # plugins/known_marketplaces.json) before it will load a plugin's own manifest --
    # a hand-crafted installed_plugins.json entry with no matching marketplace was
    # verified live to be silently ignored (never mentioned anywhere in the child's
    # transcript). The CLI's own commands populate that state correctly; their exact
    # shape is undocumented and not worth reverse-engineering by hand.
    marketplace_dir = tmp_path / "marketplace"
    fixture_dir = marketplace_dir / "harness-live-fixture"
    (fixture_dir / "agents").mkdir(parents=True)
    (fixture_dir / "agents" / "slowcheck.md").write_text(
        "---\n"
        "name: slowcheck\n"
        "description: R1 live fixture agent -- calls every stub_nonce tool plus harness_list_agents\n"
        "model: haiku\n"
        "---\n"
        "You have access to several MCP tools whose names end in `stub_nonce`, and a tool named "
        "`harness_list_agents`. Call every single tool visible to you whose name matches either of "
        "those two patterns, one at a time, each exactly once. Once every call has returned a "
        "result, reply with the single word DONE and nothing else.\n",
        encoding="utf-8",
    )
    _write_json(
        fixture_dir / ".claude-plugin" / "plugin.json",
        {
            "name": "harness-live-fixture",
            "mcpServers": {"pslow": _stub_server("pslow", logs / "pslow.log")},
        },
    )
    harness_dir = marketplace_dir / "agent-harness"
    _write_json(
        harness_dir / ".claude-plugin" / "plugin.json",
        {
            "name": "agent-harness",
            "mcpServers": {"harness": {"command": sys.executable, "args": ["-m", "harness_plugin"]}},
        },
    )
    _write_json(
        marketplace_dir / ".claude-plugin" / "marketplace.json",
        {
            "name": "lt",
            "owner": {"name": "R1 live fixture"},
            "metadata": {"version": "0.0.0", "description": "R1 live fixture marketplace"},
            "plugins": [
                {
                    "name": "harness-live-fixture",
                    "description": "R1 live fixture plugin",
                    "source": "./harness-live-fixture",
                    "category": "mcp",
                    "version": "0.0.0",
                },
                {
                    "name": "agent-harness",
                    "description": "R1 live fixture: this repo's own harness server",
                    "source": "./agent-harness",
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
    for plugin_name in ("harness-live-fixture", "agent-harness"):
        installed = subprocess.run(
            ["claude", "plugin", "install", f"{plugin_name}@lt", "-y"],
            capture_output=True, text=True, timeout=60, env=setup_env,
        )
        assert installed.returncode == 0, f"install {plugin_name} failed: {installed.stdout} {installed.stderr}"

    env = {**os.environ}
    for var in ("HARNESS_CLAUDE_ARGV",):
        env.pop(var, None)
    env["HARNESS_ARTIFACTS_DIR"] = str(tmp_path / "artifacts")
    env["CLAUDE_CONFIG_DIR"] = str(config_dir)
    env["CLAUDE_PLUGIN_DATA"] = str(tmp_path / "plugin-data")
    env["CLAUDE_CODE_SESSION_ID"] = SESSION_ID
    from conftest import plant_session_file

    plant_session_file(
        tmp_path / "plugin-data",
        SESSION_ID,
        cwd=str(project_dir),
        project_dir=str(project_dir),
        permission_mode="bypassPermissions",
        model="haiku",
    )
    from mcp import StdioServerParameters

    params = StdioServerParameters(command=sys.executable, args=["-m", "harness_plugin"], env=env)

    async def scenario(session):
        is_error, text, started = await _call(
            session, "harness_start_agent", agent="harness-live-fixture:slowcheck", cwd=str(project_dir)
        )
        assert not is_error, text
        final = await _poll_until_terminal(session, started["run_id"], budget=240.0)
        return final

    final = _run_live(scenario, params)
    assert final["state"] == "COMPLETED", final

    # lib_python_harness's own transcript_path resolution (harness.py
    # Harness._resolve_transcript_path, called once from _finalize) was observed to
    # come back None here even though the file exists -- not a timing race, but
    # Windows' 260-char MAX_PATH: exact-literal-name glob() matching silently
    # breaks once the full path crosses that ceiling (verified directly: the very
    # same file matched fine via a wildcard-suffixed pattern or via rglob(), never
    # via glob() with the session_id spelled out in full), and this run's transcript
    # path is long (a deeply-nested tmp root's cwd gets encoded whole into the
    # projects/ subdirectory name). Re-resolved here defensively with a wildcard
    # pattern instead of the exact literal name, from this test's own process.
    transcript_path = final["transcript_path"]
    if transcript_path is None:
        session_id = final["session_id"]
        matches = []
        for _ in range(10):
            matches = [
                p for p in (config_dir / "projects").glob("*/*.jsonl") if session_id in p.name
            ]
            if matches:
                break
            time.sleep(1.0)
        assert matches, (
            f"no transcript file for session {session_id!r} under {config_dir} "
            "even after the run fully completed and a 10s wait"
        )
        transcript_path = str(matches[0])

    found_delta, servers, pending, needs_auth, failed = _first_turn_announcement(transcript_path)
    assert found_delta, (
        "no deferred_tools_delta attachment entry was found before this run's first "
        f"assistant turn (transcript: {transcript_path})"
    )
    expected_min = {"plugin_harness-live-fixture_pslow", "harness", "jslow", "uslow", "lslow"}
    report = (
        f"servers={sorted(servers)} pending={sorted(pending)} "
        f"needs_auth(diagnostic only)={sorted(needs_auth)} failed(diagnostic only)={sorted(failed)}"
    )
    assert expected_min <= servers, f"not every configured server was announced: {report}"
    assert not pending, f"server(s) still pending at the model's first turn: {report}"

    # dedup edge-case coverage: once fixed, never both the rebuilt key *and* the
    # native plugin-qualified form for the same server (mcp__plugin_<p>_<s>__* is
    # how a native plugin MCP tool's name already normalises -- verified live:
    # pre-fix, harness's own native announced form is exactly
    # "plugin_agent-harness_harness", not "harness").
    assert not ({"harness", "plugin_agent-harness_harness"} <= servers), (
        f"both the bare dispatch key and the native plugin-qualified harness form "
        f"must not both appear: {report}"
    )
    assert not ({"plugin_harness-live-fixture_pslow", "pslow"} <= servers), (
        f"both the prefixed and bare forms of pslow must not both appear: {report}"
    )

    # Strengthened per round-1 test-critic tautology::F3: the plan's headline is
    # "each callable", not just "at least one call happened" -- require every one
    # of the four stub logs (uslow/lslow/jslow/pslow) to have logged its own
    # call, not merely a non-empty union across whichever stubs happened to log.
    logged_nonces = set()
    empty_stub_logs = set()
    for stub_name in ("uslow", "lslow", "jslow", "pslow"):
        log = logs / f"{stub_name}.log"
        lines = (
            {line.strip() for line in log.read_text(encoding="utf-8").splitlines() if line.strip()}
            if log.exists()
            else set()
        )
        if not lines:
            empty_stub_logs.add(stub_name)
        logged_nonces |= lines
    assert not empty_stub_logs, (
        f"stub(s) never logged a call -- their stub_nonce tool was never actually "
        f"invoked, even though the server was announced: {sorted(empty_stub_logs)}; {report}"
    )
    transcript_text = Path(transcript_path).read_text(encoding="utf-8")
    tool_result_lines = [
        line for line in transcript_text.splitlines() if '"type": "tool_result"' in line or '"type":"tool_result"' in line
    ]
    tool_result_blob = "\n".join(tool_result_lines) or transcript_text
    missing_nonces = {nonce for nonce in logged_nonces if nonce not in tool_result_blob}
    assert not missing_nonces, f"logged nonce(s) never showed up in a tool_result: {missing_nonces}"

    # The child's own call to the harness dispatch server: a tool_use naming
    # *_harness_list_agents in an assistant turn, followed by a non-error tool_result
    # for that same tool_use_id -- proves the run itself called it, not just this
    # test harness's own separate connection to a different harness_plugin process.
    tool_use_id = None
    tool_result_ok = None
    for line in transcript_text.splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        message = (entry.get("message") or {})
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            if tool_use_id is None and block.get("type") == "tool_use" and (
                block.get("name") or ""
            ).endswith("harness_list_agents"):
                tool_use_id = block.get("id")
            elif tool_use_id and block.get("type") == "tool_result" and block.get("tool_use_id") == tool_use_id:
                tool_result_ok = not block.get("is_error", False)
    assert tool_use_id is not None, f"harness_list_agents was never called by the run: {report}"
    assert tool_result_ok, f"harness_list_agents call returned an error result: {report}"

    shutil.rmtree(tmp_path, ignore_errors=True)


# Hardcoded, independent of _ACCEPTED_VALUES: the minimum this repo already commits
# to elsewhere (test_mcp_tools.py's argv assertions, scripts/build.ps1, the ticket).
# An _ACCEPTED_VALUES that is empty, truncated, or has silently drifted away from
# what the app actually documents must fail this test on its own -- looping only
# over _ACCEPTED_VALUES itself can't detect that, since an empty list makes the loop
# body never run.
#
# Not redundant with the printed-vs-documented reverse check below: that check
# can only ever compare against what --help actually prints, so it structurally
# cannot notice "default" being dropped from _ACCEPTED_VALUES -- --help never
# prints "default" in the first place (see server.py's _ACCEPTED_VALUES comment),
# so there is nothing in `printed_permission_mode` for its absence to violate.
# This fixed minimum is the only thing in this test that would catch that.
_MIN_PERMISSION_MODE = {
    "default", "acceptEdits", "auto", "bypassPermissions", "dontAsk", "manual", "plan",
}


def _help_section(help_text, flag_name):
    """The block of `claude --help` between this flag's own entry and the next
    flag's entry, so a token check can be scoped to the right flag's help instead
    of the whole document -- a token invented for, or only appearing under, an
    unrelated flag (e.g. a documented permission mode that is really the name of a
    different flag like --verbose) must not be able to satisfy the check by simply
    occurring somewhere else in the output."""
    pattern = re.compile(
        rf"^[ \t]*(?:-\w,\s*)?{re.escape(flag_name)}\b.*?"
        rf"(?=^[ \t]*(?:-\w,\s*)?-{{1,2}}[\w-]|\nCommands:|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(help_text)
    assert match, f"{flag_name} not found in `claude --help` output:\n{help_text}"
    return match.group(0)


def _help_choices(section):
    """The quoted tokens inside a `(choices: "a", "b", ...)` parenthetical in a
    `--help` section (e.g. --permission-mode's own block), with the surrounding
    quotes stripped. `_help_section` only slices the flag's own block of `--help`
    text; this turns that text into the actual list of tokens the CLI's own
    parser prints, as data, so the reverse check (every printed choice must be
    documented) has something to iterate instead of a bare substring test."""
    match = re.search(r"\(choices:\s*(.*?)\)", section, re.DOTALL)
    assert match, f"no '(choices: ...)' parenthetical found in section:\n{section}"
    return set(re.findall(r'"([^"]*)"', match.group(1)))


# Flag + real init-event field for a token whose category's --help choices
# don't list it (currently just permission_mode's "default" -- see server.py's
# _ACCEPTED_VALUES comment for why it stays documented anyway: a real run
# with `--permission-mode default` completes and its stream-json init event
# reports the value straight back). There is no generic way to know which
# init-event field mirrors an arbitrary flag, so this map only covers the
# category the fallback probe below is actually used for; extend it before
# relying on the fallback for a new category.
_LIVE_PROBE_FLAG = {"permission_mode": "--permission-mode"}
_LIVE_PROBE_INIT_FIELD = {"permission_mode": "permissionMode"}


def _probe_cli_accepts(category, token):
    """Real accept/reject probe against the live CLI for a `category` token
    that `--help`'s printed choices don't list -- proof of genuine
    acceptance instead of trusting `--help`'s incomplete text. Mirrors the
    manual probe that established "default" is genuinely accepted by
    permission_mode despite --help omitting it: runs a trivial real prompt
    with the token set and inspects the stream-json `init` event. Raises
    AssertionError (with the CLI's own error text/exit code, or the init
    event's mismatched field) on rejection."""
    assert category in _LIVE_PROBE_FLAG, (
        f"no live-probe flag/field mapping for category {category!r} -- add "
        f"one to _LIVE_PROBE_FLAG/_LIVE_PROBE_INIT_FIELD before relying on "
        f"the fallback probe for it"
    )
    flag = _LIVE_PROBE_FLAG[category]
    result = subprocess.run(
        [
            "claude", flag, token,
            "-p", "reply with just the word done",
            "--model", "haiku",
            "--output-format", "stream-json",
            "--verbose",
        ],
        capture_output=True, text=True, timeout=90,
    )
    init_event = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "system" and event.get("subtype") == "init":
            init_event = event
            break
    assert result.returncode == 0 and init_event is not None, (
        f"claude {flag} {token!r} was rejected by the live CLI "
        f"(exit {result.returncode}): stderr={result.stderr!r} stdout={result.stdout!r}"
    )
    field = _LIVE_PROBE_INIT_FIELD[category]
    assert init_event.get(field) == token, (
        f"claude {flag} {token!r} ran but the init event reports "
        f"{field}={init_event.get(field)!r}, not {token!r}: {init_event}"
    )
    return init_event


def test_accepted_values_match_the_cli():
    """R4: the `permission_mode` value list this plugin documents in its tool
    schemas (server.py's _ACCEPTED_VALUES) matches the real CLI's, not invented.
    `effort` and `model` are no longer checked here -- their authority is the
    pinned lib_python_harness's own hard validator, checked offline against the
    lib's own constants by test_mcp_tools.py::test_documented_values_match_lib_
    validator, which needs no live CLI. permission_mode has no lib validator, so
    it stays checked against the live CLI, in both directions since --help's
    printed choices are a lower bound, not an upper bound ("default" is
    genuinely accepted but not printed -- see server.py's _ACCEPTED_VALUES
    comment):
    1. every choice --help prints under --permission-mode (_help_choices,
       parsed as data, not a substring test) must be documented -- and the
       parsed set must be non-empty, so an empty/failed parse can't vacuously
       pass this check;
    2. every documented token --help does *not* print must pass a real
       accept/reject probe against the live CLI (_probe_cli_accepts), instead
       of trusting --help's incomplete text.
    Expected RED before the change: AssertionError naming auto/dontAsk/manual
    as printed by --help but not (yet) in _ACCEPTED_VALUES['permission_mode']."""
    if shutil.which("claude") is None:
        pytest.skip("the real `claude` CLI is not on PATH")

    from harness_plugin.server import _ACCEPTED_VALUES

    help_text = subprocess.run(
        ["claude", "--help"], capture_output=True, text=True, timeout=30
    ).stdout

    documented_permission_mode = set(_ACCEPTED_VALUES["permission_mode"])

    assert _MIN_PERMISSION_MODE <= documented_permission_mode, (
        "_ACCEPTED_VALUES['permission_mode'] must cover at least "
        f"{_MIN_PERMISSION_MODE}, got {documented_permission_mode}"
    )

    permission_mode_section = _help_section(help_text, "--permission-mode")
    printed_permission_mode = _help_choices(permission_mode_section)
    assert printed_permission_mode, (
        f"--permission-mode help text parsed to no choices at all: "
        f"{permission_mode_section!r}"
    )

    undocumented = printed_permission_mode - documented_permission_mode
    assert not undocumented, (
        f"--permission-mode help prints {undocumented} that "
        f"_ACCEPTED_VALUES['permission_mode'] does not document"
    )

    for token in documented_permission_mode:
        if token in printed_permission_mode:
            continue
        # Not among --help's printed choices for --permission-mode (e.g.
        # "default") -- fall back to a real accept/reject probe against the
        # live CLI instead of failing on --help's incomplete text.
        _probe_cli_accepts("permission_mode", token)


# --- MCP-server announcement stability across repeated runs (#27 R2) --------------

# Sub-fields of a `deferred_tools_delta` attachment `_first_turn_announcement` folds
# into diagnostic-only sets, in addition to `pendingMcpServers` (the one the test
# actually asserts empty). Real shape read straight from an actual evidence-run
# transcript (`~/.agent-harness/runs/e3a7c89a-...`, CLI 2.1.278/2.1.280): a transcript
# line `{"type": "attachment", "attachment": {"type": "deferred_tools_delta",
# "addedNames": [...], "removedNames": [...], "pendingMcpServers": [...],
# "needsAuthMcpServers": [...], "failedMcpServers": [...], ...}}`.
_DEFERRED_DELTA_DIAGNOSTIC_FIELDS = {
    "pending": "pendingMcpServers",
    "needs_auth": "needsAuthMcpServers",
    "failed": "failedMcpServers",
}


def _first_turn_announcement(transcript_path):
    """Ground truth for what a run's *first turn* actually told the model about MCP
    servers -- not the CLI's `system/init` event, which announces server *connection*
    status at process start, not the deferred-tools mechanism's own turn-by-turn
    surface to the model (plan #27 "premises verified": a real run's init event
    listed servers `connected` while the transcript's first delta still had them in
    `pendingMcpServers`; tools for those servers only arrived in a *later* delta).
    Reads `transcript_path`'s JSONL up to (not including) its first
    `{"type": "assistant"}` entry, and folds every `{"type": "attachment",
    "attachment": {"type": "deferred_tools_delta", ...}}` entry seen in that window:
    `addedNames` accumulate, `removedNames` retract.

    Returns `(found_delta, servers, pending, needs_auth, failed)`:
    - `found_delta`: whether at least one such entry was seen before the first
      assistant turn. False means this test's ground-truth mechanism itself broke
      (the CLI changed its transcript format) -- a hard failure, never a skip, since a
      skip here would silently hide a real disagreement just as easily as a genuine
      "nothing to test" case.
    - `servers`: the MCP-server identifiers folded out of every surviving
      `mcp__`-prefixed name in `addedNames` (`name.split("__", 2)[1]`, e.g.
      `mcp__plugin_agent-comfy_comfy__list_models` -> `plugin_agent-comfy_comfy`) --
      the actual equality-across-runs assertion target.
    - `pending`/`needs_auth`/`failed`: the union, across every delta in the window, of
      that delta's own `pendingMcpServers`/`needsAuthMcpServers`/`failedMcpServers`.
      `pending` is asserted empty (nothing still connecting when the model's first
      turn happened); the other two are diagnostics only, never asserted.
    """
    added_names: set[str] = set()
    diag: dict[str, set[str]] = {key: set() for key in _DEFERRED_DELTA_DIAGNOSTIC_FIELDS}
    found_delta = False
    with open(transcript_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") == "assistant":
                break
            if entry.get("type") != "attachment":
                continue
            attachment = entry.get("attachment")
            if not isinstance(attachment, dict) or attachment.get("type") != "deferred_tools_delta":
                continue
            found_delta = True
            added_names |= set(attachment.get("addedNames") or [])
            added_names -= set(attachment.get("removedNames") or [])
            for key, raw_key in _DEFERRED_DELTA_DIAGNOSTIC_FIELDS.items():
                diag[key] |= set(attachment.get(raw_key) or [])
    servers = {name.split("__", 2)[1] for name in added_names if name.startswith("mcp__")}
    return found_delta, servers, diag["pending"], diag["needs_auth"], diag["failed"]


def test_live_mcp_announcement_stable_across_runs(live_server_params, tmp_path):
    """R2 (#27): the reported symptom is that the deferred-tool list a harness run is
    told about at startup contains only part of the real MCP-server set, and which
    part varies run to run. There is no production fix in this repo for it -- the
    announcement is assembled entirely inside the child `claude` CLI process, outside
    this plugin's control (plan "premises verified") -- so this is a read-only probe,
    not a regression test with a code fix behind it: three identical dispatches of the
    same tiny agent must announce the same, complete MCP-server set every time, with
    nothing still `pending` by the time the model's first turn happened. Expected to
    fail today against a CLI actually exhibiting the symptom -- that failure IS the
    evidence this ticket exists to capture, not a defect in the test itself."""
    if shutil.which("claude") is None:
        pytest.skip("the real `claude` CLI is not on PATH")

    agents = tmp_path / "live-project" / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / "harness-mcp-announce.md").write_text(
        "---\nname: harness-mcp-announce\n"
        "description: Minimal agent for the MCP-announcement stability probe\n"
        "model: haiku\n---\n"
        "Reply with OK and nothing else.\n",
        encoding="utf-8",
    )

    per_run = []
    for _ in range(3):

        async def scenario(session):
            is_error, text, started = await _call(
                session,
                "harness_start_agent",
                agent="harness-mcp-announce",
                cwd=str(tmp_path / "live-project"),
                model="haiku",
            )
            assert not is_error, text
            final = await _poll_until_terminal(session, started["run_id"], budget=240.0)
            inspected = await _call(session, "harness_inspect_run", run_id=started["run_id"])
            return final, inspected

        final, (is_error, text, inspected) = _run(scenario, live_server_params)
        assert final["state"] == "COMPLETED", final
        assert not is_error, text

        found_delta, servers, pending, needs_auth, failed = _first_turn_announcement(
            final["transcript_path"]
        )
        # (a) -- a missing delta means the ground-truth mechanism itself broke; fail
        # loudly rather than let a format change masquerade as "nothing pending".
        assert found_delta, (
            "no deferred_tools_delta attachment entry was found before this run's "
            f"first assistant turn (transcript: {final['transcript_path']}); the CLI's "
            "transcript format appears to have changed -- this test can no longer tell "
            "'announced' from 'never looked'"
        )
        per_run.append(
            {
                "servers": servers,
                "pending": pending,
                "needs_auth": needs_auth,
                "failed": failed,
                "init_mcp_servers": inspected["announced"]["mcp_servers"],
                "init_tools": inspected["announced"]["tools"],
            }
        )

    report = "\n".join(
        f"run {i}: servers={sorted(r['servers'])} pending={sorted(r['pending'])} "
        f"needs_auth(diagnostic only)={sorted(r['needs_auth'])} "
        f"failed(diagnostic only)={sorted(r['failed'])} "
        f"init_mcp_servers(diagnostic only, init event != announcement)={r['init_mcp_servers']} "
        f"init_tools(diagnostic only)={r['init_tools']}"
        for i, r in enumerate(per_run)
    )

    # (b) -- the same dispatch, repeated identically three times, must be told about
    # the same MCP-server set every time. This is the ticket's own symptom.
    server_sets = [r["servers"] for r in per_run]
    assert server_sets[0] == server_sets[1] == server_sets[2], (
        f"the announced MCP-server set varied across 3 identical runs:\n{report}"
    )

    # (c) -- nothing still connecting by the time the model's first turn happened, in
    # any run: a stable-but-incomplete announcement would pass (b) while still being
    # the symptom.
    for i, r in enumerate(per_run):
        assert not r["pending"], (
            f"run {i} still had server(s) pending when the model's first turn "
            f"happened -- the announcement was incomplete, not just late:\n{report}"
        )

    # Only after every assertion above has actually passed: an environment with no MCP
    # servers configured at all would make (b)/(c) hold vacuously (equal empty sets,
    # empty pending) without ever exercising what this test exists to check. This must
    # be decided from the same ground truth as (b)/(c) -- the transcript-derived
    # `servers`/`needs_auth`/`failed` sets -- not from the init event's
    # `init_mcp_servers`/`init_tools` diagnostics: those are explicitly not the
    # announcement (see "premises verified" / `_first_turn_announcement`'s docstring),
    # so a run where the init event lists servers but the transcript's first-turn fold
    # never mentions any (nothing added, nothing pending, nothing needing auth, nothing
    # failed) would otherwise skip the vacuity check and let (b)/(c) pass on three
    # empty sets that never compared anything.
    if all(not (r["servers"] or r["needs_auth"] or r["failed"]) for r in per_run):
        pytest.skip(f"held vacuously: no MCP server was announced in any run:\n{report}")


# --- launched-agent identity visible to a real plugin Stop hook, live (#51 R4) -----


@pytest.mark.timeout(300)
def test_live_launched_agent_visible_to_plugin_stop_hook():
    """R4: in a real run of a plugin agent, the plugin's own Stop hook -- running
    inside that same child `claude` process -- reads HARNESS_LAUNCHED_AGENT and
    sees the qualified name, colon included. This is the ticket's actual consumer:
    a hook script (not this test harness) telling which agent its own process is
    running as. Self-provisions like test_live_parent_mcp_servers_at_first_turn: a
    short tempfile.mkdtemp root, a copied .credentials.json (skipped if absent or
    `claude` is not on PATH), and a real `claude plugin marketplace add` / `install
    harness-id-fixture@lt -y` under an isolated CLAUDE_CONFIG_DIR.

    There is deliberately no SessionStart fallback here: the ticket's consumer is a
    Stop-event guard, so if Stop does not fire in `-p` mode this test fails outright
    rather than silently swapping to a different event.

    Expected RED reason: seen.json's "agent" field is None (nothing sets
    HARNESS_LAUNCHED_AGENT in a real child's environment today)."""
    if shutil.which("claude") is None:
        pytest.skip("the real `claude` CLI is not on PATH")
    real_credentials = _real_credentials_path()
    if not real_credentials.is_file():
        pytest.skip(f"no real credentials at {real_credentials}; cannot run a live child")

    tmp_path = Path(tempfile.mkdtemp(prefix="ah51-"))
    config_dir = tmp_path / "claude-config"
    project_dir = tmp_path / "live-project"
    project_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    shutil.copy(real_credentials, config_dir / ".credentials.json")

    marketplace_dir = tmp_path / "marketplace"
    fixture_dir = marketplace_dir / "harness-id-fixture"
    (fixture_dir / "agents").mkdir(parents=True)
    (fixture_dir / "agents" / "idcheck.md").write_text(
        "---\n"
        "name: idcheck\n"
        "description: R4 live fixture agent -- always replies OK\n"
        "model: haiku\n"
        "---\n"
        "Reply OK.\n",
        encoding="utf-8",
    )

    seen_path = tmp_path / "seen.json"
    hook_path = fixture_dir / "hook.py"
    hook_path.write_text(
        "import json, os, sys\n"
        "payload = json.loads(sys.stdin.read())\n"
        "out = {'event': payload.get('hook_event_name'), "
        "'agent': os.environ.get('HARNESS_LAUNCHED_AGENT')}\n"
        "with open(sys.argv[1], 'w', encoding='utf-8') as fh:\n"
        "    json.dump(out, fh)\n",
        encoding="utf-8",
    )
    _write_json(
        fixture_dir / "hooks" / "hooks.json",
        {
            "hooks": {
                "Stop": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": (
                                    f'"{sys.executable}" "{hook_path}" "{seen_path}"'
                                ),
                                "timeout": 10,
                            }
                        ]
                    }
                ]
            }
        },
    )
    _write_json(
        fixture_dir / ".claude-plugin" / "plugin.json",
        {"name": "harness-id-fixture", "hooks": "./hooks/hooks.json"},
    )
    _write_json(
        marketplace_dir / ".claude-plugin" / "marketplace.json",
        {
            "name": "lt",
            "owner": {"name": "R4 live fixture"},
            "metadata": {"version": "0.0.0", "description": "R4 live fixture marketplace"},
            "plugins": [
                {
                    "name": "harness-id-fixture",
                    "description": "R4 live fixture: Stop hook reads HARNESS_LAUNCHED_AGENT",
                    "source": "./harness-id-fixture",
                    "category": "other",
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
        ["claude", "plugin", "install", "harness-id-fixture@lt", "-y"],
        capture_output=True, text=True, timeout=60, env=setup_env,
    )
    assert installed.returncode == 0, f"install failed: {installed.stdout} {installed.stderr}"

    env = {**os.environ}
    env.pop("HARNESS_CLAUDE_ARGV", None)
    env["HARNESS_ARTIFACTS_DIR"] = str(tmp_path / "artifacts")
    env["CLAUDE_CONFIG_DIR"] = str(config_dir)
    env["CLAUDE_PLUGIN_DATA"] = str(tmp_path / "plugin-data")
    env["CLAUDE_CODE_SESSION_ID"] = SESSION_ID
    from conftest import plant_session_file

    plant_session_file(
        tmp_path / "plugin-data",
        SESSION_ID,
        cwd=str(project_dir),
        project_dir=str(project_dir),
        permission_mode="bypassPermissions",
        model="haiku",
    )
    from mcp import StdioServerParameters

    params = StdioServerParameters(command=sys.executable, args=["-m", "harness_plugin"], env=env)

    async def scenario(session):
        is_error, text, started = await _call(
            session,
            "harness_start_agent",
            agent="harness-id-fixture:idcheck",
            cwd=str(project_dir),
        )
        assert not is_error, text
        return await _poll_until_terminal(session, started["run_id"], budget=240.0)

    final = _run_live(scenario, params, timeout_s=280.0)
    assert final["state"] == "COMPLETED", final
    assert seen_path.exists(), (
        "the fixture plugin's own Stop hook (hook.py) was never invoked -- Stop did "
        "not fire in -p mode"
    )
    seen = json.loads(seen_path.read_text(encoding="utf-8"))
    assert seen["event"] == "Stop"
    assert seen["agent"] == "harness-id-fixture:idcheck"

    shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.mark.timeout(300)  # exceeds the repo's global 60s default: real 240s poll budget below
def test_live_wait_run_timeout_keeps_run_alive(live_server_params):
    if shutil.which("claude") is None:
        pytest.skip("the real `claude` CLI is not on PATH")

    async def scenario(session):
        is_error, text, started = await _call(
            session,
            "harness_start_prompt",
            prompt="Count from 1 to 200, one number per line, then reply DONE.",
            model="haiku",
        )
        assert not is_error, text
        run_id = started["run_id"]
        waited = await _call(session, "harness_wait_run", run_id=run_id, timeout_seconds=3)
        final = await _poll_until_terminal(session, run_id, budget=240.0)
        return run_id, waited, final

    run_id, waited, final = _run(scenario, live_server_params)
    assert waited[0] is False, waited[1]
    if waited[2]["state"] == "RUNNING":
        assert run_id in waited[2]["next_step"]
        assert "harness wait" in waited[2]["next_step"]
        assert waited[2]["duration_s"] > 0
    assert waited[2]["state"] != "CANCELLED"
    assert final["state"] == "COMPLETED"


# --- native subagent dispatch denied at the tool-call boundary, live (#52 R4) ------


def _tool_result_text(block: dict) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


@pytest.mark.timeout(300)
def test_live_native_subagent_dispatch_denied():
    """R4: a real `claude -p` session, with the plugin's hooks installed via a
    fixture plugin, refuses Claude Code's native Agent/Task subagent dispatch
    at the tool-call boundary -- the ticket's actual consumer.

    Self-provisions like test_live_launched_agent_visible_to_plugin_stop_hook:
    a short tempfile.mkdtemp root, a copied .credentials.json (skipped if
    absent or `claude` is not on PATH), and a real `claude plugin marketplace
    add` / `install harness-deny-fixture@lt -y` under an isolated
    CLAUDE_CONFIG_DIR. The fixture plugin carries this repo's hooks/hooks.json
    verbatim plus a fixture `bin/harness` sh script
    (`exec "<sys.executable>" -m harness_plugin "$@"`) standing in for the
    frozen binary -- accepted as a proceed per plan-critic round 2 (R3 already
    covers the frozen-binary path separately, in bash).

    Expected RED reason: today the Agent/Task tool_use's tool_result is a
    normal (non-error) subagent result, not the hook's denial. If the model
    never emits the tool_use at all, this test fails loudly with the full
    stream-json dump, never skips."""
    if shutil.which("claude") is None:
        pytest.skip("the real `claude` CLI is not on PATH")
    real_credentials = _real_credentials_path()
    if not real_credentials.is_file():
        pytest.skip(f"no real credentials at {real_credentials}; cannot run a live child")

    tmp_path = Path(tempfile.mkdtemp(prefix="ah52-"))
    config_dir = tmp_path / "claude-config"
    project_dir = tmp_path / "live-project"
    project_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    shutil.copy(real_credentials, config_dir / ".credentials.json")

    marketplace_dir = tmp_path / "marketplace"
    fixture_dir = marketplace_dir / "harness-deny-fixture"
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
        {"name": "harness-deny-fixture", "hooks": "./hooks/hooks.json"},
    )
    _write_json(
        marketplace_dir / ".claude-plugin" / "marketplace.json",
        {
            "name": "lt",
            "owner": {"name": "R4 live fixture"},
            "metadata": {"version": "0.0.0", "description": "R4 live fixture marketplace"},
            "plugins": [
                {
                    "name": "harness-deny-fixture",
                    "description": "R4 live fixture: this repo's own hooks/hooks.json verbatim",
                    "source": "./harness-deny-fixture",
                    "category": "other",
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
        ["claude", "plugin", "install", "harness-deny-fixture@lt", "-y"],
        capture_output=True, text=True, timeout=60, env=setup_env,
    )
    assert installed.returncode == 0, f"install failed: {installed.stdout} {installed.stderr}"

    run_env = {**os.environ, "CLAUDE_CONFIG_DIR": str(config_dir)}
    proc = subprocess.run(
        [
            "claude", "-p",
            "Use the Agent tool to launch the general-purpose subagent with the prompt "
            "'say OK'. Do nothing else.",
            "--model", "haiku",
            "--permission-mode", "bypassPermissions",
            "--output-format", "stream-json",
            "--verbose",
        ],
        capture_output=True, text=True, timeout=280, env=run_env, cwd=str(project_dir),
    )

    tool_use_id = None
    tool_result_block = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = entry.get("message") or {}
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            if tool_use_id is None and block.get("type") == "tool_use" and block.get("name") in ("Agent", "Task"):
                tool_use_id = block.get("id")
            elif (
                tool_use_id
                and tool_result_block is None
                and block.get("type") == "tool_result"
                and block.get("tool_use_id") == tool_use_id
            ):
                tool_result_block = block

    assert tool_use_id is not None, (
        "the model never emitted a native Agent/Task tool_use; full stream-json "
        f"dump:\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    assert tool_result_block is not None, (
        f"no tool_result found for the Agent/Task tool_use {tool_use_id!r}; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert tool_result_block.get("is_error") is True, (
        f"expected the native subagent dispatch to be denied (is_error), got "
        f"{tool_result_block!r}"
    )
    result_text = _tool_result_text(tool_result_block)
    assert "harness_start_agent" in result_text, (
        f"deny message missing harness_start_agent: {result_text!r}"
    )

    shutil.rmtree(tmp_path, ignore_errors=True)


# --- #62 R5: a real tool_response shape is tracked, live ---------------------


@pytest.mark.timeout(300)
def test_live_stop_hook_tracks_started_run():
    """R5: a real `claude -p` session with this plugin's own hooks.json AND
    its own MCP server installed (fixture template:
    `test_live_native_subagent_dispatch_denied`, extended with `mcpServers`
    so the session can actually call `harness_start_prompt`) calls
    `harness_start_prompt`, `harness_wait_run`s the result, then finishes.
    The PostToolUse hook (once implemented) must have written a
    `tracked-runs/` marker for the run_id the harness server's own run store
    (HARNESS_ARTIFACTS_DIR, pinned to this test's own tmp dir) recorded --
    proving premise 1 (the real `tool_response` shape) live, since it
    cannot be checked offline (plan "Premises verified" #1).

    The Stop block itself is not asserted here -- R1 in test_hook.py already
    covers Stop's exit-2 behaviour against a synthetic marker + record; this
    test only proves PostToolUse actually produces a real marker from a
    real tool_response, in a real session.

    Where CLAUDE_PLUGIN_DATA actually lands for this fixture plugin is
    computed by the real CLI itself (per-plugin, under CLAUDE_CONFIG_DIR --
    see host_context.py's own `parent_mcp_servers` docstring, which mirrors
    the same convention for a dispatched child); this test does not
    hardcode that path but searches for a `tracked-runs/` directory
    anywhere under `config_dir`.

    Expected RED reason: no PostToolUse hook exists yet, so no
    `tracked-runs/` directory is ever created anywhere under `config_dir`."""
    if shutil.which("claude") is None:
        pytest.skip("the real `claude` CLI is not on PATH")
    real_credentials = _real_credentials_path()
    if not real_credentials.is_file():
        pytest.skip(f"no real credentials at {real_credentials}; cannot run a live child")

    tmp_path = Path(tempfile.mkdtemp(prefix="ah62-"))
    config_dir = tmp_path / "claude-config"
    project_dir = tmp_path / "live-project"
    project_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    shutil.copy(real_credentials, config_dir / ".credentials.json")

    artifacts_dir = tmp_path / "artifacts"

    marketplace_dir = tmp_path / "marketplace"
    fixture_dir = marketplace_dir / "harness-stop-fixture"
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
            "name": "harness-stop-fixture",
            "hooks": "./hooks/hooks.json",
            "mcpServers": {
                "harness": {
                    "command": sys.executable,
                    "args": ["-m", "harness_plugin"],
                    "env": {"HARNESS_ARTIFACTS_DIR": str(artifacts_dir)},
                }
            },
        },
    )
    _write_json(
        marketplace_dir / ".claude-plugin" / "marketplace.json",
        {
            "name": "lt",
            "owner": {"name": "R5 live fixture"},
            "metadata": {"version": "0.0.0", "description": "R5 live fixture marketplace"},
            "plugins": [
                {
                    "name": "harness-stop-fixture",
                    "description": "R5 live fixture: this repo's own hooks.json + MCP server",
                    "source": "./harness-stop-fixture",
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
        ["claude", "plugin", "install", "harness-stop-fixture@lt", "-y"],
        capture_output=True, text=True, timeout=60, env=setup_env,
    )
    assert installed.returncode == 0, f"install failed: {installed.stdout} {installed.stderr}"

    run_env = {**os.environ, "CLAUDE_CONFIG_DIR": str(config_dir)}
    proc = subprocess.run(
        [
            "claude", "-p",
            "Use the harness_start_prompt tool with the prompt 'reply ok', then call "
            "harness_wait_run on the returned run_id repeatedly until it reports a "
            "terminal state, then reply DONE. Do nothing else.",
            "--model", "haiku",
            "--permission-mode", "bypassPermissions",
            "--output-format", "stream-json",
            "--verbose",
        ],
        capture_output=True, text=True, timeout=280, env=run_env, cwd=str(project_dir),
    )
    assert proc.returncode == 0, (
        f"claude -p exited {proc.returncode}; stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )

    started_run_ids = (
        {record_dir.name for record_dir in artifacts_dir.iterdir() if (record_dir / "record.json").is_file()}
        if artifacts_dir.is_dir()
        else set()
    )
    assert started_run_ids, (
        f"harness_start_prompt never created a run record under {artifacts_dir}; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )

    tracked_run_ids: set[str] = set()
    for tracked_dir in config_dir.rglob("tracked-runs"):
        if not tracked_dir.is_dir():
            continue
        for session_dir in tracked_dir.iterdir():
            if session_dir.is_dir():
                tracked_run_ids |= {p.name for p in session_dir.iterdir()}
    assert started_run_ids & tracked_run_ids, (
        f"no tracked-runs/ marker for any started run {started_run_ids}; "
        f"found tracked run(s): {tracked_run_ids}; searched under {config_dir}"
    )

    shutil.rmtree(tmp_path, ignore_errors=True)


# --- a real CLEAN run's child process exits after COMPLETED, live (#68 R2) --------


@pytest.mark.timeout(300)
def test_live_clean_prompt_child_exits(live_server_params, tmp_path):
    """R2 (#68): a real CLEAN `claude -p` run (haiku, "Reply OK") reaches
    COMPLETED through `harness_poll_run` -- polled every 2s, 120s cap -- and
    its OS child process is gone afterwards. This is the ticket's own
    symptom, live: R1 (tests/test_lingering_child.py) is the fast,
    deterministic proof that the grace-kill mechanism works at all; this test
    is the AC-required evidence that it actually fires against a real CLI.

    Expected RED reason: fails only on a CLI build that lingers past its own
    terminal `result` event (the ticket's v2.1.283 does) -- there,
    `harness_poll_run`'s unfixed `poll()` has no grace-kill, so it keeps
    reporting RUNNING until this test's own 120s cap runs out. A CLI that
    exits promptly passes both before and after the fix -- falsifying
    nothing about the plugin-side fix itself, just reproducing (or not) the
    live CLI-side symptom this ticket exists to route around.

    Expected GREEN outcome: COMPLETED, pid gone. It does not assert how fast
    the real CLI actually exits (plan R2)."""
    if shutil.which("claude") is None:
        pytest.skip("the real `claude` CLI is not on PATH")

    from lib_python_harness import FileRunStore
    from lib_python_harness.runtime.process import _pid_status

    artifacts_dir = tmp_path / "artifacts"

    async def scenario(session):
        is_error, text, started = await _call(
            session, "harness_start_prompt", prompt="Reply OK", model="haiku"
        )
        assert not is_error, text
        run_id = started["run_id"]
        deadline = time.monotonic() + 120.0
        payload = None
        while True:
            is_error, text, payload = await _call(session, "harness_poll_run", run_id=run_id)
            assert not is_error, text
            if payload["state"] in TERMINAL:
                break
            assert time.monotonic() < deadline, f"run never finished: {payload}"
            await anyio.sleep(2.0)
        if payload["state"] == "RUNNING":
            await _call(session, "harness_stop_run", run_id=run_id)
        return run_id, payload

    run_id, payload = _run(scenario, live_server_params)
    assert payload["state"] == "COMPLETED", payload

    record = FileRunStore(str(artifacts_dir)).get(run_id)
    assert record is not None, f"no record.json for {run_id} under {artifacts_dir}"
    pid = record.get("pid")
    assert pid is not None, f"record for {run_id} has no pid: {record}"
    assert _pid_status(pid, record.get("start_time")) is False, (
        "the run's OS child process is still alive after COMPLETED was reported"
    )


# --- Stop finalizes a normally-exited real child held by another process (#72) -----


@pytest.mark.timeout(300)
def test_live_stop_hook_completes_exited_child(live_server_params, tmp_path):
    """#72 (live variant of tests/test_lingering_child.py's
    test_stop_hook_completes_exited_child_held_by_other_process): a real
    `claude -p` child (haiku, "Reply OK") exits on its own while the MCP server
    keeps its `Popen`; the real Stop hook must then finalize the run and exit 0
    within BUDGET of its own start, not block for its wait limit.

    Expected RED reason (lib_python_harness v0.0.10): Stop never finalizes and
    exits 2 after its wait limit. Opt-in evidence only; the default suite's
    fake-CLI test is the deterministic proof."""
    if shutil.which("claude") is None:
        pytest.skip("the real `claude` CLI is not on PATH")
    from lib_python_harness import RunState
    from test_lingering_child import BUDGET, _cleanup, _record, _stop_after_exited_child

    artifacts_dir = tmp_path / "artifacts"
    plugin_data = tmp_path / "plugin-data"
    run_id = None

    async def scenario(session):
        return await _stop_after_exited_child(
            session, artifacts_dir, plugin_data, "sess-72-live", "Reply OK", "haiku", 120
        )

    try:
        run_id, stop, stop_elapsed = _run(scenario, live_server_params)
        assert stop.returncode == 0, (
            f"expected Stop to finalize the exited run within {BUDGET}s; "
            f"stdout={stop.stdout!r} stderr={stop.stderr!r}"
        )
        assert stop_elapsed <= BUDGET, f"Stop took {stop_elapsed:.2f}s, budget {BUDGET}s"
        record = _record(artifacts_dir, run_id)
        assert record is not None and record.get("state") == RunState.COMPLETED, record
    finally:
        if run_id is not None:
            _cleanup(live_server_params, artifacts_dir, run_id)
