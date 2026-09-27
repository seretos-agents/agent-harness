"""#53: hooks/hooks.json's two hook commands and both MCP manifests
(`.claude-plugin/plugin.json` inline, root `.mcp.json`) all hardcode the
extensionless `bin/harness` path. On Windows, Git Bash executes whatever file
sits at that path literally; today that is the Linux ELF binary shipped
alongside `bin/harness.exe`, and Git Bash fails immediately with "Exec format
error". The fix (out of scope for this module) turns `bin/harness` into a
committed POSIX dispatcher and renames the Linux binary to `bin/harness-linux`
-- this module proves the fix by running the verbatim launch-point commands
against a `bin/` laid out exactly as the release ships it.

R1 -- each hooks.json command, read verbatim, run through a real bash.
R2 -- each MCP manifest's command, read verbatim, completes a real MCP
      initialize + tools/list handshake through a real Node spawn (Claude
      Code's own runtime/libuv path resolution).

Both need a real frozen binary: there is none in a bare dev checkout or in
the plain `pytest` CI job, only in test.yml's `build` job after
`pwsh -File scripts/build.ps1`. The whole module is skipped, loudly, when
`HARNESS_BIN` is unset -- never silently "passing" against nothing built.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parents[1]
RELEASE_YML = REPO / ".github" / "workflows" / "release.yml"

HARNESS_BIN = os.environ.get("HARNESS_BIN")

pytestmark = pytest.mark.skipif(
    not HARNESS_BIN,
    reason=(
        "HARNESS_BIN is not set. This module needs a real frozen binary: run "
        "`pwsh -File scripts/build.ps1` first, then set HARNESS_BIN to the "
        "resulting bin/harness[.exe] before running pytest. Skipped rather "
        "than silently reporting success against nothing built."
    ),
)


def _execs_from_release_yml() -> set[str]:
    """Regex-extract the `EXECS = {...}` set release.yml's assembly step
    embeds, the same technique test_plugin_manifests.py's `_staging_script`
    uses to avoid a hand-copied literal that could drift from the real
    workflow file."""
    text = RELEASE_YML.read_text(encoding="utf-8")
    match = re.search(r"EXECS\s*=\s*\{([^}]*)\}", text)
    assert match, "release.yml no longer defines EXECS -- update this extraction"
    names = re.findall(r'"([^"]+)"', match.group(1))
    assert names, "EXECS set parsed empty from release.yml"
    return set(names)


def _decoy_bytes(fname: str) -> bytes:
    """A foreign-format executable stub for the file this build did NOT
    produce: PE (`MZ`) magic for a `.exe` name, ELF magic otherwise -- so a
    bash/Node launch against the wrong bin/ entry fails for a genuine
    OS-format reason, the same way today's single-binary bin/ does."""
    if fname.endswith(".exe"):
        return b"MZ" + b"\x00" * 62
    return b"\x7fELF" + b"\x00" * 60


def _git_bash() -> str | None:
    """The same Git Bash lookup Claude Code uses on Windows: an explicit
    CLAUDE_CODE_GIT_BASH_PATH override, else the bash.exe shipped next to
    git.exe -- never System32's WSL bash.exe, which would pass this module
    for the wrong reason.

    git.exe's install depth under the Git root varies by how PATH found it
    (`Git\\cmd\\git.exe` vs. `Git\\mingw64\\bin\\git.exe`), so this walks
    every ancestor directory of the resolved git.exe rather than assuming a
    fixed number of parent levels -- a fixed `.parent.parent` guess found
    `Git\\cmd\\git.exe`'s root correctly but missed `Git\\mingw64\\bin\\git.exe`
    entirely on a dev machine where mingw64's copy is earlier on PATH."""
    override = os.environ.get("CLAUDE_CODE_GIT_BASH_PATH")
    if override and Path(override).is_file():
        return override
    git_exe = shutil.which("git")
    if not git_exe:
        return None
    git_exe_path = Path(git_exe).resolve()
    for base in (git_exe_path.parent, *git_exe_path.parents):
        for candidate in (base / "bin" / "bash.exe", base / "usr" / "bin" / "bash.exe"):
            if candidate.is_file():
                return str(candidate)
    return None


@pytest.fixture(scope="module")
def bash_exe() -> str:
    if sys.platform == "win32":
        found = _git_bash()
        if not found:
            pytest.skip(
                "no Git Bash found on this Windows host "
                "(CLAUDE_CODE_GIT_BASH_PATH unset, no git.exe on PATH)"
            )
        return found
    found = shutil.which("bash")
    if not found:
        pytest.skip("no bash on PATH")
    return found


@pytest.fixture
def plugin_root(tmp_path) -> Path:
    """A throwaway plugin install laid out exactly as the release ships it:
    `.claude-plugin/`, `hooks/` and `.mcp.json` copied verbatim from the repo,
    and `bin/` holding every real file from HARNESS_BIN's directory (the
    native binary, plus the committed dispatcher once it exists) plus a
    foreign-format decoy for every release.yml EXECS name still missing --
    the real binary for this OS sitting next to a file for the other OS, the
    same shape the orphan release branch's zip has."""
    root = tmp_path / "plugin"
    shutil.copytree(REPO / ".claude-plugin", root / ".claude-plugin")
    shutil.copytree(REPO / "hooks", root / "hooks")
    shutil.copy2(REPO / ".mcp.json", root / ".mcp.json")

    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    src_bin = Path(HARNESS_BIN).resolve().parent
    for f in src_bin.iterdir():
        if f.is_file():
            shutil.copy2(f, bin_dir / f.name)
            if sys.platform != "win32":
                bin_dir.joinpath(f.name).chmod(0o755)

    present = {p.name for p in bin_dir.iterdir()}
    for name in _execs_from_release_yml():
        rel = Path(name)
        assert rel.parts[0] == "bin", f"unexpected EXECS entry outside bin/: {name}"
        fname = rel.name
        if fname in present:
            continue
        decoy = bin_dir / fname
        decoy.write_bytes(_decoy_bytes(fname))
        decoy.chmod(0o755)
    return root


def _hook_entries() -> list[tuple[str, str]]:
    """[(event_name, command), ...] for every hooks.json hook entry, read
    verbatim from the repo file (not hand-copied)."""
    data = json.loads((REPO / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    entries = []
    for event_name, groups in data["hooks"].items():
        for group in groups:
            for hook in group["hooks"]:
                entries.append((event_name, hook["command"]))
    return entries


_HOOK_ENTRIES = _hook_entries()


# --- R1: hooks.json commands run the right binary under bash ----------------


@pytest.mark.parametrize(
    "event_name,command", _HOOK_ENTRIES, ids=[e for e, _ in _HOOK_ENTRIES]
)
def test_hook_commands_via_bash_write_session_file(plugin_root, bash_exe, tmp_path, event_name, command):
    """R1: each hooks.json command, read verbatim and run through bash with
    ${CLAUDE_PLUGIN_ROOT} expanded (bash itself expands it, matching how
    Claude Code's hook runner invokes the command), must exit 0 and write a
    session file -- not fail against a same-named file for the wrong OS.

    Expected RED reason (windows-latest, pre-fix): release.yml's EXECS today
    is {bin/harness, bin/harness.exe}, so the PE decoy for `bin/harness.exe`
    is real (HARNESS_BIN) but `bin/harness` itself is the ELF decoy written
    by this fixture for the *other* EXECS entry. Bash reports "cannot
    execute binary file: Exec format error" (exit 126) and no session file
    is written -- the reported symptom. On ubuntu this already passes
    (Linux was never broken).
    """
    plugin_data = tmp_path / "plugin-data"
    plugin_data.mkdir()
    session_id = f"{event_name}-1"
    stdin_payload = json.dumps(
        {
            "session_id": session_id,
            "cwd": str(plugin_root),
            "permission_mode": "acceptEdits",
            "hook_event_name": event_name,
        }
    )

    env = {
        **os.environ,
        "CLAUDE_PLUGIN_ROOT": plugin_root.as_posix(),
        "CLAUDE_PLUGIN_DATA": str(plugin_data),
    }
    result = subprocess.run(
        [bash_exe, "-c", command],
        input=stdin_payload,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )

    session_file = plugin_data / "sessions" / f"{session_id}.json"
    assert result.returncode == 0, (
        f"bash exited {result.returncode} running {command!r}; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert session_file.is_file(), (
        f"no session file after running {command!r} (exit 0); "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "acceptEdits" in session_file.read_text(encoding="utf-8")


# test_hook_commands_share_one_launch_path removed (test-critic round 1,
# tautology::F1): it only asserted hooks.json's static JSON shape (two
# entries, identical command strings), which the plan declares unchanged by
# this fix -- no implementation, correct or wrong, could move that
# assertion. test_hook_commands_via_bash_write_session_file above already
# parametrizes over both the SessionStart and PreToolUse entries and proves
# each one, run verbatim through bash, exits 0 and writes its session file
# -- that is what actually demonstrates both hook entries share one working
# launch path, RED today for the reported Exec format error and GREEN once
# the dispatcher exists.


# --- R2: MCP launch commands complete a real handshake -----------------------


_NODE_SPAWN_SCRIPT = """
const { spawn } = require('child_process');
// `node -e <script> a b c` yields process.argv = [nodeExePath, a, b, c] --
// unlike a script file invocation, -e's own source never occupies an argv
// slot, so only ONE leading element (the node executable) is skipped here.
const [, cwd, command, ...args] = process.argv;
const child = spawn(command, args, { stdio: 'inherit', cwd });
child.on('exit', (code, signal) => process.exit(code === null ? 1 : code));
child.on('error', (err) => { process.stderr.write(String(err) + '\\n'); process.exit(1); });
"""


def _mcp_launch_specs(plugin_root: Path):
    """[(id, command, args, cwd)]: `.claude-plugin/plugin.json`'s inline
    `mcpServers.harness` with `${CLAUDE_PLUGIN_ROOT}` substituted (the only
    substitution Claude Code performs), and root `.mcp.json`'s `command`/
    `args`/`cwd`, both read verbatim -- no other resolution applied here, so
    the real Node spawn below is what actually resolves the extensionless
    name, not this test."""
    specs = []

    claude_manifest = json.loads(
        (plugin_root / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
    )
    claude_server = claude_manifest["mcpServers"]["harness"]
    claude_command = claude_server["command"].replace("${CLAUDE_PLUGIN_ROOT}", str(plugin_root))
    specs.append(("claude-plugin", claude_command, claude_server.get("args", []), str(plugin_root)))

    mcp_manifest = json.loads((plugin_root / ".mcp.json").read_text(encoding="utf-8"))
    mcp_server = mcp_manifest["mcpServers"]["harness"]
    mcp_cwd = str((plugin_root / mcp_server.get("cwd", ".")).resolve())
    specs.append(("mcp-json", mcp_server["command"], mcp_server.get("args", []), mcp_cwd))

    return specs


def _handshake_tool_names(command: str, args: list[str], cwd: str) -> set[str]:
    node = shutil.which("node")
    if not node:
        pytest.skip("no node on PATH -- R2 needs Node as the real MCP launcher runtime")

    params = StdioServerParameters(
        command=node,
        args=["-e", _NODE_SPAWN_SCRIPT, cwd, command, *args],
    )

    async def _handshake():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                with anyio.fail_after(30):
                    await session.initialize()
                    return (await session.list_tools()).tools

    tools = anyio.run(_handshake)
    return {t.name for t in tools}


def _pretooluse_groups() -> list[dict]:
    """`hooks.json`'s `PreToolUse` groups, read verbatim from the repo file."""
    data = json.loads((REPO / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    return data["hooks"].get("PreToolUse", [])


def _matcher_selects(matcher: str, tool_name: str) -> bool:
    """Claude Code's own matcher rule (see plan #52 "Premises verified"): a
    matcher made up only of `[A-Za-z0-9_|]` is an exact pipe-separated name
    list; anything else is tested with an *unanchored* `re.search`, so e.g.
    `mcp__.*harness.*` also matches any tool name containing that substring
    anywhere, not just a full match."""
    if re.fullmatch(r"[A-Za-z0-9_|]+", matcher or ""):
        return tool_name in matcher.split("|")
    return re.search(matcher, tool_name) is not None


def _commands_selected_for(tool_name: str) -> list[str]:
    commands = []
    for group in _pretooluse_groups():
        if _matcher_selects(group.get("matcher", ""), tool_name):
            commands.extend(hook["command"] for hook in group["hooks"])
    return commands


def _run_pretooluse_command(command: str, plugin_root: Path, tmp_path: Path, bash_exe: str, tool_name: str):
    plugin_data = tmp_path / "plugin-data"
    plugin_data.mkdir(exist_ok=True)
    stdin_payload = json.dumps(
        {
            "session_id": f"native-{tool_name}",
            "cwd": str(plugin_root),
            "permission_mode": "bypassPermissions",
            "hook_event_name": "PreToolUse",
            "tool_name": tool_name,
            "tool_input": {},
        }
    )
    env = {
        **os.environ,
        "CLAUDE_PLUGIN_ROOT": plugin_root.as_posix(),
        "CLAUDE_PLUGIN_DATA": str(plugin_data),
    }
    return subprocess.run(
        [bash_exe, "-c", command],
        input=stdin_payload,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


# --- #52 R3: hooks.json routes Agent/Task to a deny, verbatim through bash --


@pytest.mark.parametrize("tool_name", ["Agent", "Task"])
def test_native_subagent_hook_command_denies(plugin_root, bash_exe, tmp_path, tool_name):
    """R3: every hooks.json PreToolUse group whose matcher selects `tool_name`
    under Claude Code's own matcher rule (`_matcher_selects` above), run
    verbatim through a real bash against the release-shaped `plugin_root`,
    must print the hook's deny JSON on stdout.

    Expected RED reason: no PreToolUse matcher in hooks.json selects `Agent`
    or `Task` today -- the group-selection assertion below fails first, with
    "no hooks.json PreToolUse group routes {tool_name}"."""
    commands = _commands_selected_for(tool_name)
    assert commands, f"no hooks.json PreToolUse group routes {tool_name}"

    for command in commands:
        result = _run_pretooluse_command(command, plugin_root, tmp_path, bash_exe, tool_name)
        assert result.returncode == 0, (
            f"bash exited {result.returncode} running {command!r} for tool_name={tool_name!r}; "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"stdout for {tool_name!r} did not parse as JSON: {exc}; "
                f"stdout={result.stdout!r} stderr={result.stderr!r}"
            )
        hook_output = parsed.get("hookSpecificOutput", {})
        assert hook_output.get("permissionDecision") == "deny", (
            f"expected a deny decision for {tool_name!r}, got {parsed!r}"
        )


def test_harness_tool_hook_command_not_denied(plugin_root, bash_exe, tmp_path):
    """Non-regression guard alongside R3: the group(s) selected for the
    existing harness MCP tool name must not print a deny.

    Expected RED reason: none -- may already pass; guards against an
    over-broad matcher or deny branch."""
    tool_name = "mcp__harness__harness_start_agent"
    commands = _commands_selected_for(tool_name)
    assert commands, f"expected an existing hooks.json PreToolUse group to route {tool_name}"

    for command in commands:
        result = _run_pretooluse_command(command, plugin_root, tmp_path, bash_exe, tool_name)
        assert result.returncode == 0, (
            f"bash exited {result.returncode} running {command!r}; "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert "deny" not in result.stdout, result.stdout


@pytest.mark.parametrize("spec_id", ["claude-plugin", "mcp-json"])
def test_mcp_launch_command_handshake(plugin_root, spec_id):
    """R2: the manifest's launch command, resolved and spawned by real Node
    (`child_process.spawn`, Claude Code's own runtime and libuv path
    resolution) against the release-shaped `bin/` this module built, must
    complete a real MCP initialize + tools/list handshake exposing
    `harness_list_agents`.

    Expected RED reason: probably none -- libuv's Windows spawn resolution
    appends `.com`/`.exe` to an extensionless command and never runs the
    literal file, so this likely already passes on both OSes today (kept as
    regression coverage that the eventual fix doesn't break the Linux MCP
    path or stdio passthrough through the dispatcher).
    """
    specs = {s[0]: s[1:] for s in _mcp_launch_specs(plugin_root)}
    command, args, cwd = specs[spec_id]

    names = _handshake_tool_names(command, args, cwd)
    assert names, f"{spec_id} launch produced no MCP tools"
    assert "harness_list_agents" in names
