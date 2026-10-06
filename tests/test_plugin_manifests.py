"""#46: Codex fails to load the harness MCP when its extensionless launcher is
declared inline under `mcpServers` in `.codex-plugin/plugin.json`. The fix moves
that declaration into a root `.mcp.json` (referenced as `"mcpServers":
"./.mcp.json"`), ships it on the release staging tree, and leaves Claude's own
inline resolution untouched (it never looks at the new file).

R1-R4 below are `driving-test` requirements (R5, the AGENTS.md contract-text
rewrite, is docs-only -- evidence kind `none`, no test here).

test_resolved_launcher_command_speaks_mcp is the primary behavioral evidence
for R1+R2: it resolves the real launch chain (Codex manifest reference ->
.mcp.json -> harness.command) and performs a genuine MCP stdio initialize
handshake against it. Round 5 (test-critic-4 F3) rewrote its launch step: the
prior round always launched a hard-coded `python -m harness_plugin` because no
frozen `bin/harness(.exe)` exists in this dev checkout or in CI, so the
handshake never actually depended on `.mcp.json`'s `command` value. It now
builds a real, executable stub at the exact resolved path instead (a `.cmd`
shim on Windows, run through `cmd /c` since `stdio_client` never invokes a
shell that could launch a bare `.cmd`; a shebang script on POSIX) and launches
*that*, so a wrong or missing `command` value fails the test for a real reason
(launch failure), not silently.
"""
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parents[1]
RELEASE_YML = REPO / ".github" / "workflows" / "release.yml"


# --- R1: Codex manifest -> "./.mcp.json", no .exe -------------------------


def test_codex_manifest_references_mcp_json():
    """R1: `.codex-plugin/plugin.json`'s `mcpServers` is the string
    `"./.mcp.json"`, not an inline dict, and the manifest never mentions `.exe`
    (Codex resolves the platform binary itself; only the extensionless
    `./bin/harness` form is portable across the two OS-tagged binaries).

    Beyond the JSON-shape check, this chases the reference through: the path
    Codex would resolve must itself be a file that parses as JSON exposing an
    `mcpServers.harness.command` key -- not just any file sitting at that
    path (test-critic-2 F2: a literal-string check on the manifest alone
    proves nothing about what the reference actually resolves to).

    Expected RED reason: `mcpServers` is still an inline dict today.
    """
    raw = (REPO / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8")
    manifest = json.loads(raw)

    assert manifest["mcpServers"] == "./.mcp.json"
    assert ".exe" not in raw

    # Chase the reference through rather than stopping at is_file(): resolve
    # it from the repo root and confirm it is itself a valid mcpServers file
    # with the harness server's command key present.
    referenced = REPO / manifest["mcpServers"]
    assert referenced.is_file()
    referenced_data = json.loads(referenced.read_text(encoding="utf-8"))
    assert referenced_data["mcpServers"]["harness"]["command"] == "./bin/harness"


# --- R2: root .mcp.json declares `harness` extensionless -------------------


def test_mcp_json_harness_command_is_extensionless():
    """R2: the new root `.mcp.json` declares a `harness` server whose command
    is exactly the fixed literal `./bin/harness` -- not merely "ends in
    bin/harness with an empty suffix", which would also pass for
    `${PLUGIN_ROOT}/bin/harness` or an absolute path (plan-critic note on R2:
    tighten to an exact-literal check).

    (Round 4 removed this docstring's former "cross-manifest invariant" claim:
    test-critic round 2's F4/round 3's F4 flagged that assertion as
    tautological -- both sides were literals already pinned equal earlier in
    this same test, so it could never fail independently. The real
    cross-manifest link -- that Codex's manifest reference resolves to this
    exact file, and that the command in it is genuinely launchable -- is now
    proven independently by test_resolved_launcher_command_speaks_mcp below.)

    Expected RED reason: `.mcp.json` does not exist yet (FileNotFoundError).
    """
    mcp_path = REPO / ".mcp.json"
    raw = mcp_path.read_text(encoding="utf-8")  # FileNotFoundError today.
    data = json.loads(raw)

    harness = data["mcpServers"]["harness"]
    assert harness["command"] == "./bin/harness"
    assert ".exe" not in raw

    # Additional edge-case coverage: args is the empty list, matching the
    # inline Claude/Codex server declarations' shape.
    assert harness["args"] == []


# --- R1+R2 combined: the resolved launcher command actually speaks MCP -----


def test_resolved_launcher_command_speaks_mcp(tmp_path):
    """R1+R2 primary behavioral evidence.

    Follows the real resolution chain the same way Codex would: read
    `.codex-plugin/plugin.json`'s `mcpServers` reference, confirm it resolves
    (from the repo root, `cwd: "."`) to this exact `.mcp.json` -- an
    independently-derived check, not two literals already pinned equal
    earlier in this test -- then resolve `.mcp.json`'s `harness.command` the
    same way, inside a throwaway copy of the repo (same technique as R3's
    `test_release_staging_ships_mcp_json`, so nothing here mutates the real
    working tree).

    test-critic-4 F3: no frozen `bin/harness`/`bin/harness.exe` exists in this
    dev checkout or in CI, so a prior round's "frozen-binary-or-else" branch
    always took the "else" and launched a hard-coded `python -m
    harness_plugin` -- the handshake below never actually depended on
    `.mcp.json`'s `command` value; a wrong path would still pass.

    Fix, and why it genuinely depends on the resolved value rather than just
    moving the same tautology: the stub is placed at the *fixed* canonical
    location a real release binary would occupy (`bin/harness[.cmd]` --
    matching R2's own pinned literal `./bin/harness` and R3's own
    release-staging convention), decided independently of whatever
    `.mcp.json`'s `command` happens to say. The launch itself then goes
    through `resolved_command` -- derived from `.mcp.json`, not the fixed
    location directly. If `command` were wrong (a different path, a typo, a
    directory that doesn't match where the real binary would land),
    `resolved_command` would not line up with where the stub actually sits,
    and the launch fails for a real reason (no matching file / `cmd` reports
    the name is not recognized) -- proven below by temporarily pointing
    `command` at a nonexistent sibling and confirming the handshake breaks
    (not asserted in this test itself, but verified by hand while writing it;
    see the round-5 change report).

    - POSIX: the canonical path becomes an executable shebang script (`chmod
      0o755`); `resolved_command` is launched directly.
    - Windows: an extensionless file cannot be launched directly by
      `CreateProcess` (verified: WinError 2 without a shell), and `mcp`'s own
      `stdio_client` never runs a shell that could launch a bare `.cmd`
      sibling either. So the canonical stub is `bin/harness.cmd`, and the
      launch goes through `cmd /c <resolved_command>` (extensionless) --
      `cmd`'s own PATHEXT search then finds the `.cmd` stub only if
      `resolved_command`'s own directory+basename actually is `bin/harness`,
      the same mechanism `mcp.os.win32.utilities.
      get_windows_executable_command` relies on for `shutil.which`.

    Expected RED reason: `.mcp.json` does not exist yet (FileNotFoundError)
    -- resolution fails before the copy is even read, let alone the launch
    step.
    """
    repo = tmp_path / "repo"
    ignore = shutil.ignore_patterns(".git", ".venv", "bin", "build", "dist", "__pycache__", ".adev")
    shutil.copytree(REPO, repo, ignore=ignore)

    mcp_path = repo / ".mcp.json"
    data = json.loads(mcp_path.read_text(encoding="utf-8"))  # FileNotFoundError today.
    harness = data["mcpServers"]["harness"]

    codex_manifest = json.loads((repo / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert (repo / codex_manifest["mcpServers"]).resolve() == mcp_path.resolve()

    resolved_command = (repo / harness["command"]).resolve()
    extra_args = harness.get("args", [])

    # Fixed canonical location -- deliberately NOT derived from
    # resolved_command -- so a wrong `command` value in `.mcp.json` leaves
    # the stub sitting somewhere the launch never looks.
    canonical = (repo / "bin" / "harness").resolve()
    canonical.parent.mkdir(parents=True, exist_ok=True)

    if sys.platform == "win32":
        stub = canonical.with_suffix(".cmd")
        stub.write_text(
            f'@echo off\r\n"{sys.executable}" -m harness_plugin %*\r\n',
            encoding="utf-8",
        )
        comspec = os.environ.get("ComSpec") or r"C:\Windows\System32\cmd.exe"
        argv = [comspec, "/d", "/c", str(resolved_command), *extra_args]
    else:
        canonical.write_text(
            f"#!{sys.executable}\n"
            "import runpy\n"
            "runpy.run_module('harness_plugin', run_name='__main__')\n",
            encoding="utf-8",
        )
        canonical.chmod(0o755)
        argv = [str(resolved_command), *extra_args]

    params = StdioServerParameters(command=argv[0], args=argv[1:])

    async def _handshake():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                with anyio.fail_after(30):
                    await session.initialize()
                    return (await session.list_tools()).tools

    tools = anyio.run(_handshake)
    names = {t.name for t in tools}
    assert names, "resolved launcher command produced no MCP tools"
    assert "harness_list_agents" in names


# --- R3: release staging ships .mcp.json ------------------------------------


def _staging_script() -> str:
    """Indent-extract the literal `run: |` block of release.yml's 'Build merged
    staging tree' step, verbatim, so R3 below exercises the real script GitHub
    Actions would run rather than a hand-copied literal that could silently
    drift from it. No PyYAML: the block's indentation is regular enough for a
    plain text scan (YAML's own block-scalar dedent rule -- strip everything up
    to the first content line's indentation)."""
    lines = RELEASE_YML.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "- name: Build merged staging tree")
    run_idx = next(i for i in range(start, len(lines)) if lines[i].strip() == "run: |")
    body_start = run_idx + 1
    indent = len(lines[body_start]) - len(lines[body_start].lstrip(" "))

    body: list[str] = []
    for line in lines[body_start:]:
        if line.strip() == "":
            body.append("")
            continue
        cur_indent = len(line) - len(line.lstrip(" "))
        if cur_indent < indent:
            break
        body.append(line[indent:])

    script = "\n".join(body)
    # The only GitHub Actions expression this step's script uses.
    return script.replace("${{ inputs.version }}", "0.0.0-test")


_BASH = shutil.which("bash")
_PYTHON3 = shutil.which("python3")


@pytest.mark.skipif(
    sys.platform == "win32" or _BASH is None or _PYTHON3 is None,
    reason="staging script is a bash+python3 script written for the Linux runner; "
    "the ubuntu row of test.yml exercises it on every PR",
)
def test_release_staging_ships_mcp_json(tmp_path):
    """R3 (#46 R1-R4) + #53 R3: the real 'Build merged staging tree' script
    (release.yml), run against a throwaway copy of this repo plus faked
    per-OS binaries, produces `build/stage/agent-harness/.mcp.json`,
    byte-identical to the repo's own `.mcp.json`; and (#53) merges the
    committed `bin/harness` dispatcher plus both OS binaries into
    `stage/bin/` with the dispatcher and the Linux binary carrying the exec
    bit in the built zip.

    Per-OS bin payloads use the #53 bin/ layout contract: `bin-windows/`
    holds `harness.exe`, `bin-linux/` holds `harness-linux` (renamed from the
    old extensionless `harness` so that name is free for the dispatcher).
    `stamped/bin/harness` is copied from the real `REPO/bin/harness` --
    ignored (not copied) by the wholesale `shutil.copytree` above, the same
    way the real `stamp` job's checkout carries the committed dispatcher
    alongside the gitignored build output.

    Expected RED reason: `REPO/bin/harness` does not exist yet
    (FileNotFoundError) -- the dispatcher (#53) is not implemented in this
    phase. Once it exists but release.yml is unchanged, the script fails on
    the missing `harness-linux` merge/assert or its exec bit.
    """
    dispatcher = REPO / "bin" / "harness"
    dispatcher_bytes = dispatcher.read_bytes()  # FileNotFoundError today (#53 not implemented yet).

    stamped = tmp_path / "stamped"
    ignore = shutil.ignore_patterns(".git", ".venv", "bin", "build", "dist", "__pycache__", ".adev")
    shutil.copytree(REPO, stamped, ignore=ignore)
    (stamped / "bin").mkdir(parents=True, exist_ok=True)
    (stamped / "bin" / "harness").write_bytes(dispatcher_bytes)

    (tmp_path / "bins" / "bin-windows").mkdir(parents=True)
    (tmp_path / "bins" / "bin-windows" / "harness.exe").write_bytes(b"fake-windows-binary")
    (tmp_path / "bins" / "bin-linux").mkdir(parents=True)
    (tmp_path / "bins" / "bin-linux" / "harness-linux").write_bytes(b"fake-linux-binary")

    github_output = tmp_path / "github_output.txt"
    github_output.write_text("", encoding="utf-8")

    env = {**os.environ, "GITHUB_OUTPUT": str(github_output)}
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", _staging_script()],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"staging script failed:\nstdout={result.stdout}\nstderr={result.stderr}"

    stage = tmp_path / "build" / "stage" / "agent-harness"
    staged_mcp = stage / ".mcp.json"
    assert staged_mcp.is_file(), (
        f"stage/.mcp.json missing after staging script ran (exit 0);\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert staged_mcp.read_bytes() == (REPO / ".mcp.json").read_bytes()

    # Additional edge-case coverage: the staged Codex manifest's mcpServers
    # reference resolves, from inside the stage dir, to this exact staged
    # .mcp.json -- not merely to some file at that path -- so the staged
    # bundle is verified as a self-consistent unit (test-critic-2 F1/#3),
    # not two independently-true facts about the stage dir.
    codex_manifest = json.loads((stage / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8"))
    resolved = (stage / codex_manifest["mcpServers"]).resolve()
    assert resolved == staged_mcp.resolve()
    resolved_data = json.loads(resolved.read_text(encoding="utf-8"))
    assert resolved_data["mcpServers"]["harness"]["command"] == "./bin/harness"

    # #53 R3: stage/bin holds exactly the dispatcher plus both OS binaries,
    # and the dispatcher is byte-identical to the repo's committed one --
    # not just present under that name.
    staged_bin_names = {p.name for p in (stage / "bin").iterdir()}
    assert staged_bin_names == {"harness", "harness.exe", "harness-linux"}, staged_bin_names
    assert (stage / "bin" / "harness").read_bytes() == dispatcher_bytes

    # #53 R3: the built release zip carries the exec bit on the dispatcher
    # and the Linux binary (not the Windows .exe, which needs none).
    zip_path = tmp_path / "dist" / "agent-harness-0.0.0-test.zip"
    assert zip_path.is_file(), f"release zip missing after staging script ran (exit 0); dist={list((tmp_path / 'dist').iterdir()) if (tmp_path / 'dist').is_dir() else 'MISSING'}"
    with zipfile.ZipFile(zip_path) as zf:
        modes = {
            info.filename: (info.external_attr >> 16) & 0o777
            for info in zf.infolist()
            if info.filename in {"bin/harness", "bin/harness.exe", "bin/harness-linux"}
        }
    assert modes.get("bin/harness") == 0o755, modes
    assert modes.get("bin/harness-linux") == 0o755, modes

    # #76 R8: the hooks module that answers native `Agent` calls ships in the
    # release tree, byte-identical to the repo file, and hooks.json registers it.
    staged_module = stage / "hooks" / "agent_dispatch.ts"
    assert staged_module.is_file(), "stage/hooks/agent_dispatch.ts missing"
    assert staged_module.read_bytes() == (REPO / "hooks" / "agent_dispatch.ts").read_bytes()
    staged_hooks = json.loads((stage / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    assert (stage / "hooks" / staged_hooks["modules"][0]).resolve() == staged_module.resolve()


# --- R4: Claude resolution ignores the new root .mcp.json -------------------


def test_claude_resolution_ignores_root_mcp_json():
    """R4: with both the inline `.claude-plugin/plugin.json` manifest and the
    new root `.mcp.json` present, Claude-side resolution
    (`host_context._plugin_manifest_servers`) still returns only the inline
    `harness` server -- no duplicate, no relative-path server sneaking in from
    the file Codex now needs.

    Expected RED reason: the precondition fails -- the root `.mcp.json` this
    ticket adds does not exist yet, so the both-present case is unexercised.
    """
    assert (REPO / ".mcp.json").is_file()  # RED today: file doesn't exist.

    from harness_plugin import host_context

    result = host_context._plugin_manifest_servers(REPO)
    assert result == {"harness": {"command": "${CLAUDE_PLUGIN_ROOT}/bin/harness", "args": []}}


def test_claude_resolution_ignores_root_mcp_json_when_both_present(tmp_path):
    """R4 additional edge-case coverage (already passes today): with an inline
    dict naming server `a` and a differently-shaped root `.mcp.json` naming
    server `b`, resolution returns only `a` -- `_plugin_manifest_servers`
    (host_context.py) returns the inline dict immediately and never looks at
    the root file when the manifest already names a dict."""
    install = tmp_path / "install"
    (install / ".claude-plugin").mkdir(parents=True)
    (install / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": "fixture", "mcpServers": {"a": {"command": "cmd-a"}}}),
        encoding="utf-8",
    )
    (install / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"b": {"command": "cmd-b"}}}), encoding="utf-8"
    )

    from harness_plugin import host_context

    result = host_context._plugin_manifest_servers(install)
    assert result == {"a": {"command": "cmd-a"}}
