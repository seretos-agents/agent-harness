"""#46: Codex fails to load the harness MCP when its extensionless launcher is
declared inline under `mcpServers` in `.codex-plugin/plugin.json`. The fix moves
that declaration into a root `.mcp.json` (referenced as `"mcpServers":
"./.mcp.json"`), ships it on the release staging tree, and leaves Claude's own
inline resolution untouched (it never looks at the new file).

R1-R4 below are `driving-test` requirements (R5, the AGENTS.md contract-text
rewrite, is docs-only -- evidence kind `none`, no test here).
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

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
    assert "command" in referenced_data["mcpServers"]["harness"]


# --- R2: root .mcp.json declares `harness` extensionless -------------------


def test_mcp_json_harness_command_is_extensionless():
    """R2: the new root `.mcp.json` declares a `harness` server whose command
    is exactly the fixed literal `./bin/harness` -- not merely "ends in
    bin/harness with an empty suffix", which would also pass for
    `${PLUGIN_ROOT}/bin/harness` or an absolute path (plan-critic note on R2:
    tighten to an exact-literal check).

    Beyond the exact-command check, this pins a real cross-manifest
    invariant (test-critic-2 F3): the Codex-side `.mcp.json` command and the
    Claude-side inline `.claude-plugin/plugin.json` command must, once each
    manifest's own plugin-root placeholder is stripped, resolve to the same
    `./bin/harness` path -- so a change that moves the binary in one
    manifest but not the other fails here, not just a standalone literal.

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

    # Cross-manifest invariant: both manifests' commands ultimately point at
    # bin/harness under their respective plugin root.
    claude_manifest = json.loads((REPO / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    claude_cmd = claude_manifest["mcpServers"]["harness"]["command"]
    assert claude_cmd == "${CLAUDE_PLUGIN_ROOT}/bin/harness"
    assert claude_cmd.replace("${CLAUDE_PLUGIN_ROOT}", ".") == harness["command"]


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
    """R3: the real 'Build merged staging tree' script (release.yml), run
    against a throwaway copy of this repo plus faked per-OS binaries, produces
    `build/stage/agent-harness/.mcp.json`, byte-identical to the repo's own
    `.mcp.json`.

    Expected RED reason: the script exits 0 (it doesn't fail on a missing
    optional file) but `stage/.mcp.json` is absent -- nothing in the script
    copies it yet.
    """
    stamped = tmp_path / "stamped"
    ignore = shutil.ignore_patterns(".git", ".venv", "bin", "build", "dist", "__pycache__", ".adev")
    shutil.copytree(REPO, stamped, ignore=ignore)

    (tmp_path / "bins" / "bin-windows").mkdir(parents=True)
    (tmp_path / "bins" / "bin-windows" / "harness.exe").write_bytes(b"fake-windows-binary")
    (tmp_path / "bins" / "bin-linux").mkdir(parents=True)
    (tmp_path / "bins" / "bin-linux" / "harness").write_bytes(b"fake-linux-binary")

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
