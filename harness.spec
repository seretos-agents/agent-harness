# PyInstaller spec for the agent-harness MCP server.
#
# Produces a single-file self-contained binary that bundles the Python
# interpreter, the MCP runtime, and the package itself. Output extension is
# host-OS-dependent — `.exe` on Windows, no extension on Linux. PyInstaller
# handles the per-OS suffix automatically; this spec is OS-agnostic.
#
# Build:    pwsh -File scripts/build.ps1 -Clean
# Output:   dist/harness.exe on Windows, dist/harness on Linux
# Copy to:  bin/harness.exe (Windows) or bin/harness-linux (Linux), handled by
#           scripts/build.ps1 -- bin/harness itself is a separate, committed
#           POSIX dispatcher (#53), not PyInstaller output.

# ruff: noqa
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None
ROOT = Path(SPECPATH)

# `mcp.cli` requires optional `typer`/`rich` deps the server doesn't need.
# Collect mcp manually, filtering out the CLI subpackage so PyInstaller doesn't
# fail trying to import it.
def _not_cli(name: str) -> bool:
    return not name.startswith("mcp.cli")

mcp_hiddenimports = collect_submodules("mcp", filter=_not_cli)

extra_hidden = [
    # FastMCP runtime:
    "anyio",
    "pydantic",
    "pydantic_core",
    "starlette",
]
extra_hidden += collect_submodules("harness_plugin")
extra_hidden += collect_submodules("lib_python_harness")

a = Analysis(
    ["src/harness_plugin/__main__.py"],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[],
    hiddenimports=mcp_hiddenimports + extra_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "numpy",
        "PIL",
        "test",
        "unittest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="harness",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,            # don't compress — slower startup, no real size win on stdio binaries
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,          # MUST be console=True for stdio MCP transport
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
