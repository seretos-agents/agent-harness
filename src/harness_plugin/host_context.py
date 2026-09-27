"""The parent session's collected context: where the hook writes it, how the server finds it.

Deliberately free of `mcp`/`lib_python_harness` imports at module level so the per-tool-call
`hook` subcommand stays cheap."""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SESSION_ENV = "CLAUDE_CODE_SESSION_ID"
PROJECT_ENV = "CLAUDE_PROJECT_DIR"
PLUGIN_DATA_ENV = "CLAUDE_PLUGIN_DATA"


def sessions_dir(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    base = env.get(PLUGIN_DATA_ENV)
    root = Path(base) if base else Path.home() / ".agent-harness"
    return root / "sessions"


def _safe_name(session_id: Any) -> str | None:
    if not isinstance(session_id, str) or not session_id:
        return None
    if session_id != Path(session_id).name or session_id in (".", ".."):
        return None
    return session_id


def write_session_context(payload: dict[str, Any], env: Mapping[str, str] | None = None) -> Path:
    """Atomically write `payload` to sessions/<session_id>.json (last write wins)."""
    name = _safe_name(payload.get("session_id"))
    if name is None:
        raise ValueError("payload has no usable session_id")
    directory = sessions_dir(env)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{name}.json"
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def _read(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def load_session_context(
    env: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Return (context, source): "session" (exact file for CLAUDE_CODE_SESSION_ID),
    "fallback" (newest file whose cwd/project_dir equals CLAUDE_PROJECT_DIR; mtime ties go
    to the lexicographically greatest filename) or "none"."""
    env = os.environ if env is None else env
    directory = sessions_dir(env)
    session_id = _safe_name(env.get(SESSION_ENV))
    if session_id:
        data = _read(directory / f"{session_id}.json")
        if data is not None:
            return data, "session"
    project = env.get(PROJECT_ENV)
    if project and directory.is_dir():
        best: tuple[float, str] | None = None
        best_data: dict[str, Any] | None = None
        for path in directory.glob("*.json"):
            data = _read(path)
            if data is None or project not in (data.get("cwd"), data.get("project_dir")):
                continue
            try:
                key = (path.stat().st_mtime, path.name)
            except OSError:
                continue
            if best is None or key > best:
                best, best_data = key, data
        if best_data is not None:
            return best_data, "fallback"
    return None, "none"


def probe_warning(env: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if env is None else env
    if env.get(SESSION_ENV):
        return None
    return (
        f"agent-harness: {SESSION_ENV} is not set; the parent session cannot be identified "
        f"exactly. Falling back to the newest context in {sessions_dir(env)} matching "
        f"{PROJECT_ENV}; harness_start_agent refuses if none resolves."
    )


def artifacts_root() -> Path:
    """Where run records/artifacts live (`<root>/<run_id>/record.json`, etc).
    Moved here (#62) from `runs.py` -- verbatim, `runs.py` re-imports it --
    so the per-tool-call `hook` subcommand can read a run's `state` off disk
    without importing `lib_python_harness` at module level."""
    override = os.environ.get("HARNESS_ARTIFACTS_DIR")
    root = Path(override) if override else Path.home() / ".agent-harness" / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _config_root() -> Path:
    base = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(base) if base else Path.home() / ".claude"


def _project_entry(projects: dict[str, Any], cwd: str) -> dict[str, Any]:
    """`projects[cwd]`, matched case-insensitively on Windows: `.claude.json`
    stores the CLI's own path casing, which can differ from the harness's cwd."""
    entry = projects.get(cwd)
    if entry is not None:
        return entry
    if os.name == "nt":
        target = cwd.lower()
        for key, value in projects.items():
            if key.lower() == target:
                return value
    return {}


def _project_approval(project_dir: Path, project_entry: Mapping[str, Any]) -> tuple[bool, list[str] | None]:
    """(enable_all, enabled_names): `enableAllProjectMcpServers` read from the
    project's own settings.json/settings.local.json (local wins when both state
    it), and the opt-in-by-name list from `.claude.json`'s own
    `projects[cwd].enabledMcpjsonServers` -- two independent approval files, so a
    corrupt `.claude.json` does not also blind the settings.json-based approval."""
    enable_all = False
    for name in ("settings.json", "settings.local.json"):
        data = _read(project_dir / ".claude" / name)
        if data is not None and "enableAllProjectMcpServers" in data:
            enable_all = bool(data["enableAllProjectMcpServers"])
    enabled_names = project_entry.get("enabledMcpjsonServers")
    return enable_all, enabled_names if isinstance(enabled_names, list) else None


def _expand(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        for token, replacement in mapping.items():
            value = value.replace(token, replacement)
        return value
    if isinstance(value, list):
        return [_expand(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item, mapping) for key, item in value.items()}
    return value


def _plugin_manifest_servers(install_path: Path) -> dict[str, Any]:
    """The plugin's own `mcpServers`, from its manifest -- inline dict, a path to
    another `.mcp.json`-shaped file, or (when the manifest names neither) a
    fallback to `<install_path>/.mcp.json`, the forms the CLI itself accepts."""
    manifest = _read(install_path / ".claude-plugin" / "plugin.json")
    spec = manifest.get("mcpServers") if manifest else None
    if isinstance(spec, dict):
        return spec
    if isinstance(spec, str):
        data = _read(install_path / spec)
        servers = data.get("mcpServers") if data else None
        return servers if isinstance(servers, dict) else {}
    data = _read(install_path / ".mcp.json")
    servers = data.get("mcpServers") if data else None
    return servers if isinstance(servers, dict) else {}


def parent_mcp_servers(cwd: str) -> dict[str, Any]:
    """Rebuild the parent Claude Code session's active MCP-server set from its own
    files -- neither the lib nor the hook payload collects it (#38 R2). Missing or
    corrupt files are skipped, never raised.

    Merge order: user (`<CLAUDE_CONFIG_DIR>/.claude.json` top level) -> project
    (`<cwd>/.mcp.json`, approved via `enableAllProjectMcpServers` or
    `enabledMcpjsonServers`, minus `disabledMcpjsonServers`) -> local
    (`projects[cwd].mcpServers`, forwarded only when
    `projects[cwd].hasTrustDialogAccepted` is true -- the same trust gate the real
    CLI applies before auto-connecting local-scope servers; an untrusted entry's
    local servers are dropped, not forwarded), then
    `projects[cwd].disabledMcpServers` is dropped from the merged result. Enabled
    plugins' own manifests are added last,
    keyed `plugin_<plugin>_<server>` with `${CLAUDE_PLUGIN_ROOT}`/`${PLUGIN_ROOT}`/
    `${CLAUDE_PLUGIN_DATA}`/`${CLAUDE_PROJECT_DIR}` expanded and that plugin's own
    `CLAUDE_PLUGIN_ROOT`/`CLAUDE_PLUGIN_DATA` injected into `env` -- except the
    `agent-harness` plugin's own `harness` server, kept under the literal bare key
    `harness` (AC1) so it both matches the lib's `dispatch_mcp_server_name` lookup
    and shadows any user/local server that happens to also be named `harness`.
    """
    cfg_dir = _config_root()
    project_dir = Path(cwd)
    cwd_str = str(project_dir)

    claude_json = _read(cfg_dir / ".claude.json") or {}
    user_servers = claude_json.get("mcpServers")
    user_servers = user_servers if isinstance(user_servers, dict) else {}
    projects = claude_json.get("projects")
    projects = projects if isinstance(projects, dict) else {}
    project_entry = _project_entry(projects, cwd_str)
    local_servers = project_entry.get("mcpServers")
    local_servers = local_servers if isinstance(local_servers, dict) else {}
    if not project_entry.get("hasTrustDialogAccepted"):
        # The real CLI only auto-connects local-scope servers once the project's
        # trust dialog has been accepted (normally implied by an interactive
        # `claude mcp add -s local`); an untrusted project entry's local
        # mcpServers are silently skipped, never forwarded to the child.
        local_servers = {}
    disabled_merged = set(project_entry.get("disabledMcpServers") or [])
    disabled_jsonc = set(project_entry.get("disabledMcpjsonServers") or [])

    enable_all, enabled_names = _project_approval(project_dir, project_entry)
    project_mcp = _read(project_dir / ".mcp.json") or {}
    project_all = project_mcp.get("mcpServers")
    project_all = project_all if isinstance(project_all, dict) else {}
    if enable_all:
        approved_names = set(project_all) - disabled_jsonc
    elif enabled_names is not None:
        approved_names = set(enabled_names) - disabled_jsonc
    else:
        approved_names = set()
    project_servers = {name: project_all[name] for name in approved_names if name in project_all}

    merged: dict[str, Any] = {}
    merged.update(user_servers)
    merged.update(project_servers)
    merged.update(local_servers)
    for name in disabled_merged:
        merged.pop(name, None)

    from lib_python_harness.agents.discovery import _enabled_plugin_install_paths

    for plugin_key, install_path_str in _enabled_plugin_install_paths(cfg_dir, project_dir).items():
        plugin_name = plugin_key.split("@", 1)[0]
        install_path = Path(install_path_str)
        servers = _plugin_manifest_servers(install_path)
        if not servers:
            continue
        data_dir = str(cfg_dir / "plugins" / "data" / plugin_name)
        mapping = {
            "${CLAUDE_PLUGIN_ROOT}": str(install_path),
            "${PLUGIN_ROOT}": str(install_path),
            "${CLAUDE_PLUGIN_DATA}": data_dir,
            "${CLAUDE_PROJECT_DIR}": cwd_str,
        }
        for server_name, server_def in servers.items():
            if not isinstance(server_def, dict):
                continue
            expanded = _expand(server_def, mapping)
            server_env = dict(expanded.get("env") or {})
            server_env["CLAUDE_PLUGIN_ROOT"] = str(install_path)
            server_env["CLAUDE_PLUGIN_DATA"] = data_dir
            expanded["env"] = server_env
            key = (
                "harness"
                if plugin_name == "agent-harness" and server_name == "harness"
                else f"plugin_{plugin_name}_{server_name}"
            )
            merged[key] = expanded

    return merged


def build_host_context(data: Mapping[str, Any], cwd: str, **overrides: str | None):
    """A completed lib HostContext from a session file; non-None overrides win.
    `mcp_servers`/`available_mcp_servers` are the parent's rebuilt server set
    (`parent_mcp_servers`, #38 R2), and `dispatch_mcp_server_name` is the literal
    `"harness"` (AC1) when that key is present in the rebuilt set, else `None`."""
    from lib_python_harness import HostContext

    def pick(key: str) -> str | None:
        return overrides.get(key) or data.get(key)

    servers = parent_mcp_servers(cwd)
    ctx = HostContext(
        cwd=cwd,
        session_id=data.get("session_id"),
        model=pick("model"),
        permission_mode=pick("permission_mode"),
        effort=pick("effort"),
        mcp_servers=servers,
        available_mcp_servers=servers,
        dispatch_mcp_server_name="harness" if "harness" in servers else None,
    )
    ctx.complete()
    return ctx
