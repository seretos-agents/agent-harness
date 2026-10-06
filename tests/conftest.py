import json
import os
import sys
from pathlib import Path

import pytest
from mcp import StdioServerParameters

FAKE_CLAUDE = Path(__file__).parent / "fixtures" / "fake_claude.py"


SESSION_ID = "test-session"

REPO = Path(__file__).resolve().parents[1]


def copy_classic_hooks(dest: Path) -> None:
    """Copy this repo's hooks/hooks.json to `dest` with the `modules` key stripped, so a
    live fixture exercising the classic command hooks (#52 deny, Stop wait) is not
    short-circuited by the native-Agent hooks module (#76)."""
    data = json.loads((REPO / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    data.pop("modules", None)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(data, indent=2) + "\n")


def plant_session_file(plugin_data: Path, session_id: str, **fields) -> Path:
    """Write what the hook would: <CLAUDE_PLUGIN_DATA>/sessions/<session_id>.json."""
    sessions = plugin_data / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    path = sessions / f"{session_id}.json"
    path.write_text(json.dumps({"session_id": session_id, **fields}), encoding="utf-8")
    return path


def _base_env(tmp_path) -> dict[str, str]:
    config_dir = tmp_path / "claude-config"
    home = tmp_path / "home"
    config_dir.mkdir(exist_ok=True)
    home.mkdir(exist_ok=True)
    return {
        "HARNESS_CLAUDE_ARGV": json.dumps([sys.executable, str(FAKE_CLAUDE)]),
        "HARNESS_ARTIFACTS_DIR": str(tmp_path / "artifacts"),
        "HARNESS_FAKE_ARGV_LOG": str(tmp_path / "argv.log"),
        "CLAUDE_CONFIG_DIR": str(config_dir),
        "CLAUDE_PLUGIN_DATA": str(tmp_path / "plugin-data"),
        "HOME": str(home),
        "USERPROFILE": str(home),
    }


def _params(env) -> StdioServerParameters:
    return StdioServerParameters(command=sys.executable, args=["-m", "harness_plugin"], env=env)


@pytest.fixture
def session_context(tmp_path, project_dir) -> dict:
    """The parent-session snapshot the hook would have collected, planted on disk."""
    ctx = {
        "session_id": SESSION_ID,
        "cwd": str(project_dir),
        "project_dir": str(project_dir),
        "permission_mode": "acceptEdits",
        "effort": "high",
        "model": "opus",
    }
    plant_session_file(tmp_path / "plugin-data", **ctx)
    return ctx


@pytest.fixture
def argv_log(tmp_path) -> Path:
    """Where fake_claude records the argv/cwd of every real (non --version) invocation."""
    return tmp_path / "argv.log"


@pytest.fixture
def server_params(tmp_path, session_context) -> StdioServerParameters:
    """A real `python -m harness_plugin` server wired to the fake claude CLI, with
    artifacts and user-level config isolated under tmp_path and a parent-session
    context planted and resolvable via CLAUDE_CODE_SESSION_ID."""
    env = _base_env(tmp_path)
    env["CLAUDE_CODE_SESSION_ID"] = SESSION_ID
    return _params(env)


@pytest.fixture
def live_server_params(tmp_path, session_context) -> StdioServerParameters:
    """Like `server_params` but talking to the real `claude` CLI (no HARNESS_CLAUDE_ARGV).
    PATH is inherited so the real binary resolves."""
    env = _base_env(tmp_path)
    del env["HARNESS_CLAUDE_ARGV"]
    env["CLAUDE_CODE_SESSION_ID"] = SESSION_ID
    # Real claude needs the user's credentials: keep the real config dir/home.
    for key in ("CLAUDE_CONFIG_DIR", "HOME", "USERPROFILE"):
        env.pop(key, None)
    env = {**os.environ, **env}
    return _params(env)


@pytest.fixture
def server_params_no_context(tmp_path) -> StdioServerParameters:
    """Same server, but no session id, no project dir and an empty sessions dir:
    no context can be resolved."""
    return _params(_base_env(tmp_path))


@pytest.fixture
def project_dir(tmp_path) -> Path:
    """A project cwd with two agent definitions: `demo` (no `effort:`) and
    `effort-agent` (`effort: medium` in its own frontmatter, distinct from both
    session_context's `effort: high` and any explicit argument used in tests, so a
    test can tell "the definition's value won" apart from "the argument's" or "the
    session's")."""
    agents = tmp_path / "project" / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / "demo.md").write_text(
        "---\nname: demo\ndescription: Demo agent for tests\n"
        "skills: [demo-skill-a, demo-skill-b]\n---\nSay OK.\n",
        encoding="utf-8",
    )
    (agents / "effort-agent.md").write_text(
        "---\nname: effort-agent\ndescription: Agent with its own fixed effort\n"
        "effort: medium\n---\nSay OK.\n",
        encoding="utf-8",
    )
    return tmp_path / "project"


@pytest.fixture
def plugin_agent_install(tmp_path) -> Path:
    """A minimal plugin install, `agent-harness@mk`, registered in
    `<CLAUDE_CONFIG_DIR>/plugins/installed_plugins.json` and enabled via
    `<CLAUDE_CONFIG_DIR>/settings.json`'s `enabledPlugins`, providing one agent
    definition -- `agents/probe.md` (`tools: Read, Glob`, body `Say OK.`) --
    discoverable as the colon-qualified `agent-harness:probe`. No `hooks`/
    `mcpServers`, so a run dispatched through it always takes the `--agents`
    JSON payload carrier, never the materialized one (which rewrites `:` to
    `__` in both the file stem and `--agent`, and so would not exercise the
    colon). `CLAUDE_CONFIG_DIR` itself is `tmp_path / "claude-config"`, the
    same path `_base_env` (test_live_claude.py) points every server fixture
    at. Returns the install directory."""
    install_dir = tmp_path / "plugins" / "agent-harness"
    agents = install_dir / "agents"
    agents.mkdir(parents=True)
    (agents / "probe.md").write_text(
        "---\nname: probe\ndescription: Probe agent for plugin dispatch tests\n"
        "tools: Read, Glob\n---\nSay OK.\n",
        encoding="utf-8",
    )
    config = tmp_path / "claude-config"
    (config / "plugins").mkdir(parents=True, exist_ok=True)
    (config / "plugins" / "installed_plugins.json").write_text(
        json.dumps(
            {
                "version": 2,
                "plugins": {
                    "agent-harness@mk": [
                        {"scope": "user", "installPath": str(install_dir)}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    (config / "settings.json").write_text(
        json.dumps({"enabledPlugins": {"agent-harness@mk": True}}), encoding="utf-8"
    )
    return install_dir


@pytest.fixture
def mcp_agent_project(tmp_path) -> Path:
    """A project cwd with one agent definition, `mcp-agent`, whose frontmatter sets
    `mcpServers:` — a key the `--agents` JSON schema rejects (`AGENT_JSON_KEYS`), so
    dispatching it always takes the materialized `.claude/agents/<stem>.md` carrier
    (`claude_cli.py:117-146`) — the only way `harness_inspect_run`'s system-prompt
    resolution reaches its `"materialized:<path>"` branch, and the only carrier that
    also always emits `--mcp-config` regardless of payload/materialized routing."""
    agents = tmp_path / "mcp-agent-project" / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / "mcp-agent.md").write_text(
        "---\n"
        "name: mcp-agent\n"
        "description: Agent requiring the materialized dispatch carrier\n"
        "mcpServers:\n"
        "  demo-server:\n"
        "    command: demo\n"
        "---\n"
        "Materialized agent body for inspection tests.\n",
        encoding="utf-8",
    )
    return tmp_path / "mcp-agent-project"


@pytest.fixture
def wait_run_env(tmp_path) -> dict[str, str]:
    """Environment for a `wait` child: the same HARNESS_ARTIFACTS_DIR /
    HARNESS_CLAUDE_ARGV the `server_params` server uses, layered over os.environ
    so Windows keeps SYSTEMROOT/PATH."""
    return {**os.environ, **_base_env(tmp_path)}


@pytest.fixture
def run_agent_cmd() -> list[str]:
    """argv prefix that launches the `run-agent` subcommand: a prebuilt binary when
    HARNESS_BIN points at one, else `python -m harness_plugin`."""
    binary = os.environ.get("HARNESS_BIN")
    base = [binary] if binary else [sys.executable, "-m", "harness_plugin"]
    return [*base, "run-agent"]


@pytest.fixture
def send_message_cmd() -> list[str]:
    """argv prefix that launches the `send-message` subcommand: a prebuilt binary when
    HARNESS_BIN points at one, else `python -m harness_plugin`."""
    binary = os.environ.get("HARNESS_BIN")
    base = [binary] if binary else [sys.executable, "-m", "harness_plugin"]
    return [*base, "send-message"]


@pytest.fixture
def shipped_agents_install(tmp_path) -> Path:
    """Like `plugin_agent_install`, but the `agent-harness@mk` install carries this
    repo's own shipped `agents/*.md` (general-purpose, Explore, Plan), so the native
    built-in type names map onto the definitions the plugin really ships."""
    repo = Path(__file__).resolve().parents[1]
    install_dir = tmp_path / "plugins" / "agent-harness"
    agents = install_dir / "agents"
    agents.mkdir(parents=True)
    for md in (repo / "agents").glob("*.md"):
        (agents / md.name).write_text(md.read_text(encoding="utf-8"), encoding="utf-8")
    config = tmp_path / "claude-config"
    (config / "plugins").mkdir(parents=True, exist_ok=True)
    (config / "plugins" / "installed_plugins.json").write_text(
        json.dumps(
            {
                "version": 2,
                "plugins": {
                    "agent-harness@mk": [{"scope": "user", "installPath": str(install_dir)}]
                },
            }
        ),
        encoding="utf-8",
    )
    (config / "settings.json").write_text(
        json.dumps({"enabledPlugins": {"agent-harness@mk": True}}), encoding="utf-8"
    )
    return install_dir


@pytest.fixture
def wait_run_cmd() -> list[str]:
    """argv prefix that launches the `wait` subcommand: a prebuilt binary when
    HARNESS_BIN points at one, else `python -m harness_plugin`."""
    binary = os.environ.get("HARNESS_BIN")
    base = [binary] if binary else [sys.executable, "-m", "harness_plugin"]
    return [*base, "wait"]
