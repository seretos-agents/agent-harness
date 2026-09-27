# agent-harness

Provider-independent subagent system for coding agents - replaces the host's built-in subagent tool calls.

## Quick install

**Claude Code:**

```
/plugin marketplace add seretos-agents/modular-software-factory
/plugin install agent-harness@modular-software-factory
```

Self-contained binary — no Python, no `pip install`, no dependencies. The release zip ships native binaries for both Windows (`harness.exe`) and Linux (`harness-linux`), plus a small committed dispatcher (`harness`) that every launch point actually names; each launcher resolves to the right native file for its OS.

## Alternative installs

### From the GitHub Releases page

1. Download `agent-harness-<version>.zip` from [Releases](https://github.com/seretos-agents/agent-harness/releases).
2. Unpack to a stable folder (e.g. `C:\Users\<you>\.claude\plugins\agent-harness\` on Windows, `~/.claude/plugins/agent-harness/` on Linux).
3. In Claude Code:
   ```
   /plugin install <path-to-unpacked-folder>
   ```

### From the release branch

The `release` branch always carries the latest install-ready files (no zip step):

```
git clone --branch release --depth 1 https://github.com/seretos-agents/agent-harness.git
```

Then `/plugin install <cloned-path>` in Claude Code.

### Build from source

Requires Python 3.11+ (standard python.org installer with the `py` launcher on Windows; `python3` on Linux).

```powershell
git clone https://github.com/seretos-agents/agent-harness.git
cd agent-harness
pwsh -File scripts/build.ps1 -Clean -Package
```

Output on Windows: `bin/harness.exe`. On Linux: `bin/harness-linux`. The committed `bin/harness` dispatcher (already in the repo) execs whichever one matches the host. Then install via `/plugin install <path>`.

## Waiting for a long run: `harness wait <run_id>`

No time limit in this plugin ends a run. When `harness_wait_run`'s `timeout_seconds` expires the run keeps going: the tool returns `state: RUNNING` with liveness fields (`duration_s`, `event_count`, `last_event_at`, `last_activity` - the last tool/command the run used) and a `next_step` hint. Only `harness_stop_run` cancels a run. To keep waiting beyond a tool call, run the binary's second entry point from a background shell (see the `harness-wait` skill), or call `harness_wait_run` again:

```
harness wait <run_id> [--timeout <seconds>] [--interval <seconds>]
```

`harness_list_runs` lists all recorded runs as compact rows (`run_id`, `state`, `model`, `cwd`, `created_at`, `label`; no text or usage) so a lost `run_id` can be found again; `harness_start_prompt`/`harness_start_agent` accept an optional `label`, `harness_start_agent` also accepts an optional `prompt` (the run's task / user message; the agent definition's body stays its system prompt), and `harness_poll_run` reports `event_count`/`last_event_at` while a run is RUNNING.

Every answer about a run -- both start tools plus `harness_poll_run`, `harness_wait_run` and `harness_send_message` for the same `run_id` -- carries `model`, `permission_mode` (`harness_start_agent` only; `harness_start_prompt` never sends one), `effort` and `effort_source`, all read back from the run's own launched argv so they reflect what the child actually started with rather than what was asked for. `effort_source` names where the launched effort came from: `agent_definition` (the agent's own `effort:` frontmatter, which outranks everything else), `argument` (an explicit `effort` passed to the tool), `parent_session` (inherited from the parent session's context -- `harness_start_agent` only), or `none` (nothing supplied one, so the CLI's own default effort applies).

`harness_inspect_run(run_id)` answers, for one run, what it announced (`announced.{mcp_servers,tools,skills,agents}`, from its own `events.jsonl` init event, plus `announced_counts` and the raw `init_event`), what the harness itself requested (`requested.{mcp_servers,tools,skills,agents,disallowed_tools}`, from the run's own recorded argv, populated even when the init event announced nothing), and the exact `system_prompt` text actually sent (`text`, `source`, `chars`, `sha256`, `recorded_sha256`) — so a harness-vs-native discrepancy can be diagnosed without hand-reading a Claude transcript. Works on a still-RUNNING run.

`harness_send_message(run_id, prompt)` sends a follow-up to a **finished** run: it resumes the origin's session (keeping its isolation) and returns a new RUNNING run with its own `run_id`; then use `harness_wait_run` or `harness_poll_run` on it.

It prints one `harness_poll_run`-shaped JSON object plus `waited_s` on stdout and never cancels the run. Without `--timeout` it blocks until the run ends; `--interval` defaults to 2 (minimum 0.2).

| exit code | meaning |
| --------- | ------- |
| 0 | run COMPLETED |
| 1 | run FAILED |
| 2 | `--timeout` (if given) elapsed; the run is still RUNNING |
| 3 | run CANCELLED |
| 4 | error: unknown run id, unreadable artifacts dir, or invalid arguments |

## Plugin agents

A plugin's own `agents/*.md` definitions are discovered alongside project (`.claude/agents/`) and user-scope agents. A plugin-sourced agent is listed by `harness_list_agents` and started via `harness_start_agent` under a qualified name, `<plugin>:<name>` (e.g. `agent-harness:general-purpose`) — not the bare `<name>` that project- and user-scope agents use. Its `tools:` frontmatter field binds to the run's `--tools` allowlist, the same as any other agent definition: a `tools:` line is the complete list of tools the run gets, and a definition without one gets every tool. Its `disallowedTools:` frontmatter field is a denylist on top of that: a tool named in both `tools:` and `disallowedTools:` is denied, and with no `tools:` line at all the run gets every tool except the ones `disallowedTools:` names. Claude itself enforces the denylist (a call to a denied tool fails as unavailable). `harness_inspect_run` reads it back as `requested.disallowed_tools` from the run's top-level `--disallowedTools` flag, for both the `--agents` payload and materialized-file carriers.

This plugin ships three standard agents under `agents/`:

- `agent-harness:general-purpose` — open-ended, multi-step work; no `tools:` line, so every tool (edits files, runs commands).
- `agent-harness:Explore` — locating code and answering questions about a codebase; read-only, no shell (`tools: Read, Glob, Grep, WebFetch, WebSearch`).
- `agent-harness:Plan` — designing an implementation approach before code is written; the same read-only allowlist, no shell.

None of the three pins a `model:` or `effort:`; a run inherits them from the parent session or the tool arguments.

**Overriding a shipped agent:** discovery keys everything by qualified name and resolves project scope, then user scope, then plugin scope, first writer wins. Project and user agent files are named unprefixed — `.claude/agents/general-purpose.md` is discovered as `general-purpose`, a different key from `agent-harness:general-purpose` and not a replacement for it. To override a plugin-shipped agent, give a project or user `.md` file the plugin's qualified name explicitly, e.g. `name: agent-harness:general-purpose` in its frontmatter — that definition then wins over the plugin's own under that same key.

Colon-qualified plugin-agent dispatch is covered by a fake-CLI test in CI plus an opt-in live test (`tests/test_live_claude.py::test_live_start_plugin_agent_colon_qualified`), deselected by default. To run it, provision a real `CLAUDE_CONFIG_DIR` holding a `harness-live-fixture@<any>` plugin install with `agents/echo.md` (`model: haiku`, `tools: Read`), `settings.json` enabling that plugin, and real credentials; point `HARNESS_LIVE_PLUGIN_CONFIG_DIR` at it and run `python -m pytest -m live tests/test_live_claude.py`.

## Native subagent dispatch is disabled

A `PreToolUse` hook denies Claude Code's built-in subagent tool (`Agent`, and its legacy alias `Task`) unconditionally, in every session the plugin is installed in — no opt-in flag. The denial names `harness_start_agent` and `harness_start_prompt` as the replacement: use one of those instead of the native tool. This applies inside a `harness_start_agent`-launched child too, since it loads the plugin (and this same hook) just like the parent session — a harness-launched agent cannot escape the restriction by spawning a native subagent of its own.

## MCP servers

A `harness_start_agent` run inherits the *parent* Claude Code session's own MCP-server set — project (`.mcp.json`, approved), user/local (`~/.claude.json`), and every enabled plugin's own manifest — rebuilt explicitly from those files and passed to the child's launch, rather than left for the child to discover (and possibly still be waiting to connect to at its first turn) on its own. A plugin's servers are keyed `plugin_<plugin>_<server>` (e.g. `plugin_agent-serena-wrapper_serena`) to avoid name collisions across plugins and to keep the native `mcp__plugin_<plugin>_<server>__*` tool names, with one exception: the `agent-harness` plugin's own `harness` server keeps the literal bare key `harness`, since that is the name a run's own tool calls use (`mcp__harness__*`) and the name `.seretos/harness.yml`'s `canSpawn` grants.

`.seretos/harness.yml`'s `agents: <name>: mcpServers: {add, remove}` patches this rebuilt set by those same keys — `remove: [plugin_fixture_fsrv]`, not `remove: [fsrv]`. Whether the `harness` server itself is present in a config-driven launch follows only `canSpawn: true`/`false` on that entry; this plugin does not override the pinned `lib-python-harness`'s own `canSpawn` default. A `harness.yml` entry (or `defaults:`) makes the launch `--strict-mcp-config`, so only that entry's own `mcpServers`/`canSpawn` values decide the child's final server set — no other parent server survives being left unmentioned once an entry applies to that agent.

## Agent identity inside a run

A `harness_start_agent` child's own environment carries `HARNESS_LAUNCHED_AGENT=<qualified_name>` — the same qualified name `harness_start_agent`'s own `agent` argument was given: `<plugin>:<name>` for a plugin-sourced agent (e.g. `agent-harness:general-purpose`), or the bare `<name>` for a project- or user-scope agent (there is no colon to split off). A hook running inside that child — including the child's own `Stop` hook — can read this from its process environment to tell which agent it is running as; native Claude Code's own `SubagentStop` payload has an `agent_type` field, but a harness-launched child is a fully independent top-level `claude` process with no equivalent field of its own, so this env var is the only signal.

A `harness_send_message` follow-up on that same run keeps the identical value — a follow-up turn is still that same agent. `harness_start_prompt` runs are always CLEAN (no agent, ever): their children never carry `HARNESS_LAUNCHED_AGENT`, even when the server process's own environment happens to have one set (e.g. the server itself is running inside an agent session) — a stale inherited value is stripped, never leaked into a prompt child that would otherwise report a false identity.

The bare name is derived as `value.split(":")[-1]` wherever this plugin needs it; `HARNESS_LAUNCHED_AGENT` itself is always the qualified form. One caveat: for a *materialized* agent-definition launch (the carrier of last resort, used when an agent definition sets a field the `--agents` JSON payload schema rejects, e.g. `hooks`/`mcpServers`), the qualified name's `:` is recovered by reversing the `__` the materialized file's own stem uses in place of it — a bare agent name that itself happens to contain `__` is a known, documented ambiguity in that one carrier, not something this plugin can distinguish from a real qualified-name separator.
