"""Run helpers shared by the MCP server and the `wait` subcommand.

Deliberately free of mcp/FastMCP imports: the subcommand must start without them."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any

from lib_python_harness import ClaudeCliProvider, FileRunStore, Harness, HarnessError

_HARNESS: Harness | None = None

# Public contract name (#51): the env var a `harness_start_agent` child (and any
# `harness_send_message` follow-up on it) finds its own qualified agent name under,
# so a hook running inside that child -- e.g. its own `Stop` hook -- can tell which
# agent it is running as. See README.md's "Agent identity inside a run" section.
LAUNCHED_AGENT_ENV = "HARNESS_LAUNCHED_AGENT"


def _with_identity(env: dict[str, str], name: str | None) -> dict[str, str]:
    """`env` (a `LaunchPlan.env`, never mutated in place) with `LAUNCHED_AGENT_ENV`
    set to `name` when truthy, or popped otherwise. The pop branch matters even
    though `_scrub_env()` only ever copies the *current* process's own env: a
    server running inside an agent session would otherwise leak its own identity
    into a CLEAN `harness_start_prompt` child, or a resumed `harness_send_message`
    follow-up on a prompt-origin run would inherit a stale value."""
    env = dict(env)
    if name:
        env[LAUNCHED_AGENT_ENV] = name
    else:
        env.pop(LAUNCHED_AGENT_ENV, None)
    return env


def _recover_launched_agent_name(provider_argv: list[str]) -> str | None:
    """The origin run's own qualified agent name, recovered from its recorded
    argv -- the only per-run carrier `build_resume_plan` receives. No `--agent`
    means the origin was a `harness_start_prompt` (CLEAN, no agent) run: `None`.
    With `--agents` present (the payload carrier), `--agent`'s value is already
    the qualified name verbatim (`_build_agent_payload`/`dispatch_mode`, plan
    Approach) -- used as-is, so a bare agent name that itself contains `__`
    (e.g. `my__agent`) is not mistaken for a materialized stem. Without
    `--agents` (the materialized carrier, last resort), `--agent`'s value is
    the file stem `_agent_file_stem` wrote (`:` replaced by `__`), reversed here
    with a single replacement -- the one documented, known-ambiguous case is a
    materialized-carrier agent whose own bare name contains `__` (README)."""
    agent = _flag(provider_argv, "--agent")
    if agent is None:
        return None
    if "--agents" in provider_argv:
        return agent
    return agent.replace("__", ":", 1)


class _AgentIdentityProvider(ClaudeCliProvider):
    """Wraps `ClaudeCliProvider` to inject `LAUNCHED_AGENT_ENV` into a launched
    child's (and its resumed follow-ups') environment (#51). `RunSpec` has no
    `env` field, `LaunchPlan` is frozen and `_scrub_env` is internal, so this
    injected provider is v0.0.8's only per-run override point; `ClaudeCliProvider`
    itself is stateless, so subclassing it carries no extra state to manage."""

    def build_launch_plan(self, spec, **kwargs):
        plan = super().build_launch_plan(spec, **kwargs)
        return dataclasses.replace(plan, env=_with_identity(plan.env, spec.agent_name))

    def build_resume_plan(self, *, provider_argv, **kwargs):
        plan = super().build_resume_plan(provider_argv=provider_argv, **kwargs)
        name = _recover_launched_agent_name(provider_argv)
        return dataclasses.replace(plan, env=_with_identity(plan.env, name))


# Init-event key(s) each announced category may be spelled under, primary spelling
# first. NOTE: only `mcp_servers` carries a known alternate spelling (`mcpServers`) --
# a differently-spelled real key for `tools`/`skills`/`agents` is indistinguishable
# from genuine absence (the init event simply omits that key). Known, accepted
# limitation (see the plan's "premises verified"), not a bug to fix here.
_INIT_NAME_FIELDS: dict[str, tuple[str, ...]] = {
    "mcp_servers": ("mcp_servers", "mcpServers"),
    "tools": ("tools",),
    "skills": ("skills",),
    "agents": ("agents",),
}


def _names(item: Any) -> Any:
    """One item of an init-event announcement list: a plain string is itself; a
    dict-shaped item (e.g. `{"name": "Foo", ...}`) yields `item["name"]`."""
    if isinstance(item, dict):
        return item.get("name")
    return item


def _announced_names(init_event: dict[str, Any] | None, key: str) -> list[str]:
    if init_event is None:
        return []
    for alias in _INIT_NAME_FIELDS[key]:
        if alias in init_event:
            raw = init_event[alias]
            if not isinstance(raw, list):
                return []
            return [name for name in (_names(item) for item in raw) if name is not None]
    return []


def _read_init_event(events_path: str | Path | None) -> dict[str, Any] | None:
    """The first `{"type": "system", "subtype": "init"}` line of `events_path`, or
    `None` if the file is missing, empty, or has no such line (yet -- a still-RUNNING
    run's init line may simply not have been written when this is called, though in
    practice it is emitted before anything else)."""
    if not events_path:
        return None
    path = Path(events_path)
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "system" and event.get("subtype") == "init":
            return event
    return None


def _flag(argv: list[str], name: str) -> str | None:
    """The token immediately following `name` in `argv`, or `None` if `name` is
    absent (or is the last token, which never happens for a real flag/value pair)."""
    if name not in argv:
        return None
    idx = argv.index(name)
    if idx + 1 >= len(argv):
        return None
    return argv[idx + 1]


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _requested_names(argv: list[str]) -> dict[str, list[str]]:
    """What the harness itself asked for, read back from this run's own recorded argv
    -- independent of what the init event announced (see module docstring / plan): the
    `--agents` JSON's keys and each entry's own `skills` and `disallowedTools`, the
    `--agent` value, the `--mcp-config` JSON's `mcpServers` keys, and the `--tools` CSV.
    `disallowed_tools` mirrors `skills`: read only from the `--agents` payload's
    per-agent `disallowedTools` key, so a run dispatched via the materialized `.md`
    file carrier (no `--agents` payload) reports `[]`, same as a run with no denylist
    at all -- the two are indistinguishable at this carrier (see plan, #56)."""
    agents: list[str] = []
    skills: list[str] = []
    disallowed: list[str] = []
    agents_json = _flag(argv, "--agents")
    if agents_json:
        try:
            payload = json.loads(agents_json)
        except json.JSONDecodeError:
            payload = {}
        for name, fields in payload.items():
            agents.append(name)
            if isinstance(fields, dict):
                skills.extend(fields.get("skills") or [])
                disallowed.extend(fields.get("disallowedTools") or [])
    agent_flag = _flag(argv, "--agent")
    if agent_flag:
        agents.append(agent_flag)

    mcp_servers: list[str] = []
    mcp_config = _flag(argv, "--mcp-config")
    if mcp_config:
        try:
            parsed = json.loads(mcp_config)
        except json.JSONDecodeError:
            parsed = {}
        mcp_servers = list((parsed.get("mcpServers") or {}).keys())

    tools: list[str] = []
    tools_flag = _flag(argv, "--tools")
    if tools_flag is not None:
        tools = [t for t in tools_flag.split(",") if t]

    return {
        "agents": _dedup(agents),
        "skills": _dedup(skills),
        "disallowed_tools": _dedup(disallowed),
        "mcp_servers": mcp_servers,
        "tools": tools,
    }


def _resolve_system_prompt(record: dict[str, Any], argv: list[str]) -> dict[str, Any]:
    """The exact system-prompt text this plugin sent, for whichever of the three
    mutually-exclusive carriers this run used (see plan Approach): `--system-prompt`
    (CLEAN runs), the `--agents` payload's `prompt` key, or the materialized
    `<run_dir>/agents/.claude/agents/<stem>.md` file's body (frontmatter stripped).
    `recorded_sha256` is the record's own `system_prompt_sha256` -- computed by the
    library from the same body for all three carriers -- so it and the freshly
    computed `sha256` prove the returned text is the text the run actually started
    with."""
    text: str | None = None
    source: str | None = None

    sp_flag = _flag(argv, "--system-prompt")
    if sp_flag is not None:
        text, source = sp_flag, "--system-prompt"
    else:
        agent_flag = _flag(argv, "--agent")
        agents_json = _flag(argv, "--agents")
        if agent_flag and agents_json:
            try:
                payload = json.loads(agents_json)
            except json.JSONDecodeError:
                payload = {}
            agent_payload = payload.get(agent_flag)
            if isinstance(agent_payload, dict) and "prompt" in agent_payload:
                text, source = agent_payload["prompt"], "--agents"
        if text is None and agent_flag and record.get("run_dir"):
            materialized_path = (
                Path(record["run_dir"]) / "agents" / ".claude" / "agents" / f"{agent_flag}.md"
            )
            if materialized_path.exists():
                raw = materialized_path.read_text(encoding="utf-8")
                body = raw.split("---", 2)[-1].lstrip("\n") if raw.startswith("---") else raw
                text, source = body, f"materialized:{materialized_path}"

    chars = len(text) if text is not None else 0
    sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest() if text is not None else None

    return {
        "text": text,
        "source": source,
        "chars": chars,
        "sha256": sha256,
        "recorded_sha256": record.get("system_prompt_sha256"),
    }


def inspect_run(run_id: str) -> dict[str, Any]:
    """What a single run announced, requested, and used as its system prompt -- see
    `harness_inspect_run`'s docstring in `server.py` for the answer shape. Works on a
    still-RUNNING run; an unknown or cleaned-up `run_id` raises `HarnessError`."""
    h = harness()
    record = h.store.get(run_id)
    if record is None:
        raise HarnessError(f"unknown run_id: {run_id}")

    init_event = _read_init_event(record.get("events_path"))
    announced = {key: _announced_names(init_event, key) for key in _INIT_NAME_FIELDS}
    announced_counts = {key: len(names) for key, names in announced.items()}

    argv = record.get("argv") or []
    requested = _requested_names(argv)
    system_prompt = _resolve_system_prompt(record, argv)

    return {
        "run_id": run_id,
        "state": jsonable(record.get("state")),
        "announced": announced,
        "announced_counts": announced_counts,
        "requested": requested,
        "init_event": init_event,
        "system_prompt": system_prompt,
    }


def launched_fields(run_id: str) -> dict[str, Any]:
    """`model`/`effort`/`permission_mode` for `run_id`, read back from its own
    recorded argv -- the one path every answer about a run's launched values goes
    through (plan Approach), replacing the four call sites that used to pass their
    own extras. `effort_source` comes from the record itself (see
    `remember_effort_source`) and degrades to `"unknown"` when the record predates
    this field or the run is gone -- never an error, since a still-valid poll/wait
    on an older run must keep working."""
    record = harness().store.get(run_id)
    argv = (record or {}).get("argv") or []
    return {
        "model": _flag(argv, "--model"),
        "effort": _flag(argv, "--effort"),
        "permission_mode": _flag(argv, "--permission-mode"),
        "effort_source": (record or {}).get("effort_source") or "unknown",
    }


def remember_effort_source(run_id: str, source: str) -> None:
    """Persist `source` (one of `argument`/`agent_definition`/`parent_session`/
    `none`) on `run_id`'s own record -- the only per-run carrier that already
    crosses process boundaries (`FileRunStore` re-reads `record.json` from disk,
    no in-process cache), so a later `harness_poll_run`/`harness_wait_run`, even
    from a different process (e.g. `harness wait`), can answer where the launched
    effort came from. Read-modify-put, called right after `start()`/
    `start_resume()`; a no-op if the record is already gone."""
    h = harness()
    record = h.store.get(run_id)
    if record is None:
        return
    record["effort_source"] = source
    h.store.put(run_id, record)


def artifacts_root() -> Path:
    override = os.environ.get("HARNESS_ARTIFACTS_DIR")
    root = Path(override) if override else Path.home() / ".agent-harness" / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def claude_argv() -> list[str]:
    raw = os.environ.get("HARNESS_CLAUDE_ARGV")
    return [str(a) for a in json.loads(raw)] if raw else ["claude"]


def harness() -> Harness:
    """Lazy singleton: poll/wait/stop depend on the in-process Popen map."""
    global _HARNESS
    if _HARNESS is None:
        _HARNESS = Harness(
            store=FileRunStore(artifacts_root()),
            claude_argv=claude_argv(),
            provider=_AgentIdentityProvider(),
        )
    return _HARNESS


def jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    return value


def run_to_dict(result: Any, **extra: Any) -> dict[str, Any]:
    out = {
        "run_id": result.run_id,
        "session_id": result.session_id,
        "state": jsonable(result.state),
        "text": result.text,
        "is_error": result.is_error,
        "subtype": result.subtype,
        "structured_output": result.structured_output,
        "usage": result.usage,
        "cost": result.cost,
        "transcript_path": jsonable(result.transcript_path),
        "duration_s": result.duration_s,
        "event_count": result.event_count,
        "last_event_at": result.last_event_at,
        "last_activity": result.last_activity,
    }
    if result.run_id:
        out.update(launched_fields(result.run_id))
    out.update(extra)
    return out


def summary_to_dict(summary: Any) -> dict[str, Any]:
    """Compact list row: identifying fields only, deliberately no text/usage."""
    return {
        "run_id": summary.run_id,
        "state": jsonable(summary.state),
        "model": summary.model,
        "cwd": jsonable(summary.cwd),
        "created_at": summary.created_at,
        "label": summary.label,
    }
