"""MCP tool surface over lib_python_harness: one tool per public lib entry point."""
from __future__ import annotations

import functools
import inspect
import os
import sys
from functools import partial
from typing import Annotated, Any

import anyio.to_thread
from lib_python_harness import (
    HarnessError,
    HostContext,
    Isolation,
    RunSpec,
    RunState,
    discover,
)
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import Field

from harness_plugin.host_context import (
    load_session_context,
    probe_warning,
)
from harness_plugin.runs import (
    artifacts_root,
    harness,
    inspect_run,
    launched_fields,
    remember_effort_source,
    run_to_dict,
    start_agent,
    StartRefused,
    summary_to_dict,
)

mcp = FastMCP("harness")

# "Accepted" per value, from each value's own authority (plan "Who decides
# accepted"), not this repo's own guess:
# - effort/model are enumerated from the pinned lib_python_harness's own hard
#   validator (providers/claude_cli.py's _EFFORT_VALUES/_MODEL_ALIASES), which
#   raises UnsupportedByProvider before a child ever launches -- see
#   tests/test_mcp_tools.py::test_documented_values_match_lib_validator.
#   `inherit` is excluded from the model aliases: it is an agent-definition
#   sentinel, not a CLI-accepted value (the lib's own comment, claude_cli.py:
#   33-35). model's *last* tuple element is a full-model-id example, not an
#   alias (see _MODEL_VALUES_TEXT's `[:-1]`/`[-1]` split below).
# - permission_mode has no lib validator (claude_cli.py passes it straight
#   through), so it is enumerated from the live `claude` CLI instead --
#   tests/test_live_claude.py::test_accepted_values_match_the_cli. "default"
#   is NOT among the tokens `--help` prints under `--permission-mode`'s own
#   "(choices: ...)" text, but it is genuinely accepted by the CLI: a real run
#   started with `--permission-mode default` completes, and its stream-json
#   init event reports the value straight back (`"permissionMode":"default"`)
#   -- probe-verified (_probe_cli_accepts), not just trusted from --help text.
_ACCEPTED_VALUES: dict[str, tuple[str, ...]] = {
    "effort": ("low", "medium", "high", "xhigh", "max"),
    "permission_mode": (
        "default", "acceptEdits", "auto", "bypassPermissions", "dontAsk", "manual", "plan",
    ),
    # Last element is a full-model-id *example*, not an accepted alias -- keep
    # it last, _MODEL_VALUES_TEXT and test_documented_values_match_lib_validator
    # both rely on that position (`[:-1]` for aliases, `[-1]` for the example).
    "model": ("opus", "sonnet", "haiku", "fable", "default", "claude-haiku-4-5-20251001"),
}

_EFFORT_VALUES_TEXT = ", ".join(_ACCEPTED_VALUES["effort"])
_MODEL_VALUES_TEXT = (
    ", ".join(_ACCEPTED_VALUES["model"][:-1])
    + ", or a full model id with a segment (split on non-alphanumeric "
    + "characters, e.g. `-`, `.`, `/`) that exactly equals "
    + "claude/anthropic/opus/sonnet/haiku/fable "
    + f"(e.g. {_ACCEPTED_VALUES['model'][-1]}); anything else is refused before launch"
)
_PERMISSION_MODE_VALUES_TEXT = ", ".join(_ACCEPTED_VALUES["permission_mode"])

_EFFORT_DESC_AGENT = (
    f"Effort level passed to the child as `--effort <value>` (accepted values: "
    f"{_EFFORT_VALUES_TEXT}). Unset, it falls back to the parent session's "
    f"effort; if that is absent too, the effort is left unspecified and the "
    f"CLI's own default applies."
)

_EFFORT_DESC_PROMPT = (
    f"Effort level passed to the child as `--effort <value>` (accepted values: "
    f"{_EFFORT_VALUES_TEXT}). This tool never reads the parent session's "
    f"context, so leaving it unset sends no `--effort` flag at all and the "
    f"CLI's own default effort is used -- nothing here is inherited."
)

_MODEL_DESC_AGENT = (
    f"Model for the run (accepted values: {_MODEL_VALUES_TEXT}). Unset, the "
    f"model falls back to the parent session's model; if the parent session "
    f"has none either, model resolution fails and the run is refused."
)

_MODEL_DESC_PROMPT = (
    f"Model for the run (accepted values: {_MODEL_VALUES_TEXT}). Required: "
    f"harness_start_prompt has no session context to fall back to, so this "
    f"must always be supplied explicitly."
)

_PERMISSION_MODE_DESC_AGENT = (
    f"Permission mode for the run (accepted values: "
    f"{_PERMISSION_MODE_VALUES_TEXT}). Unset, it inherits the parent session's "
    f"permission mode; the call is refused if neither an explicit value nor an "
    f"inherited one is available."
)


def _tool_errors(fn):
    """Re-raise lib errors as ToolError carrying the exception class name."""
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except HarnessError as exc:
                raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    else:

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except HarnessError as exc:
                raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    return wrapper


@mcp.tool()
@_tool_errors
def harness_list_agents(cwd: str | None = None) -> dict[str, Any]:
    """List the subagent definitions (project, user and plugin scope) visible from `cwd`
    (default: the parent session's cwd, else CLAUDE_PROJECT_DIR, else the server's working
    directory). The used cwd and the `context_source` are echoed back."""
    data, source = load_session_context()
    used = (
        cwd
        or (data or {}).get("cwd")
        or os.environ.get("CLAUDE_PROJECT_DIR")
        or os.getcwd()
    )
    found = discover(HostContext(cwd=used))
    return {
        "cwd": used,
        "context_source": source,
        "agents": [
            {
                "qualified_name": name,
                "name": d.name,
                "description": d.description,
                "source_scope": d.source_scope,
                "path": str(d.path),
                "model": d.model,
            }
            for name, d in found.items()
        ],
    }


@mcp.tool()
@_tool_errors
def harness_start_agent(
    agent: str,
    cwd: str | None = None,
    model: Annotated[str | None, Field(description=_MODEL_DESC_AGENT)] = None,
    permission_mode: Annotated[
        str | None, Field(description=_PERMISSION_MODE_DESC_AGENT)
    ] = None,
    effort: Annotated[str | None, Field(description=_EFFORT_DESC_AGENT)] = None,
    label: str | None = None,
    prompt: str | None = None,
) -> dict[str, Any]:
    """Start a discovered subagent (by qualified_name) as a background run and return its
    run_id immediately; use harness_poll_run or harness_wait_run for the result. The run
    inherits the parent session's permission mode, model, effort and cwd (collected by the
    plugin hook); explicit arguments override them, and an agent definition's own `effort:`
    (or `model:`/`permission_mode:`) outranks both. Refuses when the session context cannot
    be determined. `context_source`, `cwd`, `permission_mode`, `model`, `effort` and
    `effort_source` are echoed back -- `effort_source` names where the launched effort came
    from: `agent_definition`, `argument`, `parent_session` or `none`. An optional short
    `label` names the run and shows up in harness_list_runs. An optional `prompt` is the
    run's task (the user message); the agent definition's body stays the system prompt.
    Without `prompt` the run gets a default task. The run's MCP-server set is the parent
    session's own (project, user/local, enabled plugins, harness), rebuilt explicitly and
    passed to the launch rather than left for the child to discover on its own. The child's
    own environment carries `HARNESS_LAUNCHED_AGENT=<qualified_name>` (and so does any
    `harness_send_message` follow-up on it), so a hook running inside it -- e.g. its own
    `Stop` hook -- can tell which agent it is running as; see README.md's "Agent identity
    inside a run" section."""
    try:
        result, used, source = start_agent(
            agent,
            cwd=cwd,
            model=model,
            permission_mode=permission_mode,
            effort=effort,
            label=label,
            prompt=prompt,
        )
    except StartRefused as exc:
        raise ToolError(str(exc)) from exc
    return run_to_dict(result, cwd=used, context_source=source)


@mcp.tool()
@_tool_errors
def harness_start_prompt(
    prompt: str,
    model: Annotated[str, Field(description=_MODEL_DESC_PROMPT)],
    effort: Annotated[str | None, Field(description=_EFFORT_DESC_PROMPT)] = None,
    system_prompt: str | None = None,
    cwd: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Start an ad-hoc prompt in a clean (no memory, no project config) run and return its
    run_id immediately. With `cwd` unset the run gets a fresh empty temp directory; a given
    `cwd` must exist, be empty and not sit inside a git repository. No permission mode is
    ever sent to the child: `--permission-mode` is never passed, and the parent session's
    permission mode is not inherited -- the CLI's own default permission mode is used
    instead. `effort` and `effort_source` are echoed back in the answer (`effort_source`
    is `argument` when `effort` was passed, else `none` -- this tool never reads the
    parent session, so `none` here always means "CLI default", never a dropped
    inheritance). An optional short `label` names the run and shows up in
    harness_list_runs."""
    spec = RunSpec(
        prompt=prompt,
        isolation=Isolation.CLEAN,
        model=model,
        effort=effort,
        system_prompt=system_prompt,
        cwd=cwd,
        label=label,
        artifacts_dir=artifacts_root(),
    )
    result = harness().start(spec)
    remember_effort_source(result.run_id, "argument" if effort else "none")
    return run_to_dict(result)


@mcp.tool()
@_tool_errors
async def harness_poll_run(run_id: str) -> dict[str, Any]:
    """Return the current state (and result, once finished) of a run without waiting for
    the run to progress -- but it may take a few seconds to finalize a run whose process
    lingers after writing its result (the lib's post-completion grace period; see #68).
    While the run is RUNNING, `event_count` and `last_event_at` show progress: they only
    advance while `state` is RUNNING, so a growing count means the run is working and a
    frozen one means it may be hung; once the run is terminal they reset to 0 and None."""
    h = harness()
    result = await anyio.to_thread.run_sync(partial(h.wait, run_id, 0))
    return run_to_dict(result)


@mcp.tool()
@_tool_errors
def harness_list_runs() -> dict[str, Any]:
    """List all recorded runs as compact rows (run_id, state, model, cwd, created_at, label).
    Use it when you lost the run_id (e.g. after context compaction) to find a run again,
    then pass it to harness_stop_run, harness_poll_run or harness_cleanup_run. Rows
    deliberately carry no result text or usage; poll a run for those."""
    return {"runs": [summary_to_dict(s) for s in harness().list_runs()]}


@mcp.tool()
@_tool_errors
def harness_send_message(run_id: str, prompt: str) -> dict[str, Any]:
    """Send a follow-up `prompt` to a run that has already finished (only finished runs;
    a RUNNING run is refused) and return immediately. It resumes the origin run's session
    and returns a new run: a new `run_id` (with `resumed_from` naming the origin run_id) in
    state RUNNING. The origin run's isolation is preserved: the follow-up runs with the same
    clean/agent flags and cwd as the origin. Afterwards use harness_wait_run or
    harness_poll_run on the new run_id to get the reply."""
    if not prompt.strip():
        raise HarnessError("prompt must not be empty")
    # The origin's argv (and thus its launched `effort` value) is replayed
    # verbatim by start_resume() (harness.py's provider_argv replay), so
    # launched_fields() answers `effort` for the new run with no extra plumbing
    # -- but `effort_source` lives only on the origin's own record and is not
    # copied by the library, so it must be looked up and re-stamped explicitly.
    origin_source = launched_fields(run_id)["effort_source"]
    result = harness().start_resume(run_id, prompt)
    remember_effort_source(result.run_id, origin_source)
    return run_to_dict(result, resumed_from=run_id)


@mcp.tool()
@_tool_errors
async def harness_wait_run(run_id: str, timeout_seconds: float = 300.0) -> dict[str, Any]:
    """Wait up to `timeout_seconds` (default 300) for the run to finish and return its
    result. The time limit ends only the waiting, never the run: nothing is cancelled. If
    it expires first the run is still RUNNING and the answer says so, with liveness fields
    (`duration_s`, `event_count`, `last_event_at`, `last_activity` = the last tool/command
    the run used) and a `next_step` hint. Only harness_stop_run cancels a run. To keep
    waiting past a tool call, run the shell command `harness wait <run_id>` in the
    background (blocks until the run ends; see the `harness-wait` skill), or call this
    tool again; harness_poll_run checks without waiting."""
    h = harness()
    result = await anyio.to_thread.run_sync(partial(h.wait, run_id, timeout_seconds))
    if result.state == RunState.RUNNING:
        return run_to_dict(
            result,
            next_step=(
                f"The run is still RUNNING; the timeout only ended the waiting and nothing "
                f"was cancelled. Keep waiting with `harness wait {run_id}` (run it via Bash "
                f"in the background) or call harness_wait_run again. Only harness_stop_run "
                f"cancels the run."
            ),
        )
    return run_to_dict(result)


@mcp.tool()
@_tool_errors
def harness_stop_run(run_id: str) -> dict[str, Any]:
    """Stop a RUNNING run (terminal state CANCELLED). Errors on an already finished run."""
    return run_to_dict(harness().stop(run_id))


@mcp.tool()
@_tool_errors
def harness_cleanup_run(run_id: str) -> dict[str, Any]:
    """Forget a run's record. Artifacts on disk are kept; the run cannot be polled afterwards."""
    harness().cleanup(run_id)
    return {"run_id": run_id, "cleaned": True}


@mcp.tool()
@_tool_errors
def harness_inspect_run(run_id: str) -> dict[str, Any]:
    """Answer, for one run, what it announced, what the harness asked for, and what
    system prompt was actually sent -- so a harness-vs-native discrepancy (e.g. a
    tool/skill/agent count mismatch) can be diagnosed without hand-reading a Claude
    transcript. `announced.{mcp_servers,tools,skills,agents}` are the name lists this
    run's own CLI process announced at startup, read from its events.jsonl init event
    (`init_event`, verbatim, or null if none was seen yet), with `announced_counts`
    alongside. `requested.{mcp_servers,tools,skills,agents,disallowed_tools}` are the same
    categories the harness itself put into this run, read back from its own recorded argv --
    populated even when the init event announced nothing, since it comes from a
    different artifact. `disallowed_tools` is plain argv like the rest: it is read
    from the top-level `--disallowedTools` CSV flag, populated the same way for both
    the `--agents` payload and materialized-agent-file carriers. `system_prompt` carries the
    exact text sent, with `source`
    naming which carrier supplied it (--system-prompt, --agents, or the materialized
    agent file), `chars`, `sha256`, and `recorded_sha256` (the run's own recorded
    digest) alongside. Works on a still-RUNNING run; errors on an unknown or
    cleaned-up run_id."""
    return inspect_run(run_id)


def main() -> None:
    warning = probe_warning()
    if warning:
        print(warning, file=sys.stderr, flush=True)  # stdout is the MCP transport
    mcp.run()
