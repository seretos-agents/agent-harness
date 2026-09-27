"""MCP-level tests: a real `python -m harness_plugin` server over stdio,
talking to the fake claude CLI from tests/fixtures/fake_claude.py."""
import hashlib
import json
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import pytest
from conftest import SESSION_ID, plant_session_file
from fixtures.fake_claude import (
    ALT_INIT_MCP_SERVERS,
    ALT_INIT_SHAPE,
    ALT_INIT_TOOLS,
    INIT_ANNOUNCEMENTS,
    NO_INIT_NAMES,
)
from mcp import ClientSession
from mcp.client.stdio import stdio_client

EXPECTED_TOOLS = {
    "harness_list_agents",
    "harness_start_agent",
    "harness_start_prompt",
    "harness_poll_run",
    "harness_list_runs",
    "harness_wait_run",
    "harness_stop_run",
    "harness_cleanup_run",
    "harness_send_message",
    "harness_inspect_run",
}
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}


@asynccontextmanager
async def _session(params):
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def _run(scenario, params):
    async def main():
        with anyio.fail_after(60):
            async with _session(params) as session:
                return await scenario(session)

    return anyio.run(main)


async def _call(session, name, **arguments):
    """Returns (is_error, text, payload). payload is the JSON object the tool
    returned (None for errors)."""
    result = await session.call_tool(name, arguments)
    text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
    payload = None
    if not result.isError:
        payload = result.structuredContent
        if payload is None:
            payload = json.loads(text)
    return bool(result.isError), text, payload


async def _poll_until_terminal(session, run_id, budget=20.0):
    deadline = time.monotonic() + budget
    while True:
        is_error, text, payload = await _call(session, "harness_poll_run", run_id=run_id)
        assert not is_error, text
        if payload["state"] in TERMINAL:
            return payload
        assert time.monotonic() < deadline, f"run never finished: {payload}"
        await anyio.sleep(0.2)


def test_tools_list_exposes_harness_tools_and_no_ping(server_params):
    async def scenario(session):
        return (await session.list_tools()).tools

    tools = _run(scenario, server_params)
    names = {t.name for t in tools}
    assert EXPECTED_TOOLS <= names
    assert "ping" not in names
    for t in tools:
        assert len((t.description or "").strip()) >= 20, f"{t.name} has no real description"
    wait_desc = next(t for t in tools if t.name == "harness_wait_run").description.lower()
    desc = {t.name: (t.description or "").lower() for t in tools}
    # Phrase-level checks: the wording must explain the purpose / semantics, not just
    # mention a token.
    list_desc = desc["harness_list_runs"]
    assert re.search(r"(lost|forgot\w*|no longer (have|know))\W+(\w+\W+){0,3}run_id", list_desc), (
        "list description must explain recovering a lost run_id"
    )
    assert re.search(r"harness_(stop|poll|wait|cleanup)_run", list_desc), (
        "list description must say which tools the recovered run_id feeds"
    )
    poll_desc = desc["harness_poll_run"]
    assert re.search(r"event_count[^.]*last_event_at|last_event_at[^.]*event_count", poll_desc)
    assert re.search(r"(only|just)\W+(\w+\W+){0,4}(advance|grow|increase|change)", poll_desc) and (
        "running" in poll_desc
    ), "poll description must say the progress fields only advance while RUNNING"
    send_desc = desc["harness_send_message"]
    assert re.search(r"(only|just)\W+(\w+\W+){0,4}(finished|terminal|completed)", send_desc), (
        "send_message description must say it is only for finished runs"
    )
    assert re.search(r"new\W+(\w+\W+){0,2}run(_id)?", send_desc), (
        "send_message description must say it returns a new run / run_id"
    )
    assert re.search(r"isolation[^.]*(preserv|kept|keeps|retain|inherit)", send_desc), (
        "send_message description must say the origin run's isolation is preserved"
    )
    assert "harness_wait_run" in send_desc and "harness_poll_run" in send_desc, (
        "send_message description must point to harness_wait_run / harness_poll_run"
    )
    start_agent = next(t for t in tools if t.name == "harness_start_agent")
    assert "prompt" in start_agent.inputSchema["properties"], (
        "harness_start_agent must expose an optional `prompt` argument"
    )
    assert "prompt" not in start_agent.inputSchema.get("required", [])
    start_agent_desc = desc["harness_start_agent"]
    assert re.search(r"prompt[^.]*(task|user message)", start_agent_desc), (
        "start_agent description must say `prompt` is the run's task / user message"
    )
    assert re.search(r"(body|definition)[^.]*system prompt", start_agent_desc), (
        "start_agent description must say the definition body stays the system prompt"
    )
    for name in ("harness_start_agent", "harness_start_prompt"):
        assert re.search(r"label[^.]*harness_list_runs|harness_list_runs[^.]*label", desc[name]), (
            f"{name} description must say the label shows up in harness_list_runs"
        )


_NEGATORS = re.compile(
    r"\b(not|never|no|isn'?t|doesn'?t|won'?t|ignor\w*|discard\w*|drop\w*)\b", re.I
)


def _clauses_mentioning(text, token_re):
    """The clause(s) of `text` that mention `token_re`, splitting on sentence- and
    clause-ending punctuation (including `;`) so a negation attached to one claim
    (e.g. "...; nothing is not inherited") cannot leak into an unrelated clause
    that shares the same sentence (e.g. "--permission-mode is forwarded...")."""
    token = re.compile(token_re, re.I)
    return [s for s in re.split(r"(?<=[.!?;])\s+", text) if token.search(s)]


def _assert_states_positively(desc, token_re, label):
    """At least one sentence mentions `token_re`, and none of those sentences carry
    a negation word -- kills a description that contains the right tokens while
    stating the opposite fact (e.g. "...--effort is ignored and never passed"),
    which bare substring-presence checks cannot tell apart from a true claim."""
    clauses = _clauses_mentioning(desc, token_re)
    assert clauses, f"{label}: no sentence mentions {token_re!r} in: {desc!r}"
    for clause in clauses:
        assert not _NEGATORS.search(clause), (
            f"{label}: sentence about {token_re!r} reads as a negative/opposite "
            f"claim: {clause!r}"
        )


_UNSET_MARKER = re.compile(
    r"\b(unset|not set|omit\w*|not (?:given|provided|passed|specified)|"
    r"falls?\s+back|default(?:s|ed)?\s+to|without an? explicit|left (?:unset|blank))\b",
    re.I,
)


def _assert_states_unset_behaviour(desc, fallback_re, label):
    """At least one clause both (a) marks itself as being about the omitted/unset
    case (`_UNSET_MARKER`: "unset", "not set", "omitted", "falls back", "defaults
    to", ...) and (b) names the concrete fallback (`fallback_re`, e.g. "parent
    session"), with no negation outside that marker's own matched text -- kills a
    description that never says what happens when the parameter is left out at
    all, and kills the token-soup case where the fallback word is merely listed as
    if it were an accepted value (e.g. permission_mode's `inherit` sitting in a
    bare "default, acceptEdits, plan, bypassPermissions, inherit" list): that
    clause carries no unset marker, so it does not qualify as a genuine
    inheritance-claim clause on its own. Negation is checked only outside the
    unset-marker's own match span, so idiomatic phrasing like "if not set,
    inherits ..." -- whose own "not" is what marks the unset case, not a negation
    of the fallback claim -- is not mistaken for a false/opposite claim."""
    clauses = re.split(r"(?<=[.!?;])\s+", desc)
    matches = []
    for clause in clauses:
        marker = _UNSET_MARKER.search(clause)
        if marker and re.search(fallback_re, clause, re.I):
            matches.append((clause, marker))
    assert matches, (
        f"{label}: no clause states both an unset/omitted marker and a "
        f"{fallback_re!r} fallback together in: {desc!r}"
    )
    for clause, marker in matches:
        remainder = clause[: marker.start()] + clause[marker.end():]
        assert not _NEGATORS.search(remainder), (
            f"{label}: unset-behaviour clause reads as a negative/opposite claim "
            f"outside its own unset marker: {clause!r}"
        )


def _assert_states_negatively(desc, token_re, label):
    """At least one sentence mentions `token_re` AND carries a negation word --
    proves the description actually states the negative fact (e.g. "the parent's
    mode is not inherited") rather than merely containing the bare tokens, which a
    description asserting the opposite ("...is forwarded from the parent") would
    also satisfy if only presence were checked."""
    clauses = _clauses_mentioning(desc, token_re)
    assert clauses, f"{label}: no sentence mentions {token_re!r} in: {desc!r}"
    assert any(_NEGATORS.search(clause) for clause in clauses), (
        f"{label}: no sentence about {token_re!r} carries a negation -- can't tell "
        f"'is inherited'/'is passed' from 'is not' from bare tokens alone: {clauses!r}"
    )


_COMMON_WORD_TOKENS = {"max", "default", "auto", "manual"}


def _token_present(desc, token):
    """Presence check for an accepted-value literal. Most tokens (`xhigh`,
    `fable`, `dontAsk`, `acceptEdits`, ...) are distinctive enough that a bare
    substring test only passes when the real literal is there. A handful of
    common English words used as value literals (`max`, `default`, `auto`,
    `manual`) are not: bare substring would also pass on unrelated prose like
    "maximum", "defaults to...", "automatically" or "manually" without the
    actual token being documented (test-critic tautology::F1). For those,
    require a word-boundary match instead so only the literal itself counts."""
    if token in _COMMON_WORD_TOKENS:
        return re.search(rf"\b{re.escape(token)}\b", desc) is not None
    return token in desc


def test_start_tools_document_accepted_values(server_params):
    """R3: tools/list names the accepted values for model/effort/permission_mode, per
    tool. `harness_start_agent`'s inputSchema.properties for model, effort and
    permission_mode each carry a description naming that parameter's value list;
    harness_start_prompt's do so for model and effort, has no permission_mode
    property at all, and its tool description says it sends no permission mode to
    the child and does not inherit the parent's. Fixed literal tokens, written
    independently of _ACCEPTED_VALUES (asserting against the constant the code
    consumed would be vacuous). Beyond bare substring presence -- which a
    description stating the opposite fact while still containing every required
    token would also satisfy -- the four claims that have a "which way does this
    go" direction (start_agent's effort/model/permission_mode passthrough and
    inheritance, start_prompt's flag-omission and non-inheritance) are additionally
    checked sentence-by-sentence for the correct polarity via
    _assert_states_positively/_assert_states_negatively."""

    async def scenario(session):
        return (await session.list_tools()).tools

    tools = _run(scenario, server_params)
    by_name = {t.name: t for t in tools}

    start_agent = by_name["harness_start_agent"]
    agent_props = start_agent.inputSchema["properties"]

    effort_desc = agent_props["effort"].get("description") or ""
    assert "low" in effort_desc and "medium" in effort_desc and "high" in effort_desc, (
        "harness_start_agent effort description must name low/medium/high"
    )
    assert "xhigh" in effort_desc and _token_present(effort_desc, "max"), (
        "harness_start_agent effort description must name xhigh/max"
    )
    assert "--effort" in effort_desc
    _assert_states_positively(
        effort_desc, r"--effort", "harness_start_agent effort description"
    )
    _assert_states_unset_behaviour(
        effort_desc,
        r"parent session",
        "harness_start_agent effort description (unset falls back to parent session)",
    )

    model_desc = agent_props["model"].get("description") or ""
    for token in ("opus", "sonnet", "haiku", "fable", "default", "claude-"):
        assert _token_present(model_desc, token), (
            f"harness_start_agent model description missing {token!r}"
        )
    _assert_states_positively(
        model_desc, r"\bmodel\b", "harness_start_agent model description"
    )
    _assert_states_unset_behaviour(
        model_desc,
        r"parent session",
        "harness_start_agent model description (unset falls back to parent session)",
    )

    perm_desc = agent_props["permission_mode"].get("description") or ""
    for token in (
        "default", "acceptEdits", "auto", "bypassPermissions", "dontAsk", "manual",
        "plan", "inherit",
    ):
        assert _token_present(perm_desc, token), (
            f"harness_start_agent permission_mode description missing {token!r}"
        )
    # Tightened per round-3 test-critic F1: bare presence of "inherit" would also
    # be satisfied by a description that merely lists it as a fifth accepted
    # value (e.g. "default, acceptEdits, plan, bypassPermissions, inherit"), which
    # is not an inheritance claim at all. Require the word to sit in a genuine
    # unset-value-fallback clause instead.
    _assert_states_unset_behaviour(
        perm_desc,
        r"parent session|inherit",
        "harness_start_agent permission_mode description (unset falls back to parent session)",
    )

    start_prompt = by_name["harness_start_prompt"]
    prompt_props = start_prompt.inputSchema["properties"]

    prompt_effort_desc = prompt_props["effort"].get("description") or ""
    assert (
        "low" in prompt_effort_desc and "medium" in prompt_effort_desc and "high" in prompt_effort_desc
    ), "harness_start_prompt effort description must name low/medium/high"
    assert "xhigh" in prompt_effort_desc and _token_present(prompt_effort_desc, "max"), (
        "harness_start_prompt effort description must name xhigh/max"
    )
    # harness_start_prompt never reads parent-session context, so its unset
    # behaviour is the opposite direction from harness_start_agent's: no
    # fallback exists, the flag is simply omitted. Require a genuine negated
    # claim about the flag itself (mirrors the flag-omission check on
    # start_prompt_desc below), not just bare token presence.
    _assert_states_negatively(
        prompt_effort_desc,
        r"--effort",
        "harness_start_prompt effort description (flag not sent when unset)",
    )

    prompt_model_desc = prompt_props["model"].get("description") or ""
    for token in ("opus", "sonnet", "haiku", "fable", "default", "claude-"):
        assert _token_present(prompt_model_desc, token), (
            f"harness_start_prompt model description missing {token!r}"
        )
    # Same asymmetry for model: harness_start_prompt has no session context to
    # fall back to, so the description must say the parameter is required
    # rather than describing a fallback target. Require a negated claim about
    # the (absent) session-context fallback, not just bare token presence.
    _assert_states_negatively(
        prompt_model_desc,
        r"session context",
        "harness_start_prompt model description (no parent session to fall back to; must be supplied)",
    )

    assert "permission_mode" not in prompt_props, (
        "harness_start_prompt must not expose a permission_mode parameter"
    )
    start_prompt_desc = start_prompt.description or ""
    assert "--permission-mode" in start_prompt_desc
    assert "not inherited" in start_prompt_desc.lower(), (
        "harness_start_prompt description must say the parent session's mode is not inherited"
    )
    # Polarity: a description that says the flag *is* forwarded / the mode *is*
    # inherited would still contain both bare tokens above. Require an actual
    # negation attached to each claim, not just the tokens.
    _assert_states_negatively(
        start_prompt_desc, r"--permission-mode", "harness_start_prompt description (flag not sent)"
    )
    _assert_states_negatively(
        start_prompt_desc, r"inherit", "harness_start_prompt description (not inherited)"
    )


def test_documented_values_match_lib_validator():
    """R3: server.py's _ACCEPTED_VALUES for `effort` and `model` are exactly what
    the pinned lib_python_harness's own hard validator (providers/claude_cli.py)
    accepts before it ever launches a child -- not this repo's own guess,
    re-derived from the lib's own constants rather than from --help text (the lib
    validates effort/model itself; permission_mode is untouched by the lib and is
    checked against the live CLI instead, in test_live_claude.py). `model`'s last
    documented element is the full-model-id example, not an alias, so it is
    excluded from the alias comparison; `inherit` is an agent-definition sentinel
    (see the lib's own comment), not a CLI-accepted alias, so it is excluded too.
    A lib bump that adds/removes an alias or effort level fails this test offline,
    without needing the live CLI. Expected RED before the change: AssertionError
    showing the missing {'xhigh', 'max'} and {'fable', 'default'}."""
    from lib_python_harness.providers.claude_cli import _EFFORT_VALUES, _MODEL_ALIASES

    from harness_plugin.server import _ACCEPTED_VALUES

    documented_effort = set(_ACCEPTED_VALUES["effort"])
    assert documented_effort == set(_EFFORT_VALUES), (
        f"_ACCEPTED_VALUES['effort'] {documented_effort} must equal the lib "
        f"validator's _EFFORT_VALUES {set(_EFFORT_VALUES)}"
    )

    documented_model_aliases = set(_ACCEPTED_VALUES["model"][:-1])
    lib_aliases_minus_inherit = set(_MODEL_ALIASES) - {"inherit"}
    assert documented_model_aliases == lib_aliases_minus_inherit, (
        f"_ACCEPTED_VALUES['model'][:-1] {documented_model_aliases} must equal "
        f"the lib validator's _MODEL_ALIASES minus 'inherit' {lib_aliases_minus_inherit}"
    )


def test_list_agents_returns_project_agent(server_params, project_dir):
    async def scenario(session):
        return await _call(session, "harness_list_agents", cwd=str(project_dir))

    is_error, text, payload = _run(scenario, server_params)
    assert not is_error, text
    by_name = {a["qualified_name"]: a for a in payload["agents"]}
    assert "demo" in by_name
    assert by_name["demo"]["source_scope"] == "project"
    assert by_name["demo"]["description"] == "Demo agent for tests"


def test_list_agents_empty_cwd_is_not_an_error(server_params, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    async def scenario(session):
        return await _call(session, "harness_list_agents", cwd=str(empty))

    is_error, text, payload = _run(scenario, server_params)
    assert not is_error, text
    assert payload["agents"] == []


def test_start_agent_completes_and_reports_result(server_params, project_dir):
    async def scenario(session):
        is_error, text, started = await _call(
            session, "harness_start_agent", agent="demo", cwd=str(project_dir), model="sonnet"
        )
        assert not is_error, text
        return started, await _poll_until_terminal(session, started["run_id"])

    started, final = _run(scenario, server_params)
    assert started["run_id"]
    assert final["state"] == "COMPLETED"
    assert final["text"] == "OK"
    assert final["session_id"]


def test_start_prompt_completes_then_cleanup_forgets_run(server_params):
    async def scenario(session):
        is_error, text, started = await _call(
            session, "harness_start_prompt", prompt="Say OK.", model="sonnet"
        )
        assert not is_error, text
        final = await _poll_until_terminal(session, started["run_id"])
        cleaned = await _call(session, "harness_cleanup_run", run_id=started["run_id"])
        after = await _call(session, "harness_poll_run", run_id=started["run_id"])
        return final, cleaned, after

    final, cleaned, after = _run(scenario, server_params)
    assert final["state"] == "COMPLETED"
    assert final["text"] == "OK"
    assert cleaned[0] is False, cleaned[1]
    assert after[0] is True
    assert "HarnessError" in after[1]


def test_wait_run_returns_completed_result(server_params, project_dir):
    async def scenario(session):
        is_error, text, started = await _call(
            session, "harness_start_agent", agent="demo", cwd=str(project_dir), model="sonnet"
        )
        assert not is_error, text
        return await _call(
            session, "harness_wait_run", run_id=started["run_id"], timeout_seconds=30
        )

    is_error, text, final = _run(scenario, server_params)
    assert not is_error, text
    assert final["state"] == "COMPLETED"
    assert final["text"] == "OK"
    assert final["session_id"]


def test_wait_run_deadline_leaves_run_running_with_liveness_and_hint(server_params, project_dir):
    async def scenario(session):
        is_error, text, started = await _call(
            session, "harness_start_prompt", prompt="TOOL:Bash SLEEP:6", model="sonnet"
        )
        assert not is_error, text
        run_id = started["run_id"]
        t0 = time.monotonic()
        waited = await _call(session, "harness_wait_run", run_id=run_id, timeout_seconds=1)
        elapsed = time.monotonic() - t0
        polled = await _call(session, "harness_poll_run", run_id=run_id)
        final = await _poll_until_terminal(session, run_id)
        after = await _call(session, "harness_list_agents", cwd=str(project_dir))
        # a second run with a different tool: last_activity must follow the emitted event
        _, _, started2 = await _call(
            session, "harness_start_prompt", prompt="TOOL:Read SLEEP:3", model="sonnet"
        )
        waited2 = await _call(
            session, "harness_wait_run", run_id=started2["run_id"], timeout_seconds=1
        )
        assert waited2[2]["last_activity"] == "tool_use:Read"
        return run_id, waited, elapsed, polled, final, after

    run_id, waited, elapsed, polled, final, after = _run(scenario, server_params)
    assert waited[0] is False, waited[1]
    result = waited[2]
    assert result["state"] == "RUNNING"
    assert elapsed < 5, "the wait must return at its limit, not when the run ends"
    assert result["last_activity"] == "tool_use:Bash"
    assert result["last_event_at"] is not None
    assert result["event_count"] >= 1
    assert result["duration_s"] > 0
    assert run_id in result["next_step"]
    assert "harness wait" in result["next_step"]
    assert polled[2]["state"] == "RUNNING", "the wait limit must not cancel the run"
    assert final["state"] == "COMPLETED"
    assert after[0] is False, after[1]


def test_stop_cancels_running_run_then_errors_on_finished_run(server_params, project_dir):
    async def scenario(session):
        is_error, text, started = await _call(
            session, "harness_start_prompt", prompt="SLEEP:30", model="sonnet"
        )
        assert not is_error, text
        run_id = started["run_id"]
        first = await _call(session, "harness_stop_run", run_id=run_id)
        second = await _call(session, "harness_stop_run", run_id=run_id)
        polled = await _call(session, "harness_poll_run", run_id=run_id)
        alive = await _call(session, "harness_list_agents", cwd=str(project_dir))
        return first, second, polled, alive

    first, second, polled, alive = _run(scenario, server_params)
    assert first[0] is False, first[1]
    assert first[2]["state"] == "CANCELLED"
    assert second[0] is True
    assert "IllegalTransitionError" in second[1]
    assert "illegal run state transition: CANCELLED -> CANCELLED" in second[1]
    assert polled[2]["state"] == "CANCELLED"
    assert alive[0] is False, alive[1]


def test_error_paths_return_structured_errors(server_params, project_dir, tmp_path):
    git_cwd = tmp_path / "gitrepo"
    (git_cwd / ".git").mkdir(parents=True)

    async def scenario(session):
        unknown_agent = await _call(
            session, "harness_start_agent", agent="nope", cwd=str(project_dir), model="sonnet"
        )
        unsafe = await _call(
            session,
            "harness_start_prompt",
            prompt="hi",
            model="sonnet",
            cwd=str(git_cwd),
        )
        unknown_run = await _call(session, "harness_poll_run", run_id="does-not-exist")
        alive = await _call(session, "harness_list_agents", cwd=str(project_dir))
        return unknown_agent, unsafe, unknown_run, alive

    unknown_agent, unsafe, unknown_run, alive = _run(scenario, server_params)
    assert unknown_agent[0] is True
    assert "nope" in unknown_agent[1]
    assert "demo" in unknown_agent[1]  # lists the known qualified_names
    assert unsafe[0] is True
    assert "UnsafeCwdError" in unsafe[1]
    assert unknown_run[0] is True
    assert "HarnessError" in unknown_run[1]
    assert alive[0] is False, alive[1]


def test_wait_run_does_not_block_other_tool_calls(server_params, project_dir):
    """A wait in flight must not wedge the server: another call has to be answered
    while the wait is still blocked (proves the thread offload)."""

    async def scenario(session):
        is_error, text, started = await _call(
            session, "harness_start_prompt", prompt="SLEEP:30", model="sonnet"
        )
        assert not is_error, text
        wait_done = anyio.Event()
        box = {}

        async def waiter():
            box["wait"] = await _call(
                session, "harness_wait_run", run_id=started["run_id"], timeout_seconds=4
            )
            wait_done.set()

        async with anyio.create_task_group() as tg:
            tg.start_soon(waiter)
            await anyio.sleep(0.5)
            listed = await _call(session, "harness_list_agents", cwd=str(project_dir))
            still_waiting = not wait_done.is_set()
        await _call(session, "harness_stop_run", run_id=started["run_id"])
        return listed, still_waiting, box["wait"]

    listed, still_waiting, waited = _run(scenario, server_params)
    assert listed[0] is False, listed[1]
    assert still_waiting, "list_agents only returned after the wait finished"
    assert waited[2]["state"] == "RUNNING", "an expired wait must not cancel the run"


# --- harness_list_runs + progress fields (#6) -------------------------------------

LIST_ROW_KEYS = {"run_id", "state", "model", "cwd", "created_at", "label"}


def test_list_runs_lists_runs_same_server_and_after_restart(server_params, project_dir):
    async def session1(session):
        e1, t1, agent = await _call(
            session,
            "harness_start_agent",
            agent="demo",
            cwd=str(project_dir),
            model="sonnet",
            label="agent-a",
        )
        assert not e1, t1
        e2, t2, prompt = await _call(
            session, "harness_start_prompt", prompt="Say OK.", model="sonnet", label="prompt-b"
        )
        assert not e2, t2
        e3, t3, sleeper = await _call(
            session, "harness_start_prompt", prompt="SLEEP:30", model="sonnet", label="sleeper-c"
        )
        assert not e3, t3
        await _poll_until_terminal(session, agent["run_id"])
        await _poll_until_terminal(session, prompt["run_id"])
        listed = await _call(session, "harness_list_runs")
        return agent["run_id"], prompt["run_id"], sleeper["run_id"], listed

    t_start = time.time()
    agent_id, prompt_id, sleeper_id, same = _run(session1, server_params)
    t_end = time.time()

    async def session2(session):
        listed = await _call(session, "harness_list_runs")
        await _call(session, "harness_stop_run", run_id=sleeper_id)
        return listed

    restarted = _run(session2, server_params)

    for is_error, text, payload in (same, restarted):
        assert not is_error, text
        rows = {r["run_id"]: r for r in payload["runs"]}
        assert {agent_id, prompt_id, sleeper_id} <= set(rows)
        for row in rows.values():
            assert set(row) == LIST_ROW_KEYS, row
        assert rows[agent_id]["label"] == "agent-a"
        assert rows[prompt_id]["label"] == "prompt-b"
        assert rows[sleeper_id]["label"] == "sleeper-c"
        assert rows[agent_id]["state"] == "COMPLETED"
        assert rows[prompt_id]["state"] == "COMPLETED"
        if sys.platform == "win32":
            # Ending the stdio session kills the server's process tree, including the
            # detached fake-claude child, so after a restart the orphan may be reconciled
            # to FAILED (re-verified on lib-python-harness v0.0.9: still fails 5/5 without
            # this allowance; same limitation as the skipped wait-run cancel test).
            assert rows[sleeper_id]["state"] in {"RUNNING", "FAILED"}
        else:
            assert rows[sleeper_id]["state"] == "RUNNING"
        for row in rows.values():
            assert row["model"] == "sonnet"
            assert isinstance(row["cwd"], str) and row["cwd"]
            assert isinstance(row["created_at"], float)
            assert t_start - 1 <= row["created_at"] <= t_end + 1, row["created_at"]
        assert Path(rows[agent_id]["cwd"]).resolve() == Path(project_dir).resolve()


def test_list_runs_drops_cleaned_up_run(server_params):
    async def scenario(session):
        _, _, started = await _call(session, "harness_start_prompt", prompt="Say OK.", model="sonnet")
        await _poll_until_terminal(session, started["run_id"])
        await _call(session, "harness_cleanup_run", run_id=started["run_id"])
        return started["run_id"], await _call(session, "harness_list_runs")

    run_id, (is_error, text, payload) = _run(scenario, server_params)
    assert not is_error, text
    assert run_id not in {r["run_id"] for r in payload["runs"]}


def test_poll_reports_growing_progress_while_running(server_params):
    async def scenario(session):
        is_error, text, started = await _call(
            session, "harness_start_prompt", prompt="TICK:8:0.4", model="sonnet"
        )
        assert not is_error, text
        run_id = started["run_id"]
        await anyio.sleep(0.6)
        first = await _call(session, "harness_poll_run", run_id=run_id)
        await anyio.sleep(1.0)
        second = await _call(session, "harness_poll_run", run_id=run_id)
        await _call(session, "harness_stop_run", run_id=run_id)
        return first, second

    first, second = _run(scenario, server_params)
    assert first[0] is False, first[1]
    assert second[0] is False, second[1]
    a, b = first[2], second[2]
    assert a["state"] == "RUNNING" and b["state"] == "RUNNING"
    assert b["event_count"] > a["event_count"]
    assert isinstance(a["last_event_at"], float) and isinstance(b["last_event_at"], float)
    assert b["last_event_at"] > a["last_event_at"]


# --- parent-session HostContext wiring (#1) ---------------------------------------


def _argv_records(log):
    assert log.exists(), "fake claude was never invoked (no argv log)"
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]


def _flag(argv, name):
    assert name in argv, f"{name} missing from CLI argv: {argv}"
    return argv[argv.index(name) + 1]


# Flags whose value is expected to vary run-to-run for reasons that have nothing to do
# with what the plugin's own launch decided: `--session-id` (a fresh id every run) and
# `--add-dir` (the materialized carrier's own `<run_dir>/agents` path, which embeds
# this run's run_id -- see `claude_cli.py`'s `_build_inherit_plan`). Masking exactly
# these two, and nothing else, is what makes three identical dispatches produce
# byte-identical normalized argvs (#27 R1) -- any other divergence is a real one.
_ARGV_RUN_VARYING_FLAGS = ("--session-id", "--add-dir")


def _normalized_argv(argv):
    """`argv` with the value token following each of `_ARGV_RUN_VARYING_FLAGS`
    replaced by a fixed placeholder, so argv from separate runs of the same dispatch
    can be compared for equality regardless of their own run-identity."""
    out = list(argv)
    for flag in _ARGV_RUN_VARYING_FLAGS:
        if flag in out:
            out[out.index(flag) + 1] = f"<{flag.lstrip('-')}>"
    return out


def _start_and_finish(params, **arguments):
    async def scenario(session):
        is_error, text, started = await _call(session, "harness_start_agent", **arguments)
        assert not is_error, text
        return started, await _poll_until_terminal(session, started["run_id"])

    return _run(scenario, params)


def _poll_from_fresh_process(params, run_id):
    """Poll `run_id` from a brand-new `python -m harness_plugin` subprocess -- not
    the one that started the run. Each `_run()` call launches its own server
    process (see `_session`/`stdio_client` above), so this proves a value survives
    to a genuinely separate invocation rather than merely a later call within the
    same in-memory server process -- the distinction the plan's provenance design
    relies on ("possibly from another process (`harness wait`)"). A value kept only
    in a module-level dict keyed by run_id in the server process would be empty
    here and fail; the on-disk run record the plan actually specifies survives."""

    async def scenario(session):
        polled = await _call(session, "harness_poll_run", run_id=run_id)
        assert not polled[0], polled[1]
        return polled[2]

    return _run(scenario, params)


# --- launch determinism across repeated identical dispatches (#27 R1) -------------


@pytest.mark.parametrize("carrier", ["demo", "mcp-agent"])
def test_start_agent_repeated_launch_argv_is_identical(
    server_params, project_dir, mcp_agent_project, argv_log, carrier
):
    """R1 (#27): identical `harness_start_agent` calls must yield identical argv/cwd,
    modulo this run's own run_id/session_id -- for both dispatch carriers, the
    `--agents`-JSON payload one (`demo`) and the materialized one (`mcp-agent`, which
    also always emits `--mcp-config`). This is a pinning test: nothing in this repo's
    own launch varies between runs (see plan #27 "premises verified" -- the CLI builds
    the run-varying MCP-server announcement itself, downstream and outside this
    plugin's control), so there is no red case to demonstrate here; the driving
    assertion is that the CLI actually received a distinct `--session-id` each time
    while everything else this plugin decided stayed byte-for-byte the same."""
    cwd = project_dir if carrier == "demo" else mcp_agent_project

    for _ in range(3):
        started, final = _start_and_finish(
            server_params, agent=carrier, cwd=str(cwd), model="sonnet"
        )
        assert final["state"] == "COMPLETED", final

    records = _argv_records(argv_log)
    assert len(records) == 3

    # Distinctness must be checked on the raw `--session-id` value the CLI actually
    # received, not on the run record's reported `session_id` -- the plugin could echo
    # a fresh id in the payload while still sending the CLI a constant flag value, and
    # `_normalized_argv` below deliberately masks this flag out of the equality check,
    # so nothing else in this test reads it.
    raw_session_ids = [_flag(record["argv"], "--session-id") for record in records]
    assert len(set(raw_session_ids)) == 3, (
        f"each run must send the CLI its own distinct --session-id: {raw_session_ids}"
    )

    normalized = [_normalized_argv(r["argv"]) for r in records]
    assert normalized[0] == normalized[1] == normalized[2], normalized
    cwds = [r["cwd"] for r in records]
    assert cwds[0] == cwds[1] == cwds[2], cwds

    for record in records:
        assert _flag(record["argv"], "--setting-sources") == "user,project,local"
        # Pinning the *current* launch shape, not a hard requirement: the Frame only
        # asks that the MCP-server/setting-source flags be identical across runs. A
        # future change could legitimately add explicit/strict MCP config to make the
        # child launch deterministic in some other way; if that happens, update this
        # assertion rather than treating its failure as a regression.
        assert "--strict-mcp-config" not in record["argv"]

    if carrier == "mcp-agent":
        for record in records:
            mcp_config = json.loads(_flag(record["argv"], "--mcp-config"))
            assert mcp_config == {"mcpServers": {"demo-server": {"command": "demo"}}}


def test_start_agent_inherits_session_context(server_params, session_context, argv_log):
    started, final = _start_and_finish(server_params, agent="demo")
    assert final["state"] == "COMPLETED"
    assert started["context_source"] == "session"
    (record,) = _argv_records(argv_log)
    assert _flag(record["argv"], "--permission-mode") == "acceptEdits"
    assert _flag(record["argv"], "--effort") == "high"
    assert _flag(record["argv"], "--model") == "opus"
    assert os.path.samefile(record["cwd"], session_context["cwd"])
    assert os.path.samefile(started["cwd"], session_context["cwd"])


def test_start_agent_explicit_arguments_override_session_context(
    server_params, session_context, argv_log
):
    _start_and_finish(
        server_params, agent="demo", permission_mode="plan", model="sonnet", effort="low"
    )
    (record,) = _argv_records(argv_log)
    assert _flag(record["argv"], "--permission-mode") == "plan"
    assert _flag(record["argv"], "--effort") == "low"
    assert _flag(record["argv"], "--model") == "sonnet"


def test_start_agent_falls_back_to_newest_context_of_project_dir(
    server_params, session_context, argv_log
):
    env = dict(server_params.env)
    del env["CLAUDE_CODE_SESSION_ID"]
    env["CLAUDE_PROJECT_DIR"] = session_context["project_dir"]
    params = server_params.model_copy(update={"env": env})

    started, final = _start_and_finish(params, agent="demo")
    assert final["state"] == "COMPLETED"
    assert started["context_source"] == "fallback"
    (record,) = _argv_records(argv_log)
    assert _flag(record["argv"], "--permission-mode") == "acceptEdits"


@pytest.mark.parametrize("explicit_mode", [None, "acceptEdits"], ids=["no-args", "explicit-mode"])
def test_start_agent_refuses_without_session_context(
    server_params_no_context, project_dir, argv_log, explicit_mode
):
    args = {"agent": "demo", "cwd": str(project_dir), "model": "sonnet"}
    if explicit_mode:
        args["permission_mode"] = explicit_mode

    async def scenario(session):
        return await _call(session, "harness_start_agent", **args)

    is_error, text, _ = _run(scenario, server_params_no_context)
    assert is_error, "start_agent must refuse when no session context can be resolved"
    assert "CLAUDE_CODE_SESSION_ID" in text
    assert not argv_log.exists(), "a child was started on unconfirmed rights"


def test_start_agent_refuses_context_without_permission_mode(
    server_params, tmp_path, project_dir, argv_log
):
    # A SessionStart-only snapshot resolves, but carries no permission_mode.
    plant_session_file(tmp_path / "plugin-data", SESSION_ID, cwd=str(project_dir))

    async def scenario(session):
        return await _call(
            session, "harness_start_agent", agent="demo", cwd=str(project_dir), model="sonnet"
        )

    is_error, text, _ = _run(scenario, server_params)
    assert is_error, "a resolved context lacking permission_mode must not start a child"
    assert "permission_mode" in text
    assert not argv_log.exists()


def test_start_agent_explicit_permission_mode_overrides_file_lacking_one(
    server_params, tmp_path, project_dir, argv_log
):
    plant_session_file(tmp_path / "plugin-data", SESSION_ID, cwd=str(project_dir))
    started, final = _start_and_finish(
        server_params, agent="demo", cwd=str(project_dir), model="sonnet", permission_mode="plan"
    )
    assert final["state"] == "COMPLETED"
    (record,) = _argv_records(argv_log)
    assert _flag(record["argv"], "--permission-mode") == "plan"


# --- effort echo + provenance (#25) -----------------------------------------------


def test_effort_is_echoed_at_start_and_in_poll_and_wait(server_params, session_context, argv_log):
    """R1: harness_start_agent, harness_start_prompt and harness_send_message
    responses, and harness_poll_run/harness_wait_run for those same run_ids, carry
    `effort` equal to the `--effort` token in that run's argv-log record. Also checks
    the model/permission_mode consolidation (both must still equal their argv tokens
    now that they come from launched_fields() instead of their own extras)."""

    async def scenario(session):
        is_error, text, started = await _call(session, "harness_start_agent", agent="demo")
        assert not is_error, text
        final = await _poll_until_terminal(session, started["run_id"])
        waited = await _call(
            session, "harness_wait_run", run_id=started["run_id"], timeout_seconds=30
        )

        is_error2, text2, started_prompt = await _call(
            session, "harness_start_prompt", prompt="Say OK.", model="sonnet", effort="low"
        )
        assert not is_error2, text2
        final_prompt = await _poll_until_terminal(session, started_prompt["run_id"])

        sent = await _call(
            session, "harness_send_message", run_id=started_prompt["run_id"], prompt="hi"
        )
        assert not sent[0], sent[1]
        follow_up = sent[2]
        final_follow_up = await _poll_until_terminal(session, follow_up["run_id"])

        return started, final, waited, started_prompt, final_prompt, follow_up, final_follow_up

    (
        started,
        final,
        waited,
        started_prompt,
        final_prompt,
        follow_up,
        final_follow_up,
    ) = _run(scenario, server_params)

    assert waited[0] is False, waited[1]
    waited_payload = waited[2]

    records = _argv_records(argv_log)
    agent_effort = _flag(records[0]["argv"], "--effort")
    assert agent_effort == "high"  # inherited from session_context
    assert started["effort"] == agent_effort
    assert final["effort"] == agent_effort
    assert waited_payload["effort"] == agent_effort
    # consolidation: model/permission_mode must still equal their argv tokens
    assert started["model"] == _flag(records[0]["argv"], "--model")
    assert started["permission_mode"] == _flag(records[0]["argv"], "--permission-mode")

    prompt_effort = _flag(records[1]["argv"], "--effort")
    assert prompt_effort == "low"
    assert started_prompt["effort"] == "low"
    assert final_prompt["effort"] == "low"

    follow_up_effort = _flag(records[2]["argv"], "--effort")
    assert follow_up_effort == "low"
    assert follow_up["effort"] == "low"
    assert final_follow_up["effort"] == "low"


def test_effort_is_null_and_absent_from_argv_when_nothing_supplies_it(
    server_params, tmp_path, project_dir, argv_log
):
    """R1 additional edge-case coverage: a session context planted without `effort`
    and no explicit argument -> "--effort" not in argv and payload["effort"] is
    None."""
    plant_session_file(
        tmp_path / "plugin-data",
        SESSION_ID,
        cwd=str(project_dir),
        permission_mode="acceptEdits",
        model="opus",
    )

    async def scenario(session):
        is_error, text, started = await _call(session, "harness_start_agent", agent="demo")
        assert not is_error, text
        return started, await _poll_until_terminal(session, started["run_id"])

    started, final = _run(scenario, server_params)
    (record,) = _argv_records(argv_log)
    assert "--effort" not in record["argv"]
    assert started["effort"] is None
    assert final["effort"] is None


@pytest.mark.parametrize(
    "kwargs,expect_source,expect_effort",
    [
        pytest.param(
            {"agent": "effort-agent", "effort": "low"},
            "agent_definition",
            "medium",
            id="definition-wins-over-argument-and-session",
        ),
        pytest.param(
            {"agent": "demo", "effort": "low"},
            "argument",
            "low",
            id="explicit-argument",
        ),
        pytest.param(
            {"agent": "demo"},
            "parent_session",
            "high",
            id="inherited-from-session",
        ),
    ],
)
def test_effort_source_reports_where_the_value_came_from_for_start_agent(
    server_params, session_context, project_dir, argv_log, kwargs, expect_source, expect_effort
):
    """R2 (a)-(c): effort_source is agent_definition when the definition sets
    effort:, argument for an explicit argument with no definition value,
    parent_session for the inherited snapshot value -- and it persists into the
    later poll answer, re-read by a genuinely separate server process (not the one
    that started the run) so a value kept only in that process's memory could not
    satisfy this. `effort-agent`'s own frontmatter value must win over both the
    explicit argument and the session's `effort: high` (pre-existing library
    behaviour: `effort = definition.effort or host_context.effort`, with no
    argument re-application for effort, unlike model)."""

    async def start_scenario(session):
        is_error, text, started = await _call(session, "harness_start_agent", **kwargs)
        assert not is_error, text
        return started

    started = _run(start_scenario, server_params)
    polled = _poll_from_fresh_process(server_params, started["run_id"])
    (record,) = _argv_records(argv_log)
    assert _flag(record["argv"], "--effort") == expect_effort
    assert started["effort"] == expect_effort
    assert started["effort_source"] == expect_source
    assert polled["effort"] == expect_effort
    assert polled["effort_source"] == expect_source


def test_effort_source_none_when_session_lacks_effort(
    server_params, tmp_path, project_dir, argv_log
):
    """R2 (d): a session file without `effort` and no explicit argument ->
    effort_source == "none", no --effort token. The later poll is re-read by a
    fresh server process (not the one that started the run) -- see
    `_poll_from_fresh_process`."""
    plant_session_file(
        tmp_path / "plugin-data",
        SESSION_ID,
        cwd=str(project_dir),
        permission_mode="acceptEdits",
        model="opus",
    )

    async def start_scenario(session):
        is_error, text, started = await _call(session, "harness_start_agent", agent="demo")
        assert not is_error, text
        return started

    started = _run(start_scenario, server_params)
    polled = _poll_from_fresh_process(server_params, started["run_id"])
    (record,) = _argv_records(argv_log)
    assert "--effort" not in record["argv"]
    assert started["effort"] is None
    assert started["effort_source"] == "none"
    assert polled["effort_source"] == "none"


def test_effort_source_argument_for_start_prompt(server_params, argv_log):
    """R2 (f): harness_start_prompt(effort="low") -> effort_source == "argument".
    The later poll is re-read by a fresh server process (not the one that started
    the run) -- see `_poll_from_fresh_process`."""

    async def start_scenario(session):
        is_error, text, started = await _call(
            session, "harness_start_prompt", prompt="Say OK.", model="sonnet", effort="low"
        )
        assert not is_error, text
        return started

    started = _run(start_scenario, server_params)
    polled = _poll_from_fresh_process(server_params, started["run_id"])
    (record,) = _argv_records(argv_log)
    assert _flag(record["argv"], "--effort") == "low"
    assert started["effort_source"] == "argument"
    assert polled["effort_source"] == "argument"


def test_effort_source_none_for_start_prompt_even_with_session_effort(
    server_params, session_context, argv_log
):
    """R2 (e): harness_start_prompt with no `effort` while the session carries
    `effort: high` -> effort_source == "none" and no --effort token -- "none" must
    mean "CLI default", not a dropped inheritance, since harness_start_prompt never
    reads the session context at all. The later poll is re-read by a fresh server
    process (not the one that started the run) -- see `_poll_from_fresh_process`."""

    async def start_scenario(session):
        is_error, text, started = await _call(
            session, "harness_start_prompt", prompt="Say OK.", model="sonnet"
        )
        assert not is_error, text
        return started

    started = _run(start_scenario, server_params)
    polled = _poll_from_fresh_process(server_params, started["run_id"])
    (record,) = _argv_records(argv_log)
    assert "--effort" not in record["argv"]
    assert started["effort"] is None
    assert started["effort_source"] == "none"
    assert polled["effort_source"] == "none"


def test_send_message_reports_origin_runs_effort_source(server_params, argv_log):
    """R2 additional edge-case coverage: harness_send_message on a finished run
    reports the origin run's effort_source -- for two origins with *different*
    recorded sources, so the resume path is forced to actually look up and echo
    each origin's real recorded source rather than a constant: a
    harness_start_prompt(effort="low") origin (source "argument") and a
    harness_start_agent("demo") origin with no explicit effort, inherited from
    session_context's effort: high (source "parent_session"). An implementation
    that always answers "argument" for resumed runs -- for unrelated or wrong
    reasons -- passes the first pair here but fails the second. Both origins are
    started and finished in one server process; harness_send_message is then
    issued from a second, fresh process (not the one that started/finished the
    origin runs), so an effort_source kept only in the first process's memory
    could not answer it."""

    async def start_scenario(session):
        is_error, text, argument_origin = await _call(
            session, "harness_start_prompt", prompt="Say OK.", model="sonnet", effort="low"
        )
        assert not is_error, text
        await _poll_until_terminal(session, argument_origin["run_id"])

        is_error2, text2, session_origin = await _call(
            session, "harness_start_agent", agent="demo"
        )
        assert not is_error2, text2
        await _poll_until_terminal(session, session_origin["run_id"])

        return argument_origin, session_origin

    argument_origin, session_origin = _run(start_scenario, server_params)
    assert argument_origin["effort_source"] == "argument"
    assert session_origin["effort_source"] == "parent_session"
    assert session_origin["effort"] == "high"  # inherited from session_context

    async def send_scenario(session):
        sent_argument = await _call(
            session, "harness_send_message", run_id=argument_origin["run_id"], prompt="hi"
        )
        assert not sent_argument[0], sent_argument[1]
        sent_session = await _call(
            session, "harness_send_message", run_id=session_origin["run_id"], prompt="hi"
        )
        assert not sent_session[0], sent_session[1]
        return sent_argument[2], sent_session[2]

    follow_up_argument, follow_up_session = _run(send_scenario, server_params)
    assert follow_up_argument["effort_source"] == "argument"
    assert follow_up_argument["effort"] == "low"
    assert follow_up_session["effort_source"] == "parent_session"
    assert follow_up_session["effort"] == "high"


def test_list_agents_and_start_prompt_work_without_context(server_params_no_context, project_dir):
    async def scenario(session):
        listed = await _call(session, "harness_list_agents", cwd=str(project_dir))
        prompt = await _call(session, "harness_start_prompt", prompt="Say OK.", model="sonnet")
        return listed, prompt

    listed, prompt = _run(scenario, server_params_no_context)
    assert listed[0] is False, listed[1]
    assert prompt[0] is False, prompt[1]


def test_probe_warning_goes_to_stderr_not_stdout(server_params_no_context, tmp_path):
    errlog_path = tmp_path / "server-stderr.log"

    async def main():
        with anyio.fail_after(60):
            with open(errlog_path, "w", encoding="utf-8") as errlog:
                async with stdio_client(server_params_no_context, errlog=errlog) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()  # stdout stayed valid JSON-RPC

    anyio.run(main)
    assert "CLAUDE_CODE_SESSION_ID" in errlog_path.read_text(encoding="utf-8")


def test_list_agents_uses_session_cwd_and_respects_disabled_plugins(
    server_params, session_context, project_dir, plugin_agent_install, tmp_path
):
    # A second, disabled plugin install -- `plugin_agent_install` only ever provides
    # the one enabled `agent-harness:probe`, so the disabled-plugin exclusion below
    # needs its own install.
    config = tmp_path / "claude-config"
    beta_agents = tmp_path / "plugins" / "beta" / "agents"
    beta_agents.mkdir(parents=True)
    (beta_agents / "helper.md").write_text(
        "---\nname: helper\ndescription: beta helper\n---\nDo it.\n", encoding="utf-8"
    )
    installed_path = config / "plugins" / "installed_plugins.json"
    installed = json.loads(installed_path.read_text())
    installed["plugins"]["beta@mk"] = [
        {"scope": "user", "installPath": str(tmp_path / "plugins" / "beta")}
    ]
    installed_path.write_text(json.dumps(installed), encoding="utf-8")
    settings_path = config / "settings.json"
    settings = json.loads(settings_path.read_text())
    settings["enabledPlugins"]["beta@mk"] = True
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    (project_dir / ".claude" / "settings.local.json").write_text(
        json.dumps({"enabledPlugins": {"beta@mk": False}}), encoding="utf-8"
    )

    (tmp_path / "elsewhere").mkdir()

    async def scenario(session):
        implicit = await _call(session, "harness_list_agents")
        explicit = await _call(session, "harness_list_agents", cwd=str(tmp_path / "elsewhere"))
        return implicit, explicit

    implicit, explicit = _run(scenario, server_params)
    assert implicit[0] is False, implicit[1]
    by_name = {a["qualified_name"]: a for a in implicit[2]["agents"]}
    assert os.path.samefile(implicit[2]["cwd"], session_context["cwd"])
    assert by_name["demo"]["source_scope"] == "project"
    assert by_name["agent-harness:probe"]["source_scope"] == "plugin"
    assert "beta:helper" not in by_name
    assert explicit[0] is False, explicit[1]
    assert os.path.samefile(explicit[2]["cwd"], tmp_path / "elsewhere")


def test_plugin_agent_lists_and_starts_colon_qualified_with_tools(
    server_params, plugin_agent_install, project_dir, argv_log
):
    """R1 (list half) + R2 (start half): a plugin `agents/` definition is discovered
    as `<plugin>:<name>`, and dispatching it binds `tools:` as a real `--tools`
    allowlist -- not just carried inside the `--agents` JSON payload, which is all
    v0.0.4 did. The `--tools` assertion is the discriminator: it fails (absent from
    argv) on v0.0.4 and passes on the pinned v0.0.5."""

    async def scenario(session):
        listed = await _call(session, "harness_list_agents", cwd=str(project_dir))
        is_error, text, started = await _call(
            session,
            "harness_start_agent",
            agent="agent-harness:probe",
            cwd=str(project_dir),
            model="sonnet",
        )
        assert not is_error, text
        final = await _poll_until_terminal(session, started["run_id"])
        return listed, started, final

    listed, started, final = _run(scenario, server_params)

    # R1 -- list half.
    is_error, text, payload = listed
    assert not is_error, text
    by_name = {a["qualified_name"]: a for a in payload["agents"]}
    assert "agent-harness:probe" in by_name
    entry = by_name["agent-harness:probe"]
    assert entry["source_scope"] == "plugin"
    assert Path(entry["path"]) == plugin_agent_install / "agents" / "probe.md"

    # R2 -- start half.
    assert final["state"] == "COMPLETED"
    assert final["text"] == "OK"
    (record,) = _argv_records(argv_log)
    assert _flag(record["argv"], "--agent") == "agent-harness:probe"
    assert _flag(record["argv"], "--tools") == "Read,Glob"
    assert _agents_payload(record)["agent-harness:probe"]["tools"] == ["Read", "Glob"]


@pytest.mark.parametrize(
    "frontmatter,expect_tools",
    [
        pytest.param(
            "disallowedTools: Edit, Write",
            None,
            id="deny-only",
        ),
        pytest.param(
            "disallowedTools: Edit, Write\ntools: Read, Edit",
            ["Read", "Edit"],
            id="combined",
        ),
    ],
)
def test_plugin_agent_binds_disallowed_tools_and_inspect_reads_it_back(
    server_params, plugin_agent_install, project_dir, argv_log, frontmatter, expect_tools
):
    """#50/#56: a plugin agent's `disallowedTools:` frontmatter reaches the launched
    child via lib-python-harness v0.0.9's top-level `--disallowedTools` CSV flag
    (emitted before and independently of the payload/materialized carrier split) --
    the `--agents` JSON payload's per-agent `disallowedTools` key still carries it
    too, since the library sends it there as well -- and `harness_inspect_run` reads
    `requested.disallowed_tools` back from the flag, mirroring `tools`. The
    `combined` case additionally proves the allowlist (`tools:`) and the denylist
    (`disallowedTools:`) stay two distinct flags/payload keys, unmerged, and land as
    two distinct `requested` lists."""
    (plugin_agent_install / "agents" / "guarded.md").write_text(
        f"---\nname: guarded\ndescription: Guarded agent for disallowedTools tests\n"
        f"{frontmatter}\n---\nSay OK.\n",
        encoding="utf-8",
    )

    async def scenario(session):
        is_error, text, started = await _call(
            session,
            "harness_start_agent",
            agent="agent-harness:guarded",
            cwd=str(project_dir),
            model="sonnet",
        )
        assert not is_error, text
        final = await _poll_until_terminal(session, started["run_id"])
        inspected = await _call(session, "harness_inspect_run", run_id=started["run_id"])
        return final, inspected

    final, (is_error, text, inspected) = _run(scenario, server_params)
    assert not is_error, text
    assert final["state"] == "COMPLETED"

    (record,) = _argv_records(argv_log)
    assert _flag(record["argv"], "--agent") == "agent-harness:guarded"
    assert _flag(record["argv"], "--disallowedTools") == "Edit,Write"
    payload_entry = _agents_payload(record)["agent-harness:guarded"]
    assert payload_entry["disallowedTools"] == ["Edit", "Write"]
    requested = inspected["requested"]
    assert requested["disallowed_tools"] == ["Edit", "Write"]

    if expect_tools is None:
        assert "--tools" not in record["argv"]
        assert "tools" not in payload_entry, (
            "deny-only frontmatter sets no tools: -- the payload must not synthesize "
            "an allowlist key that was never declared"
        )
    else:
        assert _flag(record["argv"], "--tools") == ",".join(expect_tools)
        assert payload_entry["tools"] == expect_tools
        assert requested["tools"] == expect_tools


def test_materialized_agent_disallowed_tools_read_back_from_flag(
    server_params, mcp_agent_project, argv_log
):
    """#56: an agent dispatched via the materialized `.claude/agents/<stem>.md`
    carrier (its `mcpServers:` frontmatter forces that carrier -- see
    `mcp_agent_project`, no `--agents` JSON payload is ever emitted for it) still
    carries a `disallowedTools:` denylist through lib-python-harness v0.0.9's
    top-level `--disallowedTools` CSV flag, which the library emits before and
    independently of the payload/materialized carrier split. `harness_inspect_run`
    must read `requested.disallowed_tools` back from that flag -- the only carrier
    available here, since this dispatch never sends an `--agents` payload."""
    agents_dir = mcp_agent_project / ".claude" / "agents"
    (agents_dir / "guarded-mcp.md").write_text(
        "---\n"
        "name: guarded-mcp\n"
        "description: Materialized-carrier agent with a denylist\n"
        "mcpServers:\n"
        "  demo-server:\n"
        "    command: demo\n"
        "disallowedTools: Edit, Write\n"
        "---\n"
        "Say OK.\n",
        encoding="utf-8",
    )

    async def scenario(session):
        is_error, text, started = await _call(
            session,
            "harness_start_agent",
            agent="guarded-mcp",
            cwd=str(mcp_agent_project),
            model="sonnet",
        )
        assert not is_error, text
        final = await _poll_until_terminal(session, started["run_id"])
        inspected = await _call(session, "harness_inspect_run", run_id=started["run_id"])
        return final, inspected

    final, (is_error, text, inspected) = _run(scenario, server_params)
    assert not is_error, text
    assert final["state"] == "COMPLETED"

    (record,) = _argv_records(argv_log)
    assert "--agents" not in record["argv"], (
        "this carrier must never emit --agents -- otherwise this test would not "
        "exercise the materialized-carrier path #56 is about"
    )
    assert _flag(record["argv"], "--disallowedTools") == "Edit,Write"
    requested = inspected["requested"]
    assert requested["disallowed_tools"] == ["Edit", "Write"]


def test_project_agent_overrides_plugin_agent_by_qualified_name(
    server_params, plugin_agent_install, project_dir
):
    """README override claim: a project-scope definition using the shipped plugin
    agent's own qualified name (`name: agent-harness:probe`) wins over the plugin's,
    per discovery's project > user > plugin precedence (first writer wins)."""
    (project_dir / ".claude" / "agents" / "over.md").write_text(
        "---\nname: agent-harness:probe\ndescription: Project override of the plugin probe\n---\n"
        "Say OK.\n",
        encoding="utf-8",
    )

    async def scenario(session):
        return await _call(session, "harness_list_agents", cwd=str(project_dir))

    is_error, text, payload = _run(scenario, server_params)
    assert not is_error, text
    by_name = {a["qualified_name"]: a for a in payload["agents"]}
    assert by_name["agent-harness:probe"]["source_scope"] == "project"


# --- harness_send_message (#5) ------------------------------------------------------


def test_send_message_resumes_finished_run(server_params, argv_log):
    async def scenario(session):
        is_error, text, started = await _call(
            session, "harness_start_prompt", prompt="Say OK.", model="sonnet"
        )
        assert not is_error, text
        origin = await _poll_until_terminal(session, started["run_id"])
        sent = await _call(session, "harness_send_message", run_id=origin["run_id"], prompt="ECHO:BANANA")
        assert not sent[0], sent[1]
        waited = await _call(session, "harness_wait_run", run_id=sent[2]["run_id"], timeout_seconds=30)
        return origin, sent[2], waited

    origin, sent, waited = _run(scenario, server_params)
    assert origin["state"] == "COMPLETED"
    assert sent["run_id"] != origin["run_id"]
    assert sent["session_id"] == origin["session_id"]
    assert sent["state"] == "RUNNING"
    assert sent["resumed_from"] == origin["run_id"]
    assert waited[0] is False, waited[1]
    assert waited[2]["state"] == "COMPLETED"
    assert waited[2]["text"] == "BANANA"
    records = _argv_records(argv_log)
    assert len(records) == 2
    assert _flag(records[1]["argv"], "--resume") == origin["session_id"]


def test_send_message_error_paths(server_params, argv_log):
    async def scenario(session):
        unknown = await _call(session, "harness_send_message", run_id="does-not-exist", prompt="hi")
        _, _, sleeper = await _call(session, "harness_start_prompt", prompt="SLEEP:30", model="sonnet")
        running = await _call(session, "harness_send_message", run_id=sleeper["run_id"], prompt="hi")
        await _call(session, "harness_stop_run", run_id=sleeper["run_id"])
        _, _, quick = await _call(session, "harness_start_prompt", prompt="Say OK.", model="sonnet")
        await _poll_until_terminal(session, quick["run_id"])
        before = len(_argv_records(argv_log))
        empty = await _call(session, "harness_send_message", run_id=quick["run_id"], prompt="")
        after = len(_argv_records(argv_log))
        await _call(session, "harness_cleanup_run", run_id=quick["run_id"])
        cleaned = await _call(session, "harness_send_message", run_id=quick["run_id"], prompt="hi")
        alive = await _call(session, "harness_list_runs")
        return unknown, running, empty, before, after, cleaned, alive

    unknown, running, empty, before, after, cleaned, alive = _run(scenario, server_params)
    assert unknown[0] is True
    assert "HarnessError" in unknown[1]
    assert "does-not-exist" in unknown[1]
    assert running[0] is True
    assert "HarnessError" in running[1]
    assert "RUNNING" in running[1]
    assert empty[0] is True
    assert "HarnessError" in empty[1]
    assert "empty" in empty[1]
    assert after == before, "an empty prompt must not spawn a child"
    assert cleaned[0] is True
    assert "HarnessError" in cleaned[1]
    assert alive[0] is False, alive[1]


def _agents_payload(record):
    return json.loads(_flag(record["argv"], "--agents"))


def test_start_agent_prompt_becomes_user_message_and_body_stays_agent_prompt(
    server_params, project_dir, argv_log
):
    started, final = _start_and_finish(
        server_params, agent="demo", cwd=str(project_dir), model="sonnet", prompt="ECHO:CBA"
    )
    assert final["state"] == "COMPLETED"
    # The fake CLI answers from stdin: the definition body ("Say OK.") has no ECHO marker,
    # so "CBA" proves the prompt reached the run as its user message.
    assert final["text"] == "CBA"
    (record,) = _argv_records(argv_log)
    agent_prompt = _agents_payload(record)["demo"]["prompt"]
    assert "Say OK." in agent_prompt, "the definition body must stay the agent's system prompt"
    assert "ECHO" not in agent_prompt, "the task must not leak into the agent's system prompt"
    assert _flag(record["argv"], "--agent") == "demo"


def test_start_agent_without_prompt_keeps_body_as_user_message(
    server_params, project_dir, argv_log
):
    _, final = _start_and_finish(
        server_params, agent="demo", cwd=str(project_dir), model="sonnet"
    )
    assert final["text"] == "OK"
    (record,) = _argv_records(argv_log)
    assert "Say OK." in _agents_payload(record)["demo"]["prompt"]


@pytest.mark.parametrize("blank", ["", "   \n\t"], ids=["empty", "whitespace"])
def test_start_agent_refuses_blank_prompt_without_spawning(
    server_params, project_dir, argv_log, blank
):
    async def scenario(session):
        return await _call(
            session, "harness_start_agent", agent="demo", cwd=str(project_dir),
            model="sonnet", prompt=blank,
        )

    is_error, text, _ = _run(scenario, server_params)
    assert is_error
    assert "empty" in text.lower()
    assert not argv_log.exists() or not _argv_records(argv_log)


# --- harness_inspect_run (#28) ------------------------------------------------------


async def _poll_until_activity(session, run_id, activity, budget=20.0):
    deadline = time.monotonic() + budget
    while True:
        is_error, text, payload = await _call(session, "harness_poll_run", run_id=run_id)
        assert not is_error, text
        if payload.get("last_activity") == activity:
            return payload
        assert time.monotonic() < deadline, f"never reached last_activity={activity!r}: {payload}"
        await anyio.sleep(0.2)


def test_inspect_run_returns_announced_names_from_init_event(server_params):
    async def scenario(session):
        is_error, text, started = await _call(
            session, "harness_start_prompt", prompt="Say OK.", model="sonnet"
        )
        assert not is_error, text
        final = await _poll_until_terminal(session, started["run_id"])
        inspected = await _call(session, "harness_inspect_run", run_id=started["run_id"])
        return final, inspected

    final, (is_error, text, payload) = _run(scenario, server_params)
    assert not is_error, text
    announced = payload["announced"]
    for key, names in INIT_ANNOUNCEMENTS.items():
        assert announced[key] == names, key
    assert payload["announced_counts"] == {
        key: len(names) for key, names in INIT_ANNOUNCEMENTS.items()
    }
    # Verbatim passthrough: the *whole* init event, not just its session_id, must match
    # what the fake CLI actually emitted -- a filtered/synthesised copy carrying only the
    # recognised keys plus session_id would fail this.
    assert payload["init_event"] == {
        "type": "system",
        "subtype": "init",
        "session_id": final["session_id"],
        **INIT_ANNOUNCEMENTS,
    }
    # A terminal run's `state` must reflect the real run state, not a literal.
    assert payload["state"] == final["state"] == "COMPLETED"


def test_inspect_run_announced_names_handle_alias_key_and_dict_items(server_params):
    """`_INIT_NAME_FIELDS`'s `mcpServers` alias and `_names()`'s dict-item branch
    (`{"name": ...}`) are never reached by the plain-string/primary-key fixture above --
    this exercises both explicitly."""

    async def scenario(session):
        is_error, text, started = await _call(
            session, "harness_start_prompt", prompt=f"{ALT_INIT_SHAPE} Say OK.", model="sonnet"
        )
        assert not is_error, text
        await _poll_until_terminal(session, started["run_id"])
        return await _call(session, "harness_inspect_run", run_id=started["run_id"])

    is_error, text, payload = _run(scenario, server_params)
    assert not is_error, text
    announced = payload["announced"]
    assert announced["mcp_servers"] == ALT_INIT_MCP_SERVERS, (
        "the mcpServers alias key must be recognised, not just mcp_servers"
    )
    assert announced["tools"] == ALT_INIT_TOOLS, (
        "dict-shaped announcement items ({'name': ...}) must resolve via item['name']"
    )


def test_inspect_run_reports_requested_names_from_argv(
    server_params, project_dir, mcp_agent_project, argv_log
):
    async def scenario(session):
        is_error, text, started = await _call(
            session,
            "harness_start_agent",
            agent="mcp-agent",
            cwd=str(mcp_agent_project),
            model="sonnet",
        )
        assert not is_error, text
        await _poll_until_terminal(session, started["run_id"])
        inspected = await _call(session, "harness_inspect_run", run_id=started["run_id"])

        # Second case: the fake CLI's init event announces nothing (NO_INIT_NAMES in the
        # task/stdin), so `requested.agents` must still come from this run's own argv —
        # the fallback the frame demands, independent of what the init event announced.
        is_error2, text2, started2 = await _call(
            session,
            "harness_start_agent",
            agent="mcp-agent",
            cwd=str(mcp_agent_project),
            model="sonnet",
            prompt=f"{NO_INIT_NAMES} Say OK.",
        )
        assert not is_error2, text2
        await _poll_until_terminal(session, started2["run_id"])
        inspected2 = await _call(session, "harness_inspect_run", run_id=started2["run_id"])

        # Third case: a payload-carrier dispatch (`demo`, no mcpServers/hooks) is the
        # only carrier whose `--agents` JSON can carry a `skills` field at all -- the
        # `mcp-agent` runs above always take the materialized carrier (they set
        # mcpServers), which never emits `--agents`.
        is_error3, text3, started3 = await _call(
            session, "harness_start_agent", agent="demo", cwd=str(project_dir), model="sonnet"
        )
        assert not is_error3, text3
        await _poll_until_terminal(session, started3["run_id"])
        inspected3 = await _call(session, "harness_inspect_run", run_id=started3["run_id"])
        return inspected, inspected2, inspected3

    (is_error, text, payload), (is_error2, text2, payload2), (is_error3, text3, payload3) = _run(
        scenario, server_params
    )
    assert not is_error, text
    assert not is_error2, text2
    assert not is_error3, text3

    records = _argv_records(argv_log)
    record = records[0]
    requested = payload["requested"]
    assert "mcp-agent" in requested["agents"]

    mcp_config = json.loads(_flag(record["argv"], "--mcp-config"))
    assert requested["mcp_servers"] == list(mcp_config["mcpServers"].keys())

    # This fixture's dispatch never sets session tools (no `.seretos/harness.yml`
    # override in play), so `--tools` is reliably absent from this run's argv --
    # asserted explicitly so the test cannot pass regardless of which branch a stub
    # implementation takes.
    assert "--tools" not in record["argv"], (
        "this fixture's dispatch does not set session tools; if that changes, the "
        "--tools-present branch needs its own coverage"
    )
    assert requested["tools"] == []
    assert "--disallowedTools" not in record["argv"], (
        "this fixture's dispatch does not set a denylist; if that changes, the "
        "--disallowedTools-present branch needs its own coverage"
    )
    assert requested["disallowed_tools"] == []

    # Second case: init event announced nothing, but the argv-derived fallback still is.
    assert payload2["announced"]["skills"] == []
    assert payload2["announced_counts"] == {
        "mcp_servers": 0, "tools": 0, "skills": 0, "agents": 0,
    }, "announced_counts must track the actually-parsed (empty) announced lists"
    assert payload2["requested"]["agents"], (
        "requested.agents must fall back to argv even when the init event announces "
        "nothing"
    )

    # Third case: requested.skills, from the --agents payload's own "skills" key.
    demo_record = records[2]
    demo_payload = _agents_payload(demo_record)["demo"]
    assert demo_payload.get("skills"), "fixture bug: demo.md must declare skills"
    assert payload3["requested"]["skills"] == demo_payload["skills"]


def test_inspect_run_requested_defaults_to_empty_lists_for_clean_run(server_params):
    async def scenario(session):
        is_error, text, started = await _call(
            session, "harness_start_prompt", prompt="Say OK.", model="sonnet"
        )
        assert not is_error, text
        await _poll_until_terminal(session, started["run_id"])
        return await _call(session, "harness_inspect_run", run_id=started["run_id"])

    is_error, text, payload = _run(scenario, server_params)
    assert not is_error, text
    requested = payload["requested"]
    assert requested["agents"] == [], "absence must be a list, never null"
    assert requested["mcp_servers"] == [], "absence must be a list, never null"
    assert requested["disallowed_tools"] == [], "absence must be a list, never null"


def test_inspect_run_returns_system_prompt_text_per_carrier(
    server_params, project_dir, mcp_agent_project, argv_log, tmp_path
):
    marker = "System prompt marker for R3 case A"

    async def scenario(session):
        is_error_a, text_a, started_a = await _call(
            session,
            "harness_start_prompt",
            prompt="Say OK.",
            model="sonnet",
            system_prompt=marker,
        )
        assert not is_error_a, text_a
        await _poll_until_terminal(session, started_a["run_id"])
        inspected_a = await _call(session, "harness_inspect_run", run_id=started_a["run_id"])

        is_error_b, text_b, started_b = await _call(
            session,
            "harness_start_agent",
            agent="demo",
            cwd=str(project_dir),
            model="sonnet",
        )
        assert not is_error_b, text_b
        await _poll_until_terminal(session, started_b["run_id"])
        inspected_b = await _call(session, "harness_inspect_run", run_id=started_b["run_id"])

        is_error_c, text_c, started_c = await _call(
            session,
            "harness_start_agent",
            agent="mcp-agent",
            cwd=str(mcp_agent_project),
            model="sonnet",
        )
        assert not is_error_c, text_c
        await _poll_until_terminal(session, started_c["run_id"])
        inspected_c = await _call(session, "harness_inspect_run", run_id=started_c["run_id"])

        return started_c, inspected_a, inspected_b, inspected_c

    started_c, (ea, ta, payload_a), (eb, tb, payload_b), (ec, tc, payload_c) = _run(
        scenario, server_params
    )
    assert not ea, ta
    assert not eb, tb
    assert not ec, tc

    records = _argv_records(argv_log)
    record_a, record_b = records[0], records[1]

    # (a) --system-prompt carrier
    sp_a = payload_a["system_prompt"]
    assert sp_a["text"] == _flag(record_a["argv"], "--system-prompt")
    assert sp_a["source"] == "--system-prompt"
    assert sp_a["chars"] == len(sp_a["text"])
    # Anchored to a digest computed independently of the production code path (not just
    # cross-checked against recorded_sha256, which a copy-both-fields stub would also
    # satisfy).
    assert sp_a["sha256"] == hashlib.sha256(sp_a["text"].encode("utf-8")).hexdigest()
    assert sp_a["sha256"] == sp_a["recorded_sha256"]

    # (b) --agents payload carrier
    sp_b = payload_b["system_prompt"]
    assert sp_b["text"] == _agents_payload(record_b)["demo"]["prompt"]
    assert sp_b["source"] == "--agents"
    assert sp_b["chars"] == len(sp_b["text"])
    assert sp_b["sha256"] == sp_b["recorded_sha256"]

    # (c) materialized carrier
    sp_c = payload_c["system_prompt"]
    materialized_path = (
        tmp_path
        / "artifacts"
        / started_c["run_id"]
        / "agents"
        / ".claude"
        / "agents"
        / "mcp-agent.md"
    )
    assert materialized_path.exists(), "fixture bug: materialized agent file missing"
    # Anchored to the fixture's own known literal body (conftest.py's mcp_agent_project),
    # not to a formula that mirrors the implementation's own frontmatter-stripping
    # expression -- that would only prove the implementation agrees with itself.
    assert sp_c["text"] == "Materialized agent body for inspection tests.\n"
    # The path half of the label matters, not just the "materialized:" prefix: it must
    # name *this* run's own materialized file.
    assert sp_c["source"] == f"materialized:{materialized_path}"
    assert sp_c["chars"] == len(sp_c["text"])
    assert sp_c["sha256"] == sp_c["recorded_sha256"]


def test_inspect_run_system_prompt_empty_but_sent_is_empty_string_not_null(server_params):
    async def scenario(session):
        is_error, text, started = await _call(
            session, "harness_start_prompt", prompt="Say OK.", model="sonnet"
        )
        assert not is_error, text
        await _poll_until_terminal(session, started["run_id"])
        return await _call(session, "harness_inspect_run", run_id=started["run_id"])

    is_error, text, payload = _run(scenario, server_params)
    assert not is_error, text
    sp = payload["system_prompt"]
    assert sp["text"] == ""
    assert sp["chars"] == 0


def test_inspect_run_while_running_and_error_paths(server_params):
    async def scenario(session):
        is_error, text, started = await _call(
            session, "harness_start_prompt", prompt="SLEEP:30", model="sonnet"
        )
        assert not is_error, text
        run_id = started["run_id"]
        await _poll_until_activity(session, run_id, "init")
        inspected_running = await _call(session, "harness_inspect_run", run_id=run_id)
        await _call(session, "harness_stop_run", run_id=run_id)

        unknown = await _call(session, "harness_inspect_run", run_id="does-not-exist")

        is_error3, text3, quick = await _call(
            session, "harness_start_prompt", prompt="Say OK.", model="sonnet"
        )
        assert not is_error3, text3
        await _poll_until_terminal(session, quick["run_id"])
        await _call(session, "harness_cleanup_run", run_id=quick["run_id"])
        cleaned = await _call(session, "harness_inspect_run", run_id=quick["run_id"])

        alive = await _call(session, "harness_list_runs")
        return inspected_running, unknown, cleaned, alive

    inspected_running, unknown, cleaned, alive = _run(scenario, server_params)

    is_error, text, payload = inspected_running
    assert not is_error, text
    assert payload["state"] == "RUNNING"
    announced = payload["announced"]
    for key, names in INIT_ANNOUNCEMENTS.items():
        assert announced[key] == names, key

    assert unknown[0] is True
    assert "HarnessError" in unknown[1]

    assert cleaned[0] is True
    assert "HarnessError" in cleaned[1]

    assert alive[0] is False, alive[1]


# --- parent MCP server set at launch (#38 R3-R5) -----------------------------------


def _git_marker(project_dir: Path) -> None:
    """`.seretos/harness.yml` discovery (`lib_python_config.walk_project_boundaries`)
    only looks inside a git repo -- it walks outward from cwd via `.git` boundaries,
    never bare directory levels, and `project_dir` (a plain tmp dir) has none. Every
    R4/R5 test that plants a `.seretos/harness.yml` and expects it to be found needs
    this marker first; R3's tests never plant one, so `load_harness_config` finds
    nothing anywhere and returns `None` -- no marker needed there."""
    (project_dir / ".git").mkdir(parents=True, exist_ok=True)


def _provision_parent_servers(config_dir: Path, project: Path) -> set[str]:
    """Plants a full parent MCP-server set across every source `parent_mcp_servers`
    is meant to read: user + local scope (with `hasTrustDialogAccepted: True`, the
    trust gate the real CLI requires before forwarding local-scope servers) in
    `config_dir/.claude.json`, an approved
    project server (`project/.mcp.json`, approved via `project/.claude/settings.json`'s
    `enableAllProjectMcpServers`), a `fixture` plugin's own `fsrv` server (keyed
    `plugin_fixture_fsrv`) and the `agent-harness` plugin's own `harness` server
    (keyed bare `harness`). Must be called together with the `plugin_agent_install`
    fixture, which is what registers the `agent-harness@mk` plugin key this reuses
    (read-merge-write onto its installed_plugins.json/settings.json, same pattern
    `test_list_agents_uses_session_cwd_and_respects_disabled_plugins` already uses).
    Returns the expected server-name set."""
    claude_json = config_dir / ".claude.json"
    data = json.loads(claude_json.read_text()) if claude_json.exists() else {}
    data.setdefault("mcpServers", {})["user-srv"] = {"command": "user-cmd"}
    projects = data.setdefault("projects", {})
    project_entry = projects.setdefault(str(project), {})
    project_entry["mcpServers"] = {"local-srv": {"command": "local-cmd"}}
    project_entry["hasTrustDialogAccepted"] = True
    claude_json.parent.mkdir(parents=True, exist_ok=True)
    claude_json.write_text(json.dumps(data), encoding="utf-8")

    (project / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"proj-srv": {"command": "proj-cmd"}}}), encoding="utf-8"
    )
    settings_dir = project / ".claude"
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / "settings.json").write_text(
        json.dumps({"enableAllProjectMcpServers": True}), encoding="utf-8"
    )

    plugins_root = config_dir.parent / "plugins"
    fixture_dir = plugins_root / "fixture"
    (fixture_dir / ".claude-plugin").mkdir(parents=True, exist_ok=True)
    (fixture_dir / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": "fixture", "mcpServers": {"fsrv": {"command": "fsrv-cmd"}}}),
        encoding="utf-8",
    )
    harness_dir = plugins_root / "agent-harness"
    (harness_dir / ".claude-plugin").mkdir(parents=True, exist_ok=True)
    (harness_dir / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": "agent-harness", "mcpServers": {"harness": {"command": "harness-cmd"}}}),
        encoding="utf-8",
    )

    installed_path = config_dir / "plugins" / "installed_plugins.json"
    installed = json.loads(installed_path.read_text())
    installed["plugins"]["fixture@mk"] = [{"scope": "user", "installPath": str(fixture_dir)}]
    installed_path.write_text(json.dumps(installed), encoding="utf-8")
    settings_path = config_dir / "settings.json"
    settings = json.loads(settings_path.read_text())
    settings["enabledPlugins"]["fixture@mk"] = True
    settings_path.write_text(json.dumps(settings), encoding="utf-8")

    return {"user-srv", "local-srv", "proj-srv", "plugin_fixture_fsrv", "harness"}


@pytest.mark.parametrize("carrier", ["agent-harness:probe", "demo", "mcp-agent"])
def test_start_agent_launch_carries_parent_server_set(
    server_params, plugin_agent_install, project_dir, mcp_agent_project, argv_log, carrier
):
    """R3: whichever cwd/agent is dispatched, the child's --mcp-config carries the
    full parent server set `_provision_parent_servers` planted -- for a plugin-scope
    agent (agent-harness:probe, whose own frontmatter carries no mcpServers), a
    project-scope agent with no mcpServers of its own (demo), and one that does
    declare its own mcpServers (mcp-agent's demo-server, which must survive
    alongside the parent set, not be replaced by it). No --strict-mcp-config (no
    harness.yml is involved here).

    Expected RED reason: today host_context never populates ctx.mcp_servers, so
    resolve() never puts anything into spec.mcp_servers for `agent-harness:probe`
    or `demo` (no --mcp-config is emitted at all -- _flag raises), and `mcp-agent`'s
    own frontmatter mcpServers (demo-server) is the *only* thing in --mcp-config,
    not the parent set plus demo-server.
    """
    cwd = mcp_agent_project if carrier == "mcp-agent" else project_dir
    config_dir = Path(server_params.env["CLAUDE_CONFIG_DIR"])
    expected = _provision_parent_servers(config_dir, cwd)
    if carrier == "mcp-agent":
        expected = expected | {"demo-server"}

    started, final = _start_and_finish(server_params, agent=carrier, cwd=str(cwd), model="sonnet")
    assert final["state"] == "COMPLETED", final
    (record,) = _argv_records(argv_log)
    assert "--strict-mcp-config" not in record["argv"]
    mcp_config = json.loads(_flag(record["argv"], "--mcp-config"))
    assert set(mcp_config["mcpServers"].keys()) == expected

    async def scenario(session):
        return await _call(session, "harness_inspect_run", run_id=started["run_id"])

    is_error, text, inspected = _run(scenario, server_params)
    assert not is_error, text
    assert set(inspected["requested"]["mcp_servers"]) == expected


def test_start_agent_send_message_replay_keeps_same_mcp_config(
    server_params, plugin_agent_install, project_dir, argv_log
):
    """R3 additional edge-case coverage: a harness_send_message follow-up replays
    the origin run's own argv (start_resume), so its --mcp-config must carry the
    same parent server set as the origin, not a freshly recomputed (and possibly
    different) one."""
    config_dir = Path(server_params.env["CLAUDE_CONFIG_DIR"])
    expected = _provision_parent_servers(config_dir, project_dir)

    async def scenario(session):
        is_error, text, started = await _call(
            session, "harness_start_agent", agent="demo", cwd=str(project_dir), model="sonnet"
        )
        assert not is_error, text
        origin = await _poll_until_terminal(session, started["run_id"])
        sent = await _call(session, "harness_send_message", run_id=origin["run_id"], prompt="hi")
        assert not sent[0], sent[1]
        follow_up = await _poll_until_terminal(session, sent[2]["run_id"])
        return follow_up

    _run(scenario, server_params)
    records = _argv_records(argv_log)
    assert len(records) == 2
    for record in records:
        mcp_config = json.loads(_flag(record["argv"], "--mcp-config"))
        assert set(mcp_config["mcpServers"].keys()) == expected


def test_start_agent_applies_harness_yml(
    server_params, plugin_agent_install, project_dir, argv_log
):
    """R4: a `.seretos/harness.yml` `agents: demo:` entry's `mcpServers.remove` and
    `canSpawn: true` take effect once the config is actually loaded and passed to
    resolve() -- both named servers are gone from --mcp-config, --strict-mcp-config
    is present (config-driven runs are always strict), and the rest of the parent
    set (including the granted `harness` dispatch server) remains.

    Expected RED reason: harness_start_agent never loads .seretos/harness.yml today
    (no `config=` is passed to resolve()), so apply_config never runs at all: no
    server is removed and --strict-mcp-config never appears.
    """
    config_dir = Path(server_params.env["CLAUDE_CONFIG_DIR"])
    provisioned = _provision_parent_servers(config_dir, project_dir)
    _git_marker(project_dir)
    (project_dir / ".seretos").mkdir(parents=True, exist_ok=True)
    (project_dir / ".seretos" / "harness.yml").write_text(
        "agents:\n"
        "  demo:\n"
        "    canSpawn: true\n"
        "    mcpServers:\n"
        "      remove: [user-srv, plugin_fixture_fsrv]\n",
        encoding="utf-8",
    )

    started, final = _start_and_finish(server_params, agent="demo", cwd=str(project_dir), model="sonnet")
    assert final["state"] == "COMPLETED", final
    (record,) = _argv_records(argv_log)
    assert "--strict-mcp-config" in record["argv"]
    mcp_config = json.loads(_flag(record["argv"], "--mcp-config"))
    assert set(mcp_config["mcpServers"].keys()) == (provisioned - {"user-srv", "plugin_fixture_fsrv"})


def test_start_agent_harness_yml_remove_uses_the_prefixed_catalogue_key(
    server_params, plugin_agent_install, project_dir, argv_log
):
    """R4 additional edge-case coverage: `remove:` must name the catalogue key
    (`plugin_fixture_fsrv`), not the plugin manifest's own bare server name
    (`fsrv`) -- the latter matches nothing and leaves the server present. This
    documents which form `.seretos/harness.yml` actually needs. The `demo` entry
    here sets no `canSpawn`, so per R5's own default (v0.0.8: canSpawn false) the
    granted dispatch server `harness` must be absent too -- this is a
    config-driven (strict) launch like any other `agents:` entry, not a bypass of
    the canSpawn default. (Fixed per round-1 test-critic tautology::F2: the
    previous version of this test wrongly asserted `harness` present, which only
    an implementation that ignores harness.yml -- or overrides canSpawn -- could
    satisfy.)"""
    config_dir = Path(server_params.env["CLAUDE_CONFIG_DIR"])
    provisioned = _provision_parent_servers(config_dir, project_dir)
    _git_marker(project_dir)
    (project_dir / ".seretos").mkdir(parents=True, exist_ok=True)
    (project_dir / ".seretos" / "harness.yml").write_text(
        "agents:\n  demo:\n    mcpServers:\n      remove: [fsrv]\n", encoding="utf-8"
    )

    started, final = _start_and_finish(server_params, agent="demo", cwd=str(project_dir), model="sonnet")
    assert final["state"] == "COMPLETED", final
    (record,) = _argv_records(argv_log)
    assert "--strict-mcp-config" in record["argv"]
    mcp_config = json.loads(_flag(record["argv"], "--mcp-config"))
    assert "plugin_fixture_fsrv" in mcp_config["mcpServers"]
    assert set(mcp_config["mcpServers"].keys()) == (provisioned - {"harness"})


def test_start_agent_invalid_harness_yml_raises_config_error(
    server_params, project_dir, argv_log
):
    """R4 additional edge-case coverage: an invalid .seretos/harness.yml (unknown
    key) is surfaced as a ToolError naming ConfigError, and no child is started."""
    _git_marker(project_dir)
    (project_dir / ".seretos").mkdir(parents=True, exist_ok=True)
    (project_dir / ".seretos" / "harness.yml").write_text(
        "agents:\n  demo:\n    notARealKey: true\n", encoding="utf-8"
    )

    async def scenario(session):
        return await _call(
            session, "harness_start_agent", agent="demo", cwd=str(project_dir), model="sonnet"
        )

    is_error, text, _ = _run(scenario, server_params)
    assert is_error, "an invalid harness.yml must refuse the launch"
    assert "ConfigError" in text
    assert not argv_log.exists()


def test_start_agent_can_spawn_follows_lib_default(
    server_params, plugin_agent_install, project_dir, argv_log
):
    """R5 (AC4): once .seretos/harness.yml is loaded, whether the `harness`
    dispatch server is granted follows only the lib's own canSpawn default (false)
    and the file's explicit override (true) -- no override of that default lives in
    this repo. Both dispatches go strict (config-driven), proving the config really
    loaded for both; only the dispatch server's presence differs.

    Expected RED reason: harness.yml is never loaded today, so neither dispatch is
    strict at all (--strict-mcp-config absent from both).
    """
    config_dir = Path(server_params.env["CLAUDE_CONFIG_DIR"])
    _provision_parent_servers(config_dir, project_dir)
    _git_marker(project_dir)
    (project_dir / ".seretos").mkdir(parents=True, exist_ok=True)
    (project_dir / ".seretos" / "harness.yml").write_text(
        "agents:\n"
        "  demo: {}\n"
        "  effort-agent:\n"
        "    canSpawn: true\n",
        encoding="utf-8",
    )

    started_a, final_a = _start_and_finish(server_params, agent="demo", cwd=str(project_dir), model="sonnet")
    assert final_a["state"] == "COMPLETED", final_a
    started_b, final_b = _start_and_finish(
        server_params, agent="effort-agent", cwd=str(project_dir), model="sonnet"
    )
    assert final_b["state"] == "COMPLETED", final_b

    record_a, record_b = _argv_records(argv_log)
    assert "--strict-mcp-config" in record_a["argv"], "demo has no canSpawn -- must still go strict"
    assert "--strict-mcp-config" in record_b["argv"]
    mcp_config_a = json.loads(_flag(record_a["argv"], "--mcp-config"))
    mcp_config_b = json.loads(_flag(record_b["argv"], "--mcp-config"))
    assert "harness" not in mcp_config_a["mcpServers"], "canSpawn defaults false (lib v0.0.8)"
    assert "harness" in mcp_config_b["mcpServers"], "canSpawn: true must grant the dispatch server"


# --- launched-agent identity (#51) -------------------------------------------------
#
# A harness-launched child is a fully independent top-level `claude` CLI process:
# it has no equivalent of native Claude Code's own SubagentStop `agent_type` field,
# and (until this ticket) no signal in its own environment either. These tests pin
# HARNESS_LAUNCHED_AGENT (harness_plugin.runs.LAUNCHED_AGENT_ENV) as that signal, for
# both a `harness_start_agent` child and a `harness_send_message` follow-up on it,
# and its absence for a `harness_start_prompt` child (never any agent at all) --
# including when the server process's own env happens to carry a stale value.


def test_start_agent_exposes_launched_agent_env(
    server_params, plugin_agent_install, project_dir, argv_log
):
    """R1: harness_start_agent(agent="agent-harness:probe") spawns a child whose env
    has HARNESS_LAUNCHED_AGENT == "agent-harness:probe" -- the qualified name -- so a
    hook running inside that child (e.g. its own Stop hook) can tell which agent it
    is running as.

    Expected RED reason: None != "agent-harness:probe" (fake_claude's own env carries
    no HARNESS_LAUNCHED_AGENT at all today -- nothing sets it)."""
    started, final = _start_and_finish(
        server_params, agent="agent-harness:probe", cwd=str(project_dir), model="sonnet"
    )
    assert final["state"] == "COMPLETED"
    (record,) = _argv_records(argv_log)
    assert record["launched_agent"] == "agent-harness:probe"


def test_start_agent_launched_agent_env_is_bare_name_for_project_agent(
    server_params, project_dir, argv_log
):
    """R1 additional edge-case coverage: a project-scope agent (no plugin prefix)
    reports its own bare name unqualified -- there is no colon to split off."""
    started, final = _start_and_finish(
        server_params, agent="demo", cwd=str(project_dir), model="sonnet"
    )
    assert final["state"] == "COMPLETED"
    (record,) = _argv_records(argv_log)
    assert record["launched_agent"] == "demo"


def test_identity_provider_materialized_launch_uses_qualified_name(tmp_path):
    """R1 additional edge-case coverage: a materialized-carrier launch (forced here
    via accepted_keys=frozenset(), since RunSpec.prompt is always set and so is
    never in an empty accepted set -- see dispatch_mode) must still set
    HARNESS_LAUNCHED_AGENT to the run's qualified agent name ("p:a"), not the file
    stem ("p__a") the materialized carrier's own --agent value uses.

    Expected RED reason: None != "p:a" (_AgentIdentityProvider is still a plain
    pass-through subclass; nothing sets the var on the plan's env yet)."""
    from lib_python_harness import Isolation, RunSpec

    from harness_plugin.runs import LAUNCHED_AGENT_ENV, _AgentIdentityProvider

    provider = _AgentIdentityProvider()
    spec = RunSpec(
        prompt="Say OK.",
        isolation=Isolation.INHERIT,
        model="sonnet",
        agent_name="p:a",
        cwd=tmp_path,
        allow_nonempty_cwd=True,
    )
    plan = provider.build_launch_plan(
        spec, session_id="sess", run_dir=tmp_path, accepted_keys=frozenset()
    )
    assert plan.env.get(LAUNCHED_AGENT_ENV) == "p:a"


def test_start_prompt_strips_inherited_launched_agent_env(server_params, argv_log):
    """R2: harness_start_prompt (a CLEAN child, never any agent) must never carry
    HARNESS_LAUNCHED_AGENT, even when the server process's own env happens to have
    one set -- a CLEAN child inheriting a stale value from the parent process would
    report a false identity.

    Expected RED reason: "stale:parent" is not None (the server process's own env is
    copied straight into the child's plan by _scrub_env(); nothing pops the var for
    a CLEAN/no-agent launch today)."""
    env = dict(server_params.env)
    env["HARNESS_LAUNCHED_AGENT"] = "stale:parent"
    params = server_params.model_copy(update={"env": env})

    async def scenario(session):
        is_error, text, started = await _call(
            session, "harness_start_prompt", prompt="Say OK.", model="sonnet"
        )
        assert not is_error, text
        return started, await _poll_until_terminal(session, started["run_id"])

    started, final = _run(scenario, params)
    assert final["state"] == "COMPLETED"
    (record,) = _argv_records(argv_log)
    assert record["launched_agent"] is None


def test_start_agent_overrides_stale_launched_agent_env(
    server_params, plugin_agent_install, project_dir, argv_log
):
    """R2 additional edge-case coverage: the same stale server-env value does not
    leak into an agent run either -- the child gets the actually-launched agent's
    own name, not the stale value."""
    env = dict(server_params.env)
    env["HARNESS_LAUNCHED_AGENT"] = "stale:parent"
    params = server_params.model_copy(update={"env": env})

    started, final = _start_and_finish(
        params, agent="agent-harness:probe", cwd=str(project_dir), model="sonnet"
    )
    assert final["state"] == "COMPLETED"
    (record,) = _argv_records(argv_log)
    assert record["launched_agent"] == "agent-harness:probe"


def test_send_message_keeps_launched_agent_env(
    server_params, plugin_agent_install, project_dir, argv_log
):
    """R3: harness_send_message on a finished agent run spawns a follow-up child
    that still carries the origin run's own HARNESS_LAUNCHED_AGENT -- a follow-up
    turn is still that same agent, so a Stop hook in the resumed child must see the
    same identity the first turn did.

    Expected RED reason: records[1]["launched_agent"] is None (build_resume_plan
    replays the origin's argv, but nothing sets the identity env var on either the
    first turn or the follow-up today)."""

    async def scenario(session):
        is_error, text, started = await _call(
            session,
            "harness_start_agent",
            agent="agent-harness:probe",
            cwd=str(project_dir),
            model="sonnet",
        )
        assert not is_error, text
        origin = await _poll_until_terminal(session, started["run_id"])
        sent = await _call(
            session, "harness_send_message", run_id=origin["run_id"], prompt="ECHO:X"
        )
        assert not sent[0], sent[1]
        follow_up = await _poll_until_terminal(session, sent[2]["run_id"])
        return origin, follow_up

    origin, follow_up = _run(scenario, server_params)
    assert origin["state"] == "COMPLETED"
    assert follow_up["state"] == "COMPLETED"
    assert follow_up["text"] == "X"
    records = _argv_records(argv_log)
    assert len(records) == 2
    assert records[1]["launched_agent"] == "agent-harness:probe"


def test_identity_provider_build_resume_plan_recovers_name_from_materialized_argv(tmp_path):
    """R3 additional edge-case coverage: a materialized-carrier origin's recorded
    argv carries only the file-stem form of the agent name (--agent p__a, no
    --agents payload) -- build_resume_plan must reverse the stem back to the
    qualified name ("p:a") for the follow-up's own HARNESS_LAUNCHED_AGENT, per the
    plan's documented `__` <-> `:` reversal (materialized carrier only; a bare
    agent name that itself contains "__" is a known, documented limitation, not
    something this test exercises)."""
    from harness_plugin.runs import LAUNCHED_AGENT_ENV, _AgentIdentityProvider

    provider = _AgentIdentityProvider()
    argv = ["-p", "--model", "sonnet", "--agent", "p__a"]
    plan = provider.build_resume_plan(
        provider_argv=argv, session_id="sess", cwd=str(tmp_path), prompt="hi"
    )
    assert plan.env.get(LAUNCHED_AGENT_ENV) == "p:a"


def test_identity_provider_build_resume_plan_keeps_verbatim_name_from_payload_argv(tmp_path):
    """R3 additional edge-case coverage: a payload-carrier origin's recorded argv
    carries --agents alongside --agent, so the --agent value is already the real
    qualified name verbatim -- including a literal "__" that happens to be part of
    the agent's own bare name -- and must not be reversed."""
    from harness_plugin.runs import LAUNCHED_AGENT_ENV, _AgentIdentityProvider

    provider = _AgentIdentityProvider()
    argv = [
        "-p", "--model", "sonnet",
        "--agents", json.dumps({"my__agent": {"prompt": "hi"}}),
        "--agent", "my__agent",
    ]
    plan = provider.build_resume_plan(
        provider_argv=argv, session_id="sess", cwd=str(tmp_path), prompt="hi"
    )
    assert plan.env.get(LAUNCHED_AGENT_ENV) == "my__agent"


def test_identity_provider_build_resume_plan_pops_env_for_prompt_origin(tmp_path, monkeypatch):
    """R3 additional edge-case coverage: a harness_start_prompt origin's recorded
    argv has no --agent at all -- the follow-up must not carry any
    HARNESS_LAUNCHED_AGENT, stale or otherwise. The process env is seeded with a
    stale value first (build_resume_plan's base env comes from `_scrub_env()`,
    which copies `os.environ`) so this actually exercises the pop -- without the
    seed, the assertion would hold vacuously on a pytest process that never had
    the var set in the first place."""
    from harness_plugin.runs import LAUNCHED_AGENT_ENV, _AgentIdentityProvider

    monkeypatch.setenv(LAUNCHED_AGENT_ENV, "stale:leftover")
    provider = _AgentIdentityProvider()
    argv = ["-p", "--model", "sonnet"]
    plan = provider.build_resume_plan(
        provider_argv=argv, session_id="sess", cwd=str(tmp_path), prompt="hi"
    )
    assert LAUNCHED_AGENT_ENV not in plan.env
