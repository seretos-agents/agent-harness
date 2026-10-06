"""`harness run-agent`: the blocking entry point behind native `Agent` calls.

Starts a run for a named agent, waits for it in this same process (it owns the child; no
timeout, never cancels) and prints one `run_to_dict` JSON line. Imports no mcp/FastMCP code.

Exit codes (wait_run's constants): 0 COMPLETED, 1 FAILED, 3 CANCELLED, 4 error -- unknown
agent, refused context, bad arguments. On 4 stdout is empty and the message is on stderr."""
from __future__ import annotations

import argparse
import json
import sys
from typing import NoReturn

from lib_python_harness import HarnessError, RunState

from harness_plugin.runs import StartRefused, harness, run_to_dict, start_agent
from harness_plugin.wait_run import (
    EXIT_CANCELLED,
    EXIT_COMPLETED,
    EXIT_ERROR,
    EXIT_FAILED,
)

# Native subagent types whose definitions this plugin ships; discovery keys plugin agents
# by qualified name only, so the bare native name needs this mapping. Any other value is
# passed as given (a bare project/user agent name or an already-qualified name).
_BUILTIN_AGENT_TYPES = ("general-purpose", "Explore", "Plan")

_EPILOG = """\
exit codes:
  0  run COMPLETED
  1  run FAILED
  3  run CANCELLED (stopped from another process)
  4  error: unknown agent, refused session context, invalid arguments

There is no timeout: the call blocks until the run ends and never cancels it.
stdout is exactly one JSON object shaped like harness_poll_run's result."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        sys.exit(EXIT_ERROR)


def _parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="harness run-agent",
        description="Run a subagent to completion and print its result.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--subagent-type", required=True, help="agent name (native subagent_type)")
    p.add_argument("--prompt", required=True, help="the run's task")
    p.add_argument("--model", default=None, help="model override (default: the agent's / session's)")
    p.add_argument("--description", default=None, help="short label for the run")
    return p


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    agent = args.subagent_type
    if agent in _BUILTIN_AGENT_TYPES:
        agent = f"agent-harness:{agent}"
    try:
        started, _used, _source = start_agent(
            agent,
            cwd=None,
            model=args.model or None,
            permission_mode=None,
            effort=None,
            label=args.description or None,
            prompt=args.prompt,
        )
        result = harness().wait(started.run_id, timeout=None, poll_interval=1.0)
    except (StartRefused, HarnessError, OSError) as exc:
        print(f"harness run-agent: {exc}", file=sys.stderr)
        return EXIT_ERROR
    sys.stdout.write(json.dumps(run_to_dict(result)) + "\n")
    sys.stdout.flush()
    return {
        RunState.COMPLETED: EXIT_COMPLETED,
        RunState.FAILED: EXIT_FAILED,
        RunState.CANCELLED: EXIT_CANCELLED,
    }.get(result.state, EXIT_ERROR)
