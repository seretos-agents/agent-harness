# agent-harness

Provider-independent subagent system for coding agents - replaces the host's built-in subagent tool calls.

## Key features

- Start subagent runs (`harness_start_agent`, `harness_start_prompt`) that return a `run_id` immediately, then poll, wait, follow up (`harness_send_message`), list, stop and clean up.
- Non-destructive waiting: `harness_wait_run` never cancels a run when its timeout expires - it returns the still-RUNNING run with liveness info (`duration_s`, `event_count`, `last_event_at`, `last_activity`) and a next-step hint. Only `harness_stop_run` cancels.
- `harness wait <run_id>` waits from a background shell until the run ends and reports the outcome as an exit code.
- Ships three standard agents: `agent-harness:general-purpose` (every tool - edits files, runs commands), and `agent-harness:Explore` and `agent-harness:Plan` (read-only - read, search and web fetch, no shell).
- Self-contained binary for Windows and Linux; no Python toolchain needed.
