---
name: harness-wait
description: Use when a wait on a harness subagent run (started with harness_start_agent / harness_start_prompt) came back without a result - harness_wait_run's `timeout_seconds` elapsed, `harness wait` exited 2, or the run has already outlasted several tool calls - and when you want an unattended wait from a background shell. An elapsed wait limit never cancels the run: it is still RUNNING and still doing work you have already paid for, so the answer is to keep waiting, not to restart or replace it. Covers both wait paths, how to read a run's liveness fields to tell a working run from a hung one, the `harness wait` command and its exit codes.
---

# Waiting for a harness run without cancelling it

No wait limit in this plugin ends a run. `harness_wait_run`'s `timeout_seconds` and the
`harness wait` command's `--timeout` end only the waiting; the run itself keeps going.
Only `harness_stop_run` cancels a run (terminal state CANCELLED).

So when a wait limit elapses on an unfinished run: keep waiting. Do not restart the run,
do not start a replacement, and do not report the run as cancelled, failed or dead - the
work it has already done is still running and still paid for.

An elapsed limit is not news about the run. It tells you only that the run needed longer
than you allowed it this time; it says nothing about whether the run is healthy. What the
run is actually doing is in its `state` and its liveness fields, and you can read those at
any moment with `harness_poll_run` - see "Judging progress". A long run outlasts the limit
many times over; each expiry is another point to continue from, not a new fact.

## Flow

1. Start the run (`harness_start_agent` or `harness_start_prompt`) and note the `run_id`.
2. Wait for the result on path A or path B below. Either can follow the other: an elapsed
   limit in one is a reason to continue in the other, never a reason to stop.
3. When a limit elapses, read the liveness fields off the answer it returned. If you do
   not have that answer in front of you - the return was lost, the context was compacted,
   you are picking up a run somebody else started - call `harness_poll_run(run_id)`: it
   returns immediately with the same `state` and the same liveness fields. A missing
   answer is never evidence of a dead run; poll before you conclude anything.
4. If you lost the `run_id` (e.g. after context compaction), find the run with
   `harness_list_runs` first. Once you have the result, call `harness_cleanup_run` to
   forget the run record.
5. To continue a finished run's conversation, call `harness_send_message(run_id, prompt)`:
   it returns a **new** `run_id` (state RUNNING); wait on that one the same way.

## Path A - blocking tool call: `harness_wait_run(run_id, timeout_seconds)`

Blocks inside the MCP tool call for up to `timeout_seconds` (default 300). Use it for the
first wait, and whenever the run has a fair chance of finishing within that limit.

If the limit expires first, the result comes back with `state: RUNNING`, the liveness
fields, and a `next_step` hint telling you how to continue.

Continue by calling `harness_wait_run` again with a fresh `timeout_seconds` - there is no
cap on how often you may do that - or switch to path B. Once a limit has elapsed twice on
the same run, the run has shown it needs longer than you are giving it: switch to path B
instead of spending more tool calls blocking.

## Path B - unattended wait in a background shell: `harness wait <run_id>`

Call the Bash tool with `run_in_background: true` and the command:

```
<harness-binary> wait <run_id> [--timeout <seconds>] [--interval <seconds>]
```

`--timeout` is optional: omit it and the command waits until the run ends, which is the
right default for an unattended wait. Give it only when you want the shell to report back
after a bounded stretch. `--interval` (default 2, minimum 0.2) is the poll cadence.

You are notified when the command exits. Read its stdout: exactly one JSON object shaped
like a `harness_poll_run` result, plus `waited_s` (seconds actually waited). The exit code
says what happened.

## Judging progress

A wait result and a `harness_poll_run` result carry the same fields:

- `state` - RUNNING while the run is still going; COMPLETED, FAILED or CANCELLED once it
  has ended. This, not an elapsed wait limit, is what the run's condition is read from.
- `duration_s` - how long the **run** has been going, not how long you waited. It keeps
  growing across waits, so a `duration_s` many times your wait limit is normal for a long
  job and means nothing on its own. (Path B's JSON adds `waited_s`: the length of that
  wait alone.)
- `event_count` and `last_event_at` - progress markers that advance only while `state` is
  RUNNING; once the run is terminal they reset to 0 and None, so only compare them on a
  RUNNING run.
- `last_activity` - the last tool or command the run used.

Compare a reading against the one before it:

- `event_count` higher or `last_event_at` later than the previous reading: the run is
  working. Keep waiting.
- Both identical to the previous reading: that is evidence only if the two readings are at
  least one full wait limit apart. A single step of a run can sit inside one tool call for
  minutes, so two polls seconds apart tell you nothing. Take the second reading from the
  next `harness_wait_run` return (300 s by default) rather than from a poll issued right
  after the first.
- Both identical across two readings a full wait apart: the run may be hung. That is the
  only situation in which ending it is on the table at all, and it is a deliberate
  decision to state explicitly - `harness_stop_run` is the only thing that cancels a run,
  and it discards everything the run has done. An elapsed wait limit is never a reason to
  call it.

## Exit codes

| code | meaning |
| ---- | ------- |
| 0 | run COMPLETED |
| 1 | run FAILED |
| 2 | `--timeout` (if given) elapsed; the run is left RUNNING, never cancelled - judge progress from the printed liveness fields, then wait again (path A or path B) |
| 3 | run CANCELLED (e.g. by `harness_stop_run`) |
| 4 | error: unknown run id, unreadable artifacts dir, or invalid arguments (message on stderr, nothing on stdout) |

## Finding the binary

The plugin ships `bin/harness` (Linux) and `bin/harness.exe` (Windows). Resolve it without
relying on placeholder substitution - take the first of these that yields a path:

1. `command -v harness.exe || command -v harness` - works when the plugin `bin/` is on
   PATH. Ask for `harness.exe` first, as written: on Windows the bare name fails with exit
   126 or 127.
2. Glob the plugin cache: `~/.claude/plugins/**/agent-harness*/bin/harness*`.

Pass the same environment the MCP server runs with, in particular `HARNESS_ARTIFACTS_DIR`
if it is set - the command finds runs through the shared artifacts directory. If it is not
set, both sides fall back to `~/.agent-harness/runs`, so you need do nothing about it.
