agent-harness Stop hook: waited for unfinished run(s) {run_ids}

This is expected -- do not abort or treat it as an error. Do not call harness_wait_run, do not run `harness wait` in a shell, and do not start any other polling loop: the hook itself blocks and waits again the next time you end your turn.

For each run named above, call harness_poll_run exactly once, right now, and judge from its result:

- Terminal (COMPLETED/FAILED/CANCELLED): use the result and continue.
- RUNNING, and event_count or last_event_at has advanced since your previous reading of this run (or this is your first reading of it): the run is making progress. End your turn now; you will be reactivated to check again.
- RUNNING, unchanged since your previous reading, with last_activity naming a call that can legitimately run long (a shell command, build, test, install, or sub-agent dispatch), and this is the first silent reading: give it one wait of grace. End your turn now.
- RUNNING, unchanged for a second consecutive silent reading, or silent with a last_activity that is not a long-running call: the run is stalled. Call harness_stop_run(run_id) to cancel it, then end your turn with a short report naming the run_id and the last_activity value from the poll that triggered the cancel.
