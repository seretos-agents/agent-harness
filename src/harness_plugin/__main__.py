import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "hook":
        # Dispatched before importing mcp/the lib: hooks spawn once per tool call.
        from harness_plugin.hooks.write_context import main as hook_main

        sys.exit(hook_main())
    if len(sys.argv) > 1 and sys.argv[1] == "wait":
        # Also before mcp: a background wait process needs no FastMCP.
        from harness_plugin.wait_run import main as wait_run_main

        sys.exit(wait_run_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "run-agent":
        # Also before mcp: blocking entry point for native `Agent` calls (#76).
        from harness_plugin.run_agent import main as run_agent_main

        sys.exit(run_agent_main(sys.argv[2:]))
    from harness_plugin.server import main

    main()
