import type { On } from 'claude-code'

/**
 * agent-harness: answers a native `Agent` (alias `Task`) call by running the subagent
 * through the harness (`harness run-agent`, which blocks until the run ends) and handing
 * the run's answer back in the native Agent result shape. Never calls `next(e)`: the call
 * is answered here, so no native subagent ever runs. If this module fails to load, the
 * classic PreToolUse hook in hooks.json still denies the native call (#52).
 *
 * Requires Claude Code >= 2.1.291 (Function Hooks early access, `$.process.spawn`).
 *
 * The parent's permission mode and effort are not on `tool.call`; `run-agent` reads them
 * from the session file the classic `harness hook` writes (refreshed on every
 * UserPromptSubmit), so they are as fresh as the last prompt of the turn.
 */

type Spawned = { exitCode?: number; code?: number; status?: number; stdout?: string; stderr?: string }

const BUILTIN_RESULT_NOTE = 'agent-harness could not run Agent through the harness'

/** The binary under bin/ for the host OS, named directly (the extensionless dispatcher is a POSIX script). */
function binaryOf(root: string): string {
  const isWindows = /^[A-Za-z]:/.test(root) || root.includes('\\')
  const sep = isWindows ? '\\' : '/'

  return `${root.replace(/[\\/]+$/, '')}${sep}bin${sep}${isWindows ? 'harness.exe' : 'harness-linux'}`
}

function fail(text: string) {
  return { result: { isError: true, text } }
}

/** Run `harness <argv>` and map its exit code to the native tool result (0 completed; 1/3 failed/cancelled run; else error). */
async function runHarness($: { process: { spawn: (...a: any[]) => Promise<unknown> }; plugin: { root: string } }, argv: string[], what: string) {
  const done = (await $.process.spawn(binaryOf($.plugin.root), argv)) as Spawned
  const code = done.exitCode ?? done.code ?? done.status
  const stdout = (done.stdout ?? '').trim()
  const stderr = (done.stderr ?? '').trim()

  if (code === 0 || code === 1 || code === 3) {
    const run = JSON.parse(stdout.split('\n').filter(line => line.trim() !== '').pop() ?? '{}') as Record<string, unknown>
    const text = typeof run.text === 'string' ? run.text : ''

    if (code !== 0) {
      return fail(`harness run ${String(run.run_id)} ended ${String(run.state)}: ${text}`)
    }

    return {
      result: {
        status: 'completed',
        agentId: run.run_id,
        content: [{ type: 'text', text }],
        usage: run.usage ?? {},
      },
    }
  }

  return fail(`${BUILTIN_RESULT_NOTE}: ${stderr !== '' ? stderr : `harness ${what} exited ${String(code)}`}`)
}

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

export function register(on: On) {
  on('tool.call', { tool: ['Agent', 'Task'] }, async ($, e) => {
    try {
      const args = e as unknown as Readonly<Record<string, unknown>>
      const str = (key: string): string | undefined => (typeof args[key] === 'string' && args[key] !== '' ? (args[key] as string) : undefined)
      const prompt = str('prompt')

      if (prompt === undefined) {
        return fail(`${BUILTIN_RESULT_NOTE}: the Agent call has no prompt`)
      }

      const argv = ['run-agent', '--subagent-type', str('subagent_type') ?? 'general-purpose', '--prompt', prompt]
      const model = str('model')
      const description = str('description')

      if (model !== undefined) {
        argv.push('--model', model)
      }

      if (description !== undefined) {
        argv.push('--description', description)
      }

      return await runHarness($, argv, 'run-agent')
    } catch (error) {
      return fail(`${BUILTIN_RESULT_NOTE}: ${error instanceof Error ? error.message : String(error)}`)
    }
  })
  // #78: a SendMessage whose `to` is a harness run id (a canonical UUID -- what `Agent`
  // returned as `agentId`) resumes that run's chain; any other recipient stays native.
  on('tool.call', { tool: ['SendMessage'] }, async ($, e, next) => {
    try {
      const args = e as unknown as Readonly<Record<string, unknown>>
      const to = args.to
      const message = args.message

      if (typeof to !== 'string' || !UUID.test(to) || typeof message !== 'string' || message.trim() === '') {
        return next(e)
      }

      return await runHarness($, ['send-message', '--to', to, '--message', message], 'send-message')
    } catch (error) {
      return fail(`${BUILTIN_RESULT_NOTE}: ${error instanceof Error ? error.message : String(error)}`)
    }
  })
}
