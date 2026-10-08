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

const BUILTIN_RESULT_NOTE = 'agent-harness could not run Agent through the harness'

type Chunk = { stream?: string; text?: string }

type Dollar = {
  process: { spawn: (o: { argv: string[] }) => AsyncIterable<Chunk> }
  plugin: { root: string }
}

/** The binary under bin/ for the host OS, named directly (the extensionless dispatcher is a POSIX script). */
function binaryOf(root: string): string {
  const isWindows = /^[A-Za-z]:/.test(root) || root.includes('\\')
  const sep = isWindows ? '\\' : '/'

  return `${root.replace(/[\\/]+$/, '')}${sep}bin${sep}${isWindows ? 'harness.exe' : 'harness-linux'}`
}

/**
 * An error answer. A resolved `result` is validated against the tool's output schema, which
 * has no error channel (an `{isError}` object fails it, #82); `deny` is rendered as an
 * is_error tool_result instead.
 */
function fail(text: string) {
  return { deny: text }
}

const num = (value: unknown): number => (typeof value === 'number' && Number.isFinite(value) ? value : 0)
const nullableNum = (value: unknown): number | null => (typeof value === 'number' && Number.isFinite(value) ? value : null)

/** Keep a usage sub-object only when every field is a number, else null (the schema's nullable keys). */
function subUsage(value: unknown, keys: string[]): Record<string, number> | null {
  if (typeof value !== 'object' || value === null) {
    return null
  }

  const obj = value as Record<string, unknown>

  if (!keys.every(key => typeof obj[key] === 'number' && Number.isFinite(obj[key] as number))) {
    return null
  }

  return Object.fromEntries(keys.map(key => [key, obj[key] as number]))
}

function normUsage(raw: unknown) {
  const u = (typeof raw === 'object' && raw !== null ? raw : {}) as Record<string, unknown>

  return {
    input_tokens: num(u.input_tokens),
    output_tokens: num(u.output_tokens),
    cache_creation_input_tokens: nullableNum(u.cache_creation_input_tokens),
    cache_read_input_tokens: nullableNum(u.cache_read_input_tokens),
    server_tool_use: subUsage(u.server_tool_use, ['web_search_requests', 'web_fetch_requests']),
    service_tier: typeof u.service_tier === 'string' ? u.service_tier : null,
    cache_creation: subUsage(u.cache_creation, ['ephemeral_1h_input_tokens', 'ephemeral_5m_input_tokens']),
  }
}

/** The run record as the native Agent `completed` output (Claude Code 2.1.294's output shape). */
function agentOutput(run: Record<string, unknown>, prompt: string) {
  const usage = normUsage(run.usage)
  const text = typeof run.text === 'string' ? run.text : ''

  return {
    status: 'completed' as const,
    prompt,
    agentId: String(run.run_id),
    content: [{ type: 'text' as const, text }],
    totalToolUseCount: 0,
    totalDurationMs: Math.round(num(run.duration_s) * 1000),
    totalTokens: usage.input_tokens + usage.output_tokens + (usage.cache_creation_input_tokens ?? 0) + (usage.cache_read_input_tokens ?? 0),
    usage,
    canContinueAgent: true,
  }
}

/**
 * Run `harness <argv>` and read the run off its stdout (one JSON line with `state` on exits
 * 0/1/3, nothing on exit 4). `$.process.spawn` is a stream with no timeout, which a ~1000 s run needs.
 * Returns the run record, or a deny.
 */
async function runHarness($: Dollar, argv: string[], what: string): Promise<{ run: Record<string, unknown> } | { deny: string }> {
  let stdout = ''
  let stderr = ''

  for await (const chunk of $.process.spawn({ argv: [binaryOf($.plugin.root), ...argv] })) {
    if (chunk.stream === 'stderr') {
      stderr += chunk.text ?? ''
    } else if (chunk.stream === 'stdout') {
      stdout += chunk.text ?? ''
    }
  }

  const last = stdout.split('\n').filter(line => line.trim() !== '').pop()
  let run: Record<string, unknown> | undefined

  try {
    run = last === undefined ? undefined : (JSON.parse(last) as Record<string, unknown>)
  } catch {
    run = undefined
  }

  if (run === undefined || typeof run.state !== 'string') {
    const err = stderr.trim()

    return fail(`${BUILTIN_RESULT_NOTE}: ${err !== '' ? err : `harness ${what} printed no run`}`)
  }

  if (run.state !== 'COMPLETED') {
    return fail(`harness run ${String(run.run_id)} ended ${run.state}: ${typeof run.text === 'string' ? run.text : ''}`)
  }

  return { run }
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

      const done = await runHarness($, argv, 'run-agent')

      return 'deny' in done ? done : { result: agentOutput(done.run, prompt) }
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

      const done = await runHarness($, ['send-message', '--to', to, '--message', message], 'send-message')

      return 'deny' in done ? done : { result: { success: true, message: typeof done.run.text === 'string' ? done.run.text : '' } }
    } catch (error) {
      return fail(`${BUILTIN_RESULT_NOTE}: ${error instanceof Error ? error.message : String(error)}`)
    }
  })
}
