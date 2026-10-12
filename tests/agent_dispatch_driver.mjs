// #82: runs the REAL hooks/agent_dispatch.ts module under Node against a fake `$` whose
// `process.spawn` mirrors Claude Code's engine contract, and prints what the tool.call
// handler resolved with as one JSON line.
//
// argv: <module.mts path> <tool name> <event JSON>
// env:  HARNESS_DRIVER_CMD  JSON argv prefix replacing the plugin binary, i.e. the
//                           `harness` launcher WITHOUT a subcommand (e.g. [python, -m, harness_plugin]).
//       HARNESS_DRIVER_HOST_ENV  REQUIRED. JSON object: the child's base environment, i.e. what the
//                           engine's HOST process hands a `$.process.spawn` child. The child gets
//                           exactly this plus the spawn request's `env`; the node process's own
//                           environment is never passed on implicitly (#84).
//       HARNESS_DRIVER_SESSION_ID  what the fake `$.session.id()` resolves; it rejects when unset.
//
// Engine contract mirrored (Claude Code 2.1.294): `$.process.spawn({argv, cwd?, env?, input?})`
// takes ONE object with an array `argv` and returns an async stream of
// `{stream: 'stdout'|'stderr', text}` chunks whose return value is `{code, signal}`.
import { spawn } from 'node:child_process'
import { pathToFileURL } from 'node:url'

const [modulePath, toolName, eventJson] = process.argv.slice(2)
const prefix = JSON.parse(process.env.HARNESS_DRIVER_CMD ?? '[]')
if (process.env.HARNESS_DRIVER_HOST_ENV === undefined) {
  throw new Error('HARNESS_DRIVER_HOST_ENV is required: the driver passes no implicit environment to the child')
}
const hostEnv = JSON.parse(process.env.HARNESS_DRIVER_HOST_ENV)
const NEXT_SENTINEL = { passedToNext: true }

function spawnStream(o) {
  // Validated eagerly, like the engine's wrapper: a positional (cmd, args) call throws here.
  if (typeof o !== 'object' || o === null || !Array.isArray(o.argv) || o.argv.length === 0) {
    throw new TypeError('process.spawn expects one object with a non-empty argv array')
  }

  return streamOf(o)
}

async function* streamOf(o) {
  const [cmd, ...args] = [...prefix, ...o.argv.slice(1)]
  const child = spawn(cmd, args, { cwd: o.cwd, env: { ...hostEnv, ...(o.env ?? {}) }, stdio: ['pipe', 'pipe', 'pipe'] })
  const queue = []
  let wake = null
  let open = 2
  const push = item => {
    queue.push(item)
    wake?.()
  }

  for (const name of ['stdout', 'stderr']) {
    child[name].setEncoding('utf8')
    child[name].on('data', text => push({ stream: name, text }))
    child[name].on('end', () => {
      open -= 1
      push(null)
    })
  }

  const exited = new Promise(resolve => child.on('close', (code, signal) => resolve({ code, signal })))
  child.stdin.end(o.input ?? '')

  while (open > 0 || queue.length > 0) {
    if (queue.length === 0) {
      await new Promise(resolve => {
        wake = resolve
      })
      wake = null
      continue
    }

    const item = queue.shift()
    if (item !== null) {
      yield item
    }
  }

  return await exited
}

const $ = {
  process: { spawn: spawnStream },
  session: {
    id: async () => {
      const id = process.env.HARNESS_DRIVER_SESSION_ID
      if (id === undefined) {
        throw new Error('HARNESS_DRIVER_SESSION_ID is not set')
      }
      return id
    },
  },
  plugin: { root: process.env.HARNESS_DRIVER_PLUGIN_ROOT ?? 'C:\dummy\plugin' },
}

const handlers = []
const mod = await import(pathToFileURL(modulePath).href)
mod.register((event, filter, handler) => handlers.push({ event, filter, handler }))

const entry = handlers.find(h => h.event === 'tool.call' && h.filter.tool.includes(toolName))
if (entry === undefined) {
  throw new Error(`no tool.call handler registered for ${toolName}`)
}

const out = await entry.handler($, JSON.parse(eventJson), () => NEXT_SENTINEL)
process.stdout.write(JSON.stringify(out === undefined ? { undefinedResult: true } : out) + '\n')
