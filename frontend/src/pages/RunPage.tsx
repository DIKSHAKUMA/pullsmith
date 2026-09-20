/**
 * Live run view.
 *
 * State is loaded from the server and then followed over SSE, so a hard refresh
 * mid-run rebuilds the full timeline instead of losing it.
 */

import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'

import { StateBadge } from '@/components/StateBadge'
import { Timeline } from '@/components/Timeline'
import { useRunStore } from '@/stores/runStore'
import { TERMINAL_STATES } from '@/lib/types'

const CONNECTION_LABEL: Record<string, string> = {
  idle: 'idle',
  loading: 'loading…',
  live: 'live',
  reconnecting: 'reconnecting…',
  closed: 'stream closed',
  error: 'stream error',
}

function useElapsed(startedAt: string | null, finishedAt: string | null): string {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    if (!startedAt || finishedAt) return
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [startedAt, finishedAt])

  if (!startedAt) return '—'

  const end = finishedAt ? new Date(finishedAt).getTime() : now
  const seconds = Math.max(0, Math.round((end - new Date(startedAt).getTime()) / 1000))

  return seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`
}

export function RunPage() {
  const { runId = '' } = useParams()

  const { run, events, connection, error, watch, reset, cancel } = useRunStore()
  const [actionError, setActionError] = useState<string | null>(null)

  useEffect(() => {
    let close: (() => void) | undefined
    let cancelled = false

    reset()

    watch(runId).then((closer) => {
      // The component may unmount before watch() resolves.
      if (cancelled) closer()
      else close = closer
    })

    return () => {
      cancelled = true
      close?.()
    }
  }, [runId, watch, reset])

  const elapsed = useElapsed(run?.started_at ?? null, run?.finished_at ?? null)
  const isActive = run ? !TERMINAL_STATES.includes(run.state) : false

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="mono text-lg font-semibold text-neutral-100">run {runId.slice(0, 8)}</h1>

        {run && <StateBadge state={run.state} />}

        <span className="mono text-xs text-neutral-500">{CONNECTION_LABEL[connection]}</span>

        {run && isActive && (
          <button
            type="button"
            onClick={() =>
              cancel(runId).catch((cause: Error) => setActionError(cause.message))
            }
            className="mono ml-auto rounded border border-neutral-700 px-3 py-1.5 text-xs text-neutral-300 hover:bg-neutral-900"
          >
            cancel run
          </button>
        )}
      </div>

      {(error ?? actionError) && (
        <p className="rounded border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
          {error ?? actionError}
        </p>
      )}

      {run?.awaiting_human && (
        <p className="rounded border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-200">
          Waiting for a human decision. The agent will not continue until it is reviewed.
        </p>
      )}

      {run?.failure_category && (
        <p className="mono rounded border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-300">
          {run.failure_category}
        </p>
      )}

      {run && (
        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[
            { label: 'Elapsed', value: elapsed },
            { label: 'Iteration', value: `${run.iteration}/${run.max_iterations}` },
            { label: 'Tool calls', value: run.tool_call_count },
            { label: 'Events', value: events.length },
          ].map((stat) => (
            <div key={stat.label} className="rounded border border-neutral-800 px-3 py-2">
              <dt className="text-xs text-neutral-500">{stat.label}</dt>
              <dd className="mono mt-0.5 text-sm text-neutral-100">{stat.value}</dd>
            </div>
          ))}
        </dl>
      )}

      <section className="space-y-3">
        <h2 className="text-sm font-medium text-neutral-300">Execution timeline</h2>
        <Timeline events={events} />
      </section>
    </div>
  )
}
