/**
 * Operational run timeline.
 *
 * Shows only what the backend recorded as operational events. The agent's reasoning is
 * never streamed or rendered; the user sees what the system did, not how a model
 * deliberated.
 */

import type { RunEvent } from '@/lib/types'

const LEVEL_MARK: Record<string, { symbol: string; className: string }> = {
  info: { symbol: '✓', className: 'text-emerald-400' },
  warning: { symbol: '!', className: 'text-amber-400' },
  error: { symbol: '✗', className: 'text-rose-400' },
}

function timeOf(iso: string): string {
  return new Date(iso).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

export function Timeline({ events }: { events: RunEvent[] }) {
  if (!events.length) {
    return <p className="text-sm text-neutral-500">No events recorded yet.</p>
  }

  return (
    <ol className="space-y-0">
      {events.map((event, index) => {
        const mark = LEVEL_MARK[event.level] ?? LEVEL_MARK.info!
        const isLast = index === events.length - 1

        return (
          <li key={event.sequence} className="relative flex gap-3 pb-4">
            {!isLast && (
              <span
                className="absolute left-[7px] top-5 h-full w-px bg-neutral-800"
                aria-hidden
              />
            )}

            <span className={`mono relative z-10 select-none text-sm ${mark.className}`}>
              {mark.symbol}
            </span>

            <div className="min-w-0 flex-1">
              <p className="text-sm text-neutral-200">{event.message}</p>

              <p className="mono mt-0.5 text-xs text-neutral-500">
                #{event.sequence} · {timeOf(event.created_at)}
                {event.state ? ` · ${event.state.toLowerCase()}` : ''}
              </p>

              {event.payload && Object.keys(event.payload).length > 0 && (
                <pre className="mono mt-1.5 overflow-x-auto rounded border border-neutral-800 bg-neutral-900/60 p-2 text-xs text-neutral-400">
                  {JSON.stringify(event.payload, null, 2)}
                </pre>
              )}
            </div>
          </li>
        )
      })}
    </ol>
  )
}
