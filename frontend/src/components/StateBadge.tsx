import type { RunState } from '@/lib/types'
import { HUMAN_GATE_STATES, TERMINAL_STATES } from '@/lib/types'

const TONE: Record<string, string> = {
  running: 'bg-sky-500/10 text-sky-300 ring-sky-500/30',
  gate: 'bg-amber-500/10 text-amber-300 ring-amber-500/30',
  done: 'bg-emerald-500/10 text-emerald-300 ring-emerald-500/30',
  bad: 'bg-rose-500/10 text-rose-300 ring-rose-500/30',
  idle: 'bg-neutral-500/10 text-neutral-300 ring-neutral-500/30',
}

export function toneFor(state: RunState): keyof typeof TONE {
  if (state === 'COMPLETED' || state === 'PR_CREATED') return 'done'
  if (state === 'FAILED' || state === 'CANCELLED') return 'bad'
  if (HUMAN_GATE_STATES.includes(state)) return 'gate'
  if (state === 'CREATED') return 'idle'
  return 'running'
}

export function StateBadge({ state }: { state: RunState }) {
  const tone = toneFor(state)
  const isActive = !TERMINAL_STATES.includes(state)

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset ${TONE[tone]}`}
    >
      {isActive && tone === 'running' && (
        <span className="size-1.5 animate-pulse rounded-full bg-current" aria-hidden />
      )}
      {state.replaceAll('_', ' ').toLowerCase()}
    </span>
  )
}
