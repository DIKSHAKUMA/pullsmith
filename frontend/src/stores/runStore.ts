/**
 * Live run state.
 *
 * The server is the source of truth. This store only mirrors it, which is why a hard
 * refresh mid-run can rebuild the whole timeline: history is fetched, then the live
 * stream resumes from the highest sequence already held.
 */

import { create } from 'zustand'

import { api } from '@/lib/api'
import { openEventStream } from '@/lib/sse'
import type { Run, RunEvent } from '@/lib/types'
import { TERMINAL_STATES } from '@/lib/types'

interface RunStoreState {
  run: Run | null
  events: RunEvent[]
  connection: 'idle' | 'loading' | 'live' | 'reconnecting' | 'closed' | 'error'
  error: string | null

  /** Highest sequence held. This is the cursor the stream resumes from. */
  lastSequence: () => number

  applyEvent: (event: RunEvent) => void
  applyEvents: (events: RunEvent[]) => void
  reset: () => void
  watch: (runId: string) => Promise<() => void>
  refreshRun: (runId: string) => Promise<void>
  cancel: (runId: string) => Promise<void>
}

/**
 * Inserts an event in sequence order, ignoring duplicates.
 *
 * Duplicates are expected, not exceptional: after a dropped connection the browser
 * replays from Last-Event-ID, and the history backfill can overlap the live stream.
 */
export function mergeEvent(events: RunEvent[], incoming: RunEvent): RunEvent[] {
  if (events.some((event) => event.sequence === incoming.sequence)) {
    return events
  }

  const last = events[events.length - 1]

  // Fast path: events almost always arrive in order.
  if (!last || incoming.sequence > last.sequence) {
    return [...events, incoming]
  }

  const merged = [...events, incoming]
  merged.sort((a, b) => a.sequence - b.sequence)
  return merged
}

export function mergeEvents(events: RunEvent[], incoming: RunEvent[]): RunEvent[] {
  return incoming.reduce(mergeEvent, events)
}

export const useRunStore = create<RunStoreState>((set, get) => ({
  run: null,
  events: [],
  connection: 'idle',
  error: null,

  lastSequence: () => {
    const { events } = get()
    return events.length ? events[events.length - 1]!.sequence : 0
  },

  applyEvent: (event) =>
    set((state) => ({
      events: mergeEvent(state.events, event),
      run: state.run && event.state ? { ...state.run, state: event.state } : state.run,
    })),

  applyEvents: (incoming) =>
    set((state) => ({ events: mergeEvents(state.events, incoming) })),

  reset: () => set({ run: null, events: [], connection: 'idle', error: null }),

  refreshRun: async (runId) => {
    const run = await api.getRun(runId)
    set({ run })
  },

  cancel: async (runId) => {
    const run = await api.cancelRun(runId)
    set({ run })
  },

  /**
   * Loads the run, backfills recorded events, then opens the live stream.
   *
   * Order matters: backfilling first means the stream can start from a known cursor,
   * so no event is missed between the two calls.
   */
  watch: async (runId) => {
    set({ connection: 'loading', error: null })

    try {
      const [run, history] = await Promise.all([
        api.getRun(runId),
        api.eventHistory(runId, 0),
      ])

      set({ run, events: mergeEvents([], history) })

      if (TERMINAL_STATES.includes(run.state)) {
        // Nothing more will happen; opening a stream would just burn a connection.
        set({ connection: 'closed' })
        return () => {}
      }

      const close = openEventStream({
        url: api.eventStreamUrl(runId, get().lastSequence()),
        onEvent: (event) => get().applyEvent(event),
        onStatus: (connection) => set({ connection }),
        onFinished: () => {
          set({ connection: 'closed' })
          void get().refreshRun(runId)
        },
      })

      return close
    } catch (error) {
      set({ connection: 'error', error: (error as Error).message })
      return () => {}
    }
  },
}))
