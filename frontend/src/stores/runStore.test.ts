/**
 * Event merge tests.
 *
 * These matter because SSE reconnection replays from Last-Event-ID and the history
 * backfill can overlap the live stream. Duplicates and out-of-order frames are the
 * normal case, not the exception.
 */

import { describe, expect, it } from 'vitest'

import { mergeEvent, mergeEvents } from '@/stores/runStore'
import type { RunEvent } from '@/lib/types'

function event(sequence: number, message = `step ${sequence}`): RunEvent {
  return {
    sequence,
    kind: 'state_changed',
    state: 'TESTING',
    message,
    level: 'info',
    payload: null,
    created_at: new Date(1_700_000_000_000 + sequence * 1000).toISOString(),
  }
}

describe('mergeEvent', () => {
  it('appends events arriving in order', () => {
    const result = [event(1), event(2), event(3)].reduce(mergeEvent, [] as RunEvent[])
    expect(result.map((item) => item.sequence)).toEqual([1, 2, 3])
  })

  it('ignores a duplicate sequence', () => {
    const initial = mergeEvents([], [event(1), event(2)])
    const result = mergeEvent(initial, event(2, 'replayed after reconnect'))

    expect(result.map((item) => item.sequence)).toEqual([1, 2])
    expect(result[1]!.message).toBe('step 2')
  })

  it('returns the same array reference for a duplicate so React can skip re-render', () => {
    const initial = mergeEvents([], [event(1)])
    expect(mergeEvent(initial, event(1))).toBe(initial)
  })

  it('sorts an out-of-order arrival back into place', () => {
    const initial = mergeEvents([], [event(1), event(3)])
    const result = mergeEvent(initial, event(2))

    expect(result.map((item) => item.sequence)).toEqual([1, 2, 3])
  })

  it('rebuilds a full timeline when history and live stream overlap', () => {
    // History returns 1..5, then the stream replays 4..7 after a dropped connection.
    const history = mergeEvents([], [1, 2, 3, 4, 5].map((n) => event(n)))
    const result = mergeEvents(history, [4, 5, 6, 7].map((n) => event(n)))

    expect(result.map((item) => item.sequence)).toEqual([1, 2, 3, 4, 5, 6, 7])
  })

  it('handles an empty batch', () => {
    const initial = mergeEvents([], [event(1)])
    expect(mergeEvents(initial, [])).toEqual(initial)
  })
})
