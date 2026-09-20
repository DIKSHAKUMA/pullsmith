/**
 * Server-Sent Events client.
 *
 * SSE is used instead of WebSocket because the traffic is one-directional: the server
 * pushes run events, the client never pushes back. SSE also reconnects natively and
 * resends Last-Event-ID, which is exactly the resume semantics the backend implements.
 */

import type { RunEvent } from './types'

export type StreamStatus = 'live' | 'reconnecting' | 'closed' | 'error'

interface StreamOptions {
  url: string
  onEvent: (event: RunEvent) => void
  onStatus?: (status: StreamStatus) => void
  onFinished?: () => void
}

export function openEventStream({
  url,
  onEvent,
  onStatus,
  onFinished,
}: StreamOptions): () => void {
  // withCredentials keeps the session cookie attached on reconnects too.
  const source = new EventSource(url, { withCredentials: true })
  let closed = false

  source.addEventListener('open', () => onStatus?.('live'))

  source.addEventListener('run_event', (message) => {
    try {
      onEvent(JSON.parse((message as MessageEvent<string>).data) as RunEvent)
    } catch {
      // A malformed frame must not kill the stream; the next frame may be fine.
    }
  })

  source.addEventListener('run_finished', () => {
    closed = true
    source.close()
    onFinished?.()
  })

  source.addEventListener('error', () => {
    if (closed) return

    // EventSource reconnects on its own and replays Last-Event-ID, so this is a
    // status change rather than a failure to handle.
    onStatus?.(source.readyState === EventSource.CLOSED ? 'error' : 'reconnecting')
  })

  return () => {
    closed = true
    source.close()
    onStatus?.('closed')
  }
}
