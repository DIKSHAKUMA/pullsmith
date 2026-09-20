import { useCallback, useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'

import { api, ApiError } from '@/lib/api'
import type { Issue, Repository } from '@/lib/types'

export function RepositoryPage() {
  const { repositoryId = '' } = useParams()
  const navigate = useNavigate()

  const [repository, setRepository] = useState<Repository | null>(null)
  const [issues, setIssues] = useState<Issue[]>([])
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    const [repo, issueList] = await Promise.all([
      api.getRepository(repositoryId),
      api.listIssues(repositoryId),
    ])
    setRepository(repo)
    setIssues(issueList)
  }, [repositoryId])

  useEffect(() => {
    load().catch((cause: Error) => setError(cause.message))
  }, [load])

  async function sync() {
    setBusy('sync')
    setError(null)
    try {
      const result = await api.syncIssues(repositoryId)
      setIssues(result.issues)
    } catch (cause) {
      setError((cause as Error).message)
    } finally {
      setBusy(null)
    }
  }

  async function startRun(issueId: string) {
    setBusy(issueId)
    setError(null)
    try {
      const created = await api.createRun({
        repository_id: repositoryId,
        issue_id: issueId,
        require_plan_approval: true,
      })
      navigate(`/runs/${created.run_id}`)
    } catch (cause) {
      // 409 means a run is already active for this issue, which is a normal outcome
      // rather than a bug, so it is surfaced as a message.
      const message =
        cause instanceof ApiError && cause.status === 409
          ? cause.message
          : (cause as Error).message
      setError(message)
    } finally {
      setBusy(null)
    }
  }

  if (error && !repository) {
    return <p className="text-sm text-rose-300">{error}</p>
  }

  return (
    <div className="space-y-6">
      <div className="flex items-start gap-4">
        <div>
          <h1 className="text-xl font-semibold text-neutral-100">
            {repository?.full_name ?? 'Repository'}
          </h1>
          <p className="mono mt-1 text-xs text-neutral-500">
            {repository?.primary_language ?? 'unknown'} · {repository?.default_branch}
          </p>
        </div>

        <button
          type="button"
          onClick={sync}
          disabled={busy === 'sync'}
          className="mono ml-auto rounded border border-neutral-700 px-3 py-1.5 text-xs text-neutral-300 hover:bg-neutral-900 disabled:opacity-50"
        >
          {busy === 'sync' ? 'syncing…' : 'sync issues'}
        </button>
      </div>

      {error && (
        <p className="rounded border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-300">
          {error}
        </p>
      )}

      <section className="space-y-3">
        <h2 className="text-sm font-medium text-neutral-300">Issues</h2>

        {issues.length === 0 ? (
          <p className="text-sm text-neutral-500">No issues stored. Sync from GitHub.</p>
        ) : (
          <ul className="divide-y divide-neutral-800 rounded border border-neutral-800">
            {issues.map((issue) => (
              <li key={issue.id} className="flex items-start gap-4 px-4 py-3">
                <div className="min-w-0">
                  <p className="truncate text-sm text-neutral-200">
                    <span className="mono text-neutral-500">#{issue.number}</span> {issue.title}
                  </p>
                  {issue.labels && issue.labels.length > 0 && (
                    <p className="mono mt-1 text-xs text-neutral-500">
                      {issue.labels.join(' · ')}
                    </p>
                  )}
                </div>

                <button
                  type="button"
                  onClick={() => startRun(issue.id)}
                  disabled={busy === issue.id}
                  className="mono ml-auto shrink-0 rounded bg-sky-500/10 px-3 py-1.5 text-xs text-sky-300 ring-1 ring-inset ring-sky-500/30 hover:bg-sky-500/20 disabled:opacity-50"
                >
                  {busy === issue.id ? 'starting…' : 'start agent'}
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  )
}
