import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

import { StateBadge } from '@/components/StateBadge'
import { api } from '@/lib/api'
import type { Repository, Run } from '@/lib/types'
import { TERMINAL_STATES } from '@/lib/types'

export function Dashboard() {
  const [runs, setRuns] = useState<Run[]>([])
  const [repositories, setRepositories] = useState<Repository[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([api.listRuns(), api.listRepositories()])
      .then(([runList, repoList]) => {
        setRuns(runList)
        setRepositories(repoList)
      })
      .catch((cause: Error) => setError(cause.message))
      .finally(() => setLoading(false))
  }, [])

  const active = runs.filter((run) => !TERMINAL_STATES.includes(run.state)).length

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-xl font-semibold text-neutral-100">Dashboard</h1>
        <p className="mt-1 text-sm text-neutral-500">
          Agent runs across your connected repositories.
        </p>
      </div>

      {error && (
        <p className="rounded border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
          {error}
        </p>
      )}

      <dl className="grid grid-cols-3 gap-3">
        {[
          { label: 'Repositories', value: repositories.length },
          { label: 'Runs', value: runs.length },
          { label: 'Active', value: active },
        ].map((stat) => (
          <div key={stat.label} className="rounded border border-neutral-800 px-4 py-3">
            <dt className="text-xs text-neutral-500">{stat.label}</dt>
            <dd className="mono mt-1 text-2xl text-neutral-100">{stat.value}</dd>
          </div>
        ))}
      </dl>

      <section className="space-y-3">
        <h2 className="text-sm font-medium text-neutral-300">Recent runs</h2>

        {loading ? (
          <p className="text-sm text-neutral-500">Loading…</p>
        ) : runs.length === 0 ? (
          <p className="text-sm text-neutral-500">
            No runs yet. Pick an issue from a repository to start one.
          </p>
        ) : (
          <ul className="divide-y divide-neutral-800 rounded border border-neutral-800">
            {runs.map((run) => (
              <li key={run.id} className="flex items-center gap-4 px-4 py-3">
                <Link
                  to={`/runs/${run.id}`}
                  className="mono truncate text-sm text-sky-300 hover:underline"
                >
                  {run.id.slice(0, 8)}
                </Link>

                <StateBadge state={run.state} />

                <span className="mono ml-auto text-xs text-neutral-500">
                  iter {run.iteration}/{run.max_iterations} · {run.last_event_sequence} events
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="space-y-3">
        <h2 className="text-sm font-medium text-neutral-300">Repositories</h2>

        {repositories.length === 0 ? (
          <p className="text-sm text-neutral-500">No repositories linked yet.</p>
        ) : (
          <ul className="divide-y divide-neutral-800 rounded border border-neutral-800">
            {repositories.map((repo) => (
              <li key={repo.id} className="flex items-center gap-4 px-4 py-3">
                <Link
                  to={`/repositories/${repo.id}`}
                  className="text-sm text-sky-300 hover:underline"
                >
                  {repo.full_name}
                </Link>
                <span className="mono ml-auto text-xs text-neutral-500">
                  {repo.primary_language ?? 'unknown'} · {repo.default_branch}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  )
}
