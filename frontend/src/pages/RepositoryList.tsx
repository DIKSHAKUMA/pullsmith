/**
 * Linked repositories, and the way to link a new one.
 *
 * The browse-and-link step used to be missing: the empty state said "connect GitHub, then link a
 * repository" while offering no way to do either, so a new user reached a dead end. Linking is a
 * deliberate, per-repository action rather than importing everything the token can see, because
 * the agent writes to what it is pointed at.
 */

import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

import { ApiError, api } from '@/lib/api'
import type { GitHubRepositoryOption, Repository } from '@/lib/types'

export function RepositoryList() {
  const [repositories, setRepositories] = useState<Repository[]>([])
  const [options, setOptions] = useState<GitHubRepositoryOption[] | null>(null)
  const [login, setLogin] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      setRepositories(await api.listRepositories())
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Could not load repositories')
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  async function browseGitHub() {
    setError(null)
    setBusy('browse')

    try {
      // The login is needed to tell apart repositories the user owns from ones they only have
      // write access to. Pointing the agent at somebody else's project is a bigger decision.
      const [available, user] = await Promise.all([
        api.listGitHubRepositories(),
        api.me(),
      ])

      setLogin(user.github_login)
      setOptions(available)
    } catch (caught) {
      setError(
        caught instanceof ApiError
          ? `${caught.message} (the token may lack repository access)`
          : 'Could not reach GitHub',
      )
    } finally {
      setBusy(null)
    }
  }

  async function link(option: GitHubRepositoryOption) {
    setError(null)
    setBusy(option.full_name)

    try {
      await api.linkRepository({ owner: option.owner, name: option.name })
      await load()
      setOptions((current) =>
        current?.filter((item) => item.full_name !== option.full_name) ?? null,
      )
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Could not link that repository')
    } finally {
      setBusy(null)
    }
  }

  const linkedNames = new Set(repositories.map((item) => item.full_name))

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-neutral-100">Repositories</h1>

        <button
          type="button"
          onClick={() => void browseGitHub()}
          disabled={busy === 'browse'}
          className="rounded border border-neutral-700 px-3 py-1.5 text-sm text-neutral-100 hover:bg-neutral-900 disabled:opacity-50"
        >
          {busy === 'browse' ? 'Loading…' : 'Link from GitHub'}
        </button>
      </div>

      {error && (
        <p role="alert" className="rounded border border-red-900/60 bg-red-950/30 p-3 text-sm text-red-200">
          {error}
        </p>
      )}

      {repositories.length === 0 ? (
        <p className="text-sm text-neutral-500">
          Nothing linked yet. Use <span className="text-neutral-300">Link from GitHub</span> to
          pick a repository the agent may work on.
        </p>
      ) : (
        <ul className="divide-y divide-neutral-800 rounded border border-neutral-800">
          {repositories.map((repo) => (
            <li key={repo.id} className="px-4 py-3">
              <Link
                to={`/repositories/${repo.id}`}
                className="text-sm text-sky-300 hover:underline"
              >
                {repo.full_name}
              </Link>
              <p className="mono mt-0.5 text-xs text-neutral-500">
                {repo.primary_language ?? 'unknown'} · {repo.default_branch}
                {repo.is_private ? ' · private' : ''}
              </p>
            </li>
          ))}
        </ul>
      )}

      {options && (
        <section className="space-y-3">
          <h2 className="text-sm font-medium text-neutral-300">Available on GitHub</h2>

          {options.length === 0 ? (
            <p className="text-sm text-neutral-500">
              No repositories came back. The token needs repository access to list them.
            </p>
          ) : (
            <ul className="divide-y divide-neutral-800 rounded border border-neutral-800">
              {options.map((option) => (
                <li
                  key={option.github_repo_id}
                  className="flex items-center gap-4 px-4 py-2.5"
                >
                  <div className="min-w-0">
                    <p className="mono truncate text-sm text-neutral-200">{option.full_name}</p>
                    <p className="mono text-xs text-neutral-500">
                      {option.primary_language ?? 'unknown'} · {option.default_branch}
                      {option.is_private ? ' · private' : ''}
                    </p>
                  </div>

                  <span className="ml-auto flex shrink-0 gap-3 text-xs">
                    {option.is_private && (
                      <span
                        className="text-amber-300"
                        title="The agent clones without credentials, so private repositories cannot be cloned yet"
                      >
                        private: cannot be cloned yet
                      </span>
                    )}

                    {login && option.owner !== login && (
                      <span
                        className="text-sky-300"
                        title="You have write access but do not own this repository"
                      >
                        owned by {option.owner}
                      </span>
                    )}
                  </span>

                  <button
                    type="button"
                    onClick={() => void link(option)}
                    disabled={busy === option.full_name || linkedNames.has(option.full_name)}
                    className="shrink-0 rounded border border-neutral-700 px-2.5 py-1 text-xs text-neutral-100 hover:bg-neutral-900 disabled:opacity-40"
                  >
                    {linkedNames.has(option.full_name)
                      ? 'Linked'
                      : busy === option.full_name
                        ? 'Linking…'
                        : 'Link'}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}
    </div>
  )
}
