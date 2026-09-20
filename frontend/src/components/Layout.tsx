/**
 * Shell, and the authentication gate.
 *
 * Every route below this needs a session, so the gate lives here rather than being repeated per
 * page: one place decides, and a page added later cannot forget to check.
 *
 * A 401 from /auth/me is the *normal* not-signed-in state, not an error, so it renders the
 * sign-in screen. Any other failure is a real problem and says so instead of silently showing a
 * login form, which would send someone looking for a password when the database is down.
 */

import { NavLink, Outlet } from 'react-router-dom'
import { useCallback, useEffect, useState } from 'react'

import { SignIn } from '@/pages/SignIn'
import { ApiError, api } from '@/lib/api'
import type { Health, User } from '@/lib/types'

const NAV = [
  { to: '/', label: 'Dashboard' },
  { to: '/repositories', label: 'Repositories' },
]

type AuthState =
  | { status: 'loading' }
  | { status: 'anonymous' }
  | { status: 'signed-in'; user: User }
  | { status: 'error'; message: string }

export function Layout() {
  const [health, setHealth] = useState<Health | null>(null)
  const [auth, setAuth] = useState<AuthState>({ status: 'loading' })

  const loadUser = useCallback(async () => {
    try {
      setAuth({ status: 'signed-in', user: await api.me() })
    } catch (caught) {
      if (caught instanceof ApiError && caught.status === 401) {
        setAuth({ status: 'anonymous' })
        return
      }

      setAuth({
        status: 'error',
        message: caught instanceof ApiError ? caught.message : 'Cannot reach the API',
      })
    }
  }, [])

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null))
    void loadUser()
  }, [loadUser])

  async function signOut() {
    await api.logout().catch(() => undefined)
    setAuth({ status: 'anonymous' })
  }

  if (auth.status === 'loading') {
    return <p className="px-6 py-16 text-center text-sm text-neutral-500">Loading…</p>
  }

  if (auth.status === 'error') {
    return (
      <div className="mx-auto max-w-md px-6 py-16">
        <p role="alert" className="rounded border border-red-900/60 bg-red-950/30 p-3 text-sm text-red-200">
          {auth.message}
        </p>
        <button
          type="button"
          onClick={() => void loadUser()}
          className="mt-4 rounded border border-neutral-700 px-3 py-1.5 text-sm text-neutral-200 hover:bg-neutral-900"
        >
          Retry
        </button>
      </div>
    )
  }

  if (auth.status === 'anonymous') {
    return (
      <SignIn health={health} onSignedIn={(user) => setAuth({ status: 'signed-in', user })} />
    )
  }

  return (
    <div className="min-h-dvh">
      <header className="border-b border-neutral-800">
        <div className="mx-auto flex max-w-6xl items-center gap-6 px-6 py-3">
          <span className="mono text-sm font-semibold tracking-tight text-neutral-100">
            Pullsmith
          </span>

          <nav className="flex gap-1">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  `rounded px-2.5 py-1 text-sm ${
                    isActive
                      ? 'bg-neutral-800 text-neutral-100'
                      : 'text-neutral-400 hover:text-neutral-200'
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-4">
            {health && (
              <span className="mono text-xs text-neutral-500">
                {health.app_env} · sandbox: {health.sandbox_backend}
              </span>
            )}

            <span className="mono text-xs text-neutral-300">{auth.user.github_login}</span>

            <button
              type="button"
              onClick={() => void signOut()}
              className="rounded border border-neutral-800 px-2 py-1 text-xs text-neutral-400 hover:bg-neutral-900 hover:text-neutral-200"
            >
              Sign out
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        <Outlet />
      </main>
    </div>
  )
}
