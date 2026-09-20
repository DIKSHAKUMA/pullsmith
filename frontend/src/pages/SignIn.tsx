/**
 * Sign-in screen.
 *
 * Only offers methods the server says it can honour, read from /health. Rendering a
 * "Continue with GitHub" button when no OAuth app is configured would produce a 503 and leave
 * the user guessing, and offering the token form in production would suggest a route the backend
 * deliberately 404s.
 *
 * The token is posted once and never stored in the browser. The response sets an HttpOnly
 * session cookie, so page JavaScript cannot read the credential afterwards, which is the whole
 * reason the session is a cookie rather than a token in localStorage.
 */

import { useState } from 'react'

import { ApiError, api } from '@/lib/api'
import type { Health, User } from '@/lib/types'

interface Props {
  health: Health | null
  onSignedIn: (user: User) => void
}

export function SignIn({ health, onSignedIn }: Props) {
  const [token, setToken] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function signInWithGitHub() {
    setError(null)
    setBusy(true)

    try {
      const { authorize_url } = await api.authStart()
      window.location.href = authorize_url
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Could not start GitHub sign-in')
      setBusy(false)
    }
  }

  async function signInWithToken(event: React.FormEvent) {
    event.preventDefault()
    setError(null)
    setBusy(true)

    try {
      onSignedIn(await api.devLogin(token.trim()))
    } catch (caught) {
      setError(
        caught instanceof ApiError
          ? caught.message
          : 'Sign-in failed. Check the token and try again.',
      )
      setBusy(false)
    }
  }

  const nothingAvailable = health && !health.github_oauth_configured && !health.dev_login_available

  return (
    <div className="mx-auto max-w-md px-6 py-16">
      <h1 className="mono text-lg font-semibold text-neutral-100">Pullsmith</h1>
      <p className="mt-2 text-sm text-neutral-400">
        Sign in with GitHub to pick a repository and an issue for the agent to work on.
      </p>

      {health?.github_oauth_configured && (
        <button
          type="button"
          onClick={signInWithGitHub}
          disabled={busy}
          className="mt-8 w-full rounded bg-neutral-100 px-4 py-2 text-sm font-medium text-neutral-900 hover:bg-white disabled:opacity-50"
        >
          Continue with GitHub
        </button>
      )}

      {health?.dev_login_available && (
        <form onSubmit={signInWithToken} className="mt-8">
          {health.github_oauth_configured && (
            <div className="mb-6 flex items-center gap-3 text-xs text-neutral-600">
              <span className="h-px flex-1 bg-neutral-800" />
              or
              <span className="h-px flex-1 bg-neutral-800" />
            </div>
          )}

          <label htmlFor="token" className="block text-sm text-neutral-300">
            Personal access token
          </label>

          <input
            id="token"
            type="password"
            autoComplete="off"
            value={token}
            onChange={(event) => setToken(event.target.value)}
            placeholder="github_pat_..."
            aria-describedby="token-help"
            className="mono mt-2 w-full rounded border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-neutral-100 placeholder:text-neutral-600 focus:border-neutral-500 focus:outline-none"
          />

          <p id="token-help" className="mt-2 text-xs text-neutral-500">
            Needs Contents, Issues and Pull requests set to read and write on the repositories you
            want the agent to touch. Development only; this form is unavailable in production.
          </p>

          <button
            type="submit"
            disabled={busy || token.trim().length < 8}
            className="mt-4 w-full rounded border border-neutral-700 px-4 py-2 text-sm text-neutral-100 hover:bg-neutral-900 disabled:opacity-40"
          >
            {busy ? 'Signing in…' : 'Sign in with token'}
          </button>
        </form>
      )}

      {nothingAvailable && (
        <p className="mt-8 rounded border border-amber-900/60 bg-amber-950/30 p-3 text-sm text-amber-200">
          No sign-in method is configured. Set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET, or run
          with APP_ENV=development to enable token sign-in.
        </p>
      )}

      {error && (
        <p role="alert" className="mt-4 rounded border border-red-900/60 bg-red-950/30 p-3 text-sm text-red-200">
          {error}
        </p>
      )}
    </div>
  )
}
