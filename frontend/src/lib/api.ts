/**
 * Typed API client.
 *
 * Paths are relative and proxied by Vite to the backend, so the browser stays on one
 * origin and the SameSite=Lax session cookie is sent with every request.
 */

import type {
  Approval,
  GitHubRepositoryOption,
  Health,
  Issue,
  Repository,
  ReviewBundle,
  Run,
  RunEvent,
  User,
} from './types'

const BASE = '/api'

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly requestId?: string | null,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    // Required so the session cookie travels with the request.
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  })

  if (response.status === 204) {
    return undefined as T
  }

  const text = await response.text()
  const body = text ? JSON.parse(text) : null

  if (!response.ok) {
    // The backend always returns { error: { code, message, request_id } }, so the UI
    // can surface a request id the user can quote when reporting a problem.
    const error = body?.error
    throw new ApiError(
      response.status,
      error?.code ?? `HTTP_${response.status}`,
      error?.message ?? 'Request failed',
      error?.request_id,
    )
  }

  return body as T
}

export const api = {
  health: () => request<Health>('/health'),

  me: () => request<User>('/auth/me'),

  /** The GitHub authorize URL to redirect to. 503s when OAuth is not configured. */
  authStart: () => request<{ authorize_url: string }>('/auth/github/start'),

  /**
   * Signs in with a personal access token. Blocked outside development by the backend.
   *
   * The token is posted once and never held in the browser: the response sets an HttpOnly
   * session cookie, so page JavaScript cannot read the credential afterwards.
   */
  devLogin: (githubToken: string) =>
    request<User>('/auth/dev-login', {
      method: 'POST',
      body: JSON.stringify({ github_token: githubToken }),
    }),

  logout: () => request<void>('/auth/logout', { method: 'POST' }),

  listRepositories: () => request<Repository[]>('/repositories'),

  /** Repositories on GitHub the signed-in user owns, whether linked here or not. */
  listGitHubRepositories: () => request<GitHubRepositoryOption[]>('/github/repositories'),

  linkRepository: (body: { owner: string; name: string }) =>
    request<Repository>('/repositories', { method: 'POST', body: JSON.stringify(body) }),

  getRepository: (id: string) => request<Repository>(`/repositories/${id}`),

  listIssues: (repositoryId: string) =>
    request<Issue[]>(`/repositories/${repositoryId}/issues`),

  syncIssues: (repositoryId: string) =>
    request<{ synced: number; issues: Issue[] }>(
      `/repositories/${repositoryId}/issues/sync`,
      { method: 'POST' },
    ),

  listRuns: (limit = 20) => request<Run[]>(`/runs?limit=${limit}`),

  getRun: (id: string) => request<Run>(`/runs/${id}`),

  createRun: (body: { repository_id: string; issue_id: string; require_plan_approval: boolean }) =>
    request<{ run_id: string; state: string }>('/runs', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  cancelRun: (id: string) => request<Run>(`/runs/${id}/cancel`, { method: 'POST' }),

  /** Events already recorded. Used to backfill before opening the live stream. */
  eventHistory: (runId: string, after = 0) =>
    request<RunEvent[]>(`/runs/${runId}/events/history?after=${after}`),

  /** Plan, diff, risk and any existing decision, in one round trip. */
  reviewBundle: (runId: string) => request<ReviewBundle>(`/runs/${runId}/review`),

  approveRun: (runId: string, comment?: string) =>
    request<Approval>(`/runs/${runId}/approve`, {
      method: 'POST',
      body: JSON.stringify({ comment: comment ?? null }),
    }),

  rejectRun: (runId: string, comment?: string) =>
    request<Approval>(`/runs/${runId}/reject`, {
      method: 'POST',
      body: JSON.stringify({ comment: comment ?? null }),
    }),

  requestRevision: (runId: string, comment: string) =>
    request<Approval>(`/runs/${runId}/request-revision`, {
      method: 'POST',
      body: JSON.stringify({ comment }),
    }),

  eventStreamUrl: (runId: string, after = 0) => `${BASE}/runs/${runId}/events?after=${after}`,
}
