/** Mirrors the backend response schemas in app/schemas/api.py. */

export const RUN_STATES = [
  'CREATED',
  'CLONING_REPOSITORY',
  'ANALYZING_ISSUE',
  'EXPLORING_REPOSITORY',
  'INDEXING_REPOSITORY',
  'RETRIEVING_CONTEXT',
  'PLANNING',
  'WAITING_FOR_PLAN_REVIEW',
  'IMPLEMENTING',
  'TESTING',
  'ANALYZING_FAILURE',
  'REVISING',
  'VERIFYING',
  'WAITING_FOR_APPROVAL',
  'PR_CREATED',
  'COMPLETED',
  'FAILED',
  'CANCELLED',
] as const

export type RunState = (typeof RUN_STATES)[number]

export const TERMINAL_STATES: RunState[] = ['COMPLETED', 'FAILED', 'CANCELLED']
export const HUMAN_GATE_STATES: RunState[] = ['WAITING_FOR_PLAN_REVIEW', 'WAITING_FOR_APPROVAL']

export interface User {
  id: string
  github_login: string
  display_name: string | null
  avatar_url: string | null
}

/** A repository on GitHub, which may or may not be linked here yet. */
export interface GitHubRepositoryOption {
  github_repo_id: number
  owner: string
  name: string
  full_name: string
  default_branch: string
  primary_language: string | null
  is_private: boolean
}

export interface Health {
  status: string
  database: string
  app_env: string
  sandbox_backend: string

  /** Which sign-in methods the server can actually honour, so the UI offers only those. */
  github_oauth_configured: boolean
  dev_login_available: boolean

  /** Queue depth, and how long the oldest job has waited. Used to warn that no worker is running. */
  queued_jobs: number
  oldest_queued_job_seconds: number | null
}

/** Past this, a queued job almost certainly means nothing is consuming the queue. */
export const WORKER_STALL_SECONDS = 45

export interface Repository {
  id: string
  owner: string
  name: string
  full_name: string
  default_branch: string
  primary_language: string | null
  is_private: boolean
}

export interface Issue {
  id: string
  number: number
  title: string
  body: string | null
  state: string
  labels: string[] | null
  html_url: string | null
}

export interface Run {
  id: string
  repository_id: string
  issue_id: string
  state: RunState
  failure_category: string | null
  iteration: number
  max_iterations: number
  tool_call_count: number
  require_plan_approval: boolean
  awaiting_human: boolean
  started_at: string | null
  finished_at: string | null
  created_at: string
  last_event_sequence: number
}

export interface RunEvent {
  sequence: number
  kind: string
  state: RunState | null
  message: string
  level: 'info' | 'warning' | 'error'
  payload: Record<string, unknown> | null
  created_at: string
}

export interface ApiErrorBody {
  code: string
  message: string
  request_id?: string | null
}


export interface Plan {
  id: string
  version: number
  problem_understanding: string
  suspected_root_cause: string
  root_cause_confidence: string
  verification_strategy: string
  payload: Record<string, unknown>
}

export interface RunDiff {
  id: string
  diff_text: string
  diff_hash: string
  files_changed: number
  lines_added: number
  lines_removed: number
  risk_level: string
  risk_score: number
  risk_reasons: string[] | null
  risk_warnings: string[] | null
  risk_blocking: boolean
  sensitive_paths: string[] | null
  verification: Record<string, unknown> | null
}

export interface Approval {
  id: string
  decision: 'approved' | 'rejected' | 'revision_requested'
  diff_hash: string
  comment: string | null
  created_at: string
}

export interface ReviewBundle {
  run_id: string
  state: RunState
  awaiting_approval: boolean
  iterations_used: number
  plan: Plan | null
  diff: RunDiff | null
  approval: Approval | null

  /** Comes from the server. The UI must not re-derive the approval rule. */
  can_approve: boolean
}
