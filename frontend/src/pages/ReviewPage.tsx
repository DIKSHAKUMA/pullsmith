/**
 * The review screen: the only place a change can be authorised.
 *
 * Three principles shape it:
 *
 * 1. Evidence before action. Risk, warnings, plan and diff all appear above the buttons, so a
 *    reviewer has to scroll past the facts to reach the decision.
 * 2. The UI never decides eligibility. `can_approve` comes from the server, because the real
 *    gate lives in the service layer and a second copy of the rule here would eventually
 *    disagree with it.
 * 3. A blocked change offers no approve button at all, and says why.
 */

import { useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'

import { DiffViewer } from '@/components/DiffViewer'
import { RiskBadge } from '@/components/RiskBadge'
import { api, ApiError } from '@/lib/api'
import type { ReviewBundle } from '@/lib/types'

type Action = 'approve' | 'reject' | 'revise'

export function ReviewPage() {
  const { runId = '' } = useParams()

  const [bundle, setBundle] = useState<ReviewBundle | null>(null)
  const [comment, setComment] = useState('')
  const [busy, setBusy] = useState<Action | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    try {
      setBundle(await api.reviewBundle(runId))
    } catch (cause) {
      setError((cause as Error).message)
    } finally {
      setLoading(false)
    }
  }, [runId])

  useEffect(() => {
    void load()
  }, [load])

  async function decide(action: Action) {
    setBusy(action)
    setError(null)

    try {
      if (action === 'approve') await api.approveRun(runId, comment || undefined)
      if (action === 'reject') await api.rejectRun(runId, comment || undefined)
      if (action === 'revise') await api.requestRevision(runId, comment)

      await load()
      setComment('')
    } catch (cause) {
      // 409 means the run moved on or was already decided — a normal outcome, not a bug.
      setError(
        cause instanceof ApiError && cause.status === 409
          ? cause.message
          : (cause as Error).message,
      )
    } finally {
      setBusy(null)
    }
  }

  if (loading) return <p className="text-sm text-neutral-500">Loading review…</p>

  if (!bundle) {
    return <p className="text-sm text-rose-300">{error ?? 'Review not available.'}</p>
  }

  const { plan, diff, approval, can_approve: canApprove } = bundle
  const warnings = diff?.risk_warnings ?? []

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="mono text-lg font-semibold text-neutral-100">
          review {runId.slice(0, 8)}
        </h1>
        {diff && <RiskBadge level={diff.risk_level} score={diff.risk_score} />}
        <span className="mono text-xs text-neutral-500">
          {bundle.iterations_used} attempt(s) · {bundle.state.toLowerCase().replaceAll('_', ' ')}
        </span>
      </div>

      {error && (
        <p className="rounded border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
          {error}
        </p>
      )}

      {diff?.risk_blocking && (
        <div className="rounded border border-rose-500/40 bg-rose-500/10 px-4 py-3">
          <p className="text-sm font-medium text-rose-200">
            This change cannot be approved
          </p>
          <p className="mt-1 text-sm text-rose-300/90">
            A possible credential was detected in the diff. It must be removed before this can
            be reviewed again.
          </p>
        </div>
      )}

      {warnings.length > 0 && (
        <ul className="space-y-1 rounded border border-amber-500/30 bg-amber-500/10 px-4 py-3">
          {warnings.map((warning) => (
            <li key={warning} className="text-sm text-amber-200">
              {warning}
            </li>
          ))}
        </ul>
      )}

      {plan && (
        <section className="space-y-3 rounded border border-neutral-800 p-4">
          <h2 className="text-sm font-medium text-neutral-300">Plan</h2>

          <p className="text-sm text-neutral-200">{plan.problem_understanding}</p>

          <div>
            <p className="text-xs text-neutral-500">
              Suspected root cause ({plan.root_cause_confidence} confidence — the agent's
              hypothesis, not a confirmed finding)
            </p>
            <p className="mt-1 text-sm text-neutral-200">{plan.suspected_root_cause}</p>
          </div>

          <div>
            <p className="text-xs text-neutral-500">Verification</p>
            <p className="mt-1 text-sm text-neutral-200">{plan.verification_strategy}</p>
          </div>
        </section>
      )}

      {diff && (
        <section className="space-y-3">
          <div className="flex items-center gap-3">
            <h2 className="text-sm font-medium text-neutral-300">Changes</h2>
            <span className="mono text-xs text-neutral-500">
              {diff.files_changed} file(s) · +{diff.lines_added}/-{diff.lines_removed}
            </span>
            {diff.sensitive_paths && diff.sensitive_paths.length > 0 && (
              <span className="mono text-xs text-amber-300">
                flagged: {diff.sensitive_paths.join(', ')}
              </span>
            )}
          </div>

          <DiffViewer diffText={diff.diff_text} />
        </section>
      )}

      {approval ? (
        <p className="rounded border border-neutral-800 px-4 py-3 text-sm text-neutral-300">
          Already decided: <span className="mono">{approval.decision}</span>
          {approval.comment ? ` — ${approval.comment}` : ''}
        </p>
      ) : (
        <section className="space-y-3 rounded border border-neutral-800 p-4">
          <h2 className="text-sm font-medium text-neutral-300">Decision</h2>

          <textarea
            value={comment}
            onChange={(event) => setComment(event.target.value)}
            rows={3}
            placeholder="Comment (required when requesting a revision)"
            className="w-full rounded border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-neutral-200 placeholder:text-neutral-600"
          />

          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => decide('approve')}
              disabled={!canApprove || busy !== null}
              className="mono rounded bg-emerald-500/10 px-3 py-1.5 text-xs text-emerald-300 ring-1 ring-inset ring-emerald-500/30 hover:bg-emerald-500/20 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {busy === 'approve' ? 'approving…' : 'approve and open PR'}
            </button>

            <button
              type="button"
              onClick={() => decide('revise')}
              disabled={busy !== null || comment.trim().length < 3}
              className="mono rounded bg-amber-500/10 px-3 py-1.5 text-xs text-amber-300 ring-1 ring-inset ring-amber-500/30 hover:bg-amber-500/20 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {busy === 'revise' ? 'sending…' : 'request revision'}
            </button>

            <button
              type="button"
              onClick={() => decide('reject')}
              disabled={busy !== null}
              className="mono rounded border border-neutral-700 px-3 py-1.5 text-xs text-neutral-300 hover:bg-neutral-900 disabled:opacity-40"
            >
              {busy === 'reject' ? 'rejecting…' : 'reject'}
            </button>
          </div>

          {!canApprove && !diff?.risk_blocking && (
            <p className="text-xs text-neutral-500">
              Approval is unavailable: this run is not awaiting a decision.
            </p>
          )}
        </section>
      )}
    </div>
  )
}
