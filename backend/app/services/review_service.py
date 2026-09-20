"""Review decisions and the push gate.

The rule this module exists to enforce:

> **No push and no pull request without a stored approval row that matches the exact diff
> being pushed.**

Enforced here, in the service layer, not in the UI. A UI-only guard is bypassed by one `curl`
command. Two conditions are checked before any push:

1. an `Approval` row exists for the run with `decision = "approved"`;
2. its `diff_hash` equals the hash of the diff about to be pushed.

The second condition is what stops an approval being reused. If the workspace changed after a
human looked at it, the hash differs and the push is refused — so "approved" can never silently
mean "approved something else".

A blocking risk assessment (a credential in the diff) is refused before a human is even asked.
"""

import hashlib
import logging
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.diff import ChangeSet
from app.agent.risk import RiskAssessment
from app.agent.schemas import ImplementationPlan, VerificationResult
from app.agent.states import FailureCategory, RunState
from app.db.base import new_id, utc_now
from app.models.core import AgentRun
from app.models.review import AgentPlan, Approval, PullRequest, RunDiff
from app.services import event_service, job_queue

logger = logging.getLogger(__name__)


class Decision:
    approved = "approved"
    rejected = "rejected"
    revision_requested = "revision_requested"


class ReviewError(RuntimeError):
    """A review action that is not permitted in the run's current state."""


class ApprovalRequiredError(RuntimeError):
    """Raised when a push is attempted without valid approval. Never caught and retried."""


def hash_diff(diff_text: str) -> str:
    return hashlib.sha256(diff_text.encode("utf-8")).hexdigest()


async def store_plan(
    session: AsyncSession, *, run_id: str, plan: ImplementationPlan
) -> AgentPlan:
    """Saves a plan, versioned so a revision does not overwrite the original."""
    existing = (
        await session.execute(select(AgentPlan).where(AgentPlan.run_id == run_id))
    ).scalars().all()

    row = AgentPlan(
        id=new_id(),
        run_id=run_id,
        version=len(existing) + 1,
        problem_understanding=plan.problem_understanding,
        suspected_root_cause=plan.suspected_root_cause,
        root_cause_confidence=str(plan.root_cause_confidence),
        verification_strategy=plan.verification_strategy,
        payload=plan.model_dump(mode="json"),
    )
    session.add(row)
    await session.flush()

    return row


async def store_diff(
    session: AsyncSession,
    *,
    run_id: str,
    change_set: ChangeSet,
    risk: RiskAssessment,
    verification: VerificationResult | None = None,
) -> RunDiff:
    """Saves the reviewed change set and its risk assessment."""
    diff_text = change_set.unified_diff()

    row = RunDiff(
        id=new_id(),
        run_id=run_id,
        diff_text=diff_text,
        diff_hash=hash_diff(diff_text),
        files_changed=change_set.file_count,
        lines_added=change_set.lines_added,
        lines_removed=change_set.lines_removed,
        risk_level=str(risk.level),
        risk_score=risk.score,
        risk_reasons=risk.reasons,
        risk_warnings=risk.warnings,
        risk_blocking=risk.blocking,
        sensitive_paths=change_set.sensitive_paths,
        verification=verification.model_dump(mode="json") if verification else None,
    )
    session.add(row)
    await session.flush()

    logger.info(
        "diff stored for run %s: %s files, risk=%s, blocking=%s",
        run_id,
        change_set.file_count,
        risk.level,
        risk.blocking,
    )
    return row


async def latest_diff(session: AsyncSession, run_id: str) -> RunDiff | None:
    return (
        await session.execute(
            select(RunDiff)
            .where(RunDiff.run_id == run_id)
            .order_by(RunDiff.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


async def latest_plan(session: AsyncSession, run_id: str) -> AgentPlan | None:
    return (
        await session.execute(
            select(AgentPlan)
            .where(AgentPlan.run_id == run_id)
            .order_by(AgentPlan.version.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


async def get_approval(session: AsyncSession, run_id: str) -> Approval | None:
    return (
        await session.execute(select(Approval).where(Approval.run_id == run_id))
    ).scalar_one_or_none()


async def _decide(
    session: AsyncSession,
    *,
    run: AgentRun,
    user_id: str,
    decision: str,
    comment: str | None,
) -> Approval:
    """Records a decision, refusing a second one for the same run."""
    if await get_approval(session, run.id) is not None:
        # A second decision is a conflict, not an overwrite: the first one may already have
        # authorised a push.
        raise ReviewError("This run has already been decided")

    diff = await latest_diff(session, run.id)

    if diff is None:
        raise ReviewError("There is no diff to review for this run")

    approval = Approval(
        id=new_id(),
        run_id=run.id,
        user_id=user_id,
        decision=decision,
        # Binding to the reviewed content is what makes the approval specific.
        diff_hash=diff.diff_hash,
        comment=comment,
        created_at=utc_now(),
    )
    session.add(approval)
    await session.flush()

    return approval


async def approve(
    session: AsyncSession,
    *,
    run: AgentRun,
    user_id: str,
    comment: str | None = None,
) -> Approval:
    """Approves a run for pull-request creation."""
    if RunState(run.state) is not RunState.WAITING_FOR_APPROVAL:
        raise ReviewError(f"Run is in state {run.state}, not awaiting approval")

    diff = await latest_diff(session, run.id)

    if diff is not None and diff.risk_blocking:
        # Refused before a human is even asked. Approving a diff containing a credential is
        # not a decision anyone should be offered.
        raise ReviewError(
            "This change cannot be approved: a possible credential was detected in the diff"
        )

    approval = await _decide(
        session, run=run, user_id=user_id, decision=Decision.approved, comment=comment
    )

    await event_service.append_event(
        session,
        run.id,
        kind=event_service.EventKind.APPROVAL,
        message="Change approved by reviewer",
        payload={"decision": Decision.approved},
    )

    # The approval row and the job that acts on it are written in one transaction. If they were
    # separate, a crash between them would leave a run approved but never published.
    await job_queue.enqueue(
        session, kind=job_queue.JobKind.CREATE_PULL_REQUEST, run_id=run.id, max_attempts=3
    )

    logger.info("run %s approved by %s, pull request queued", run.id, user_id)
    return approval


async def reject(
    session: AsyncSession,
    *,
    run: AgentRun,
    user_id: str,
    comment: str | None = None,
) -> Approval:
    """Rejects a run. Terminal: the work is discarded."""
    if RunState(run.state) is not RunState.WAITING_FOR_APPROVAL:
        raise ReviewError(f"Run is in state {run.state}, not awaiting approval")

    approval = await _decide(
        session, run=run, user_id=user_id, decision=Decision.rejected, comment=comment
    )

    await event_service.transition(
        session,
        run.id,
        RunState.FAILED,
        message="Rejected by reviewer",
        failure_category=FailureCategory.HUMAN_REJECTION,
        failure_detail=comment,
    )

    logger.info("run %s rejected by %s", run.id, user_id)
    return approval


async def request_revision(
    session: AsyncSession,
    *,
    run: AgentRun,
    user_id: str,
    comment: str,
) -> Approval:
    """Sends the run back for another attempt with reviewer feedback.

    Distinct from rejection: the work is kept and the agent is given direction. Requires a
    comment, because "try again" without saying what was wrong is not feedback.
    """
    if RunState(run.state) is not RunState.WAITING_FOR_APPROVAL:
        raise ReviewError(f"Run is in state {run.state}, not awaiting approval")

    if not comment.strip():
        raise ReviewError("A revision request must explain what needs to change")

    approval = await _decide(
        session,
        run=run,
        user_id=user_id,
        decision=Decision.revision_requested,
        comment=comment,
    )

    await event_service.transition(
        session,
        run.id,
        RunState.PLANNING,
        message="Revision requested by reviewer",
        payload={"comment": comment[:500]},
    )

    # Sending a run back is only meaningful if something picks it up again.
    await job_queue.enqueue(session, kind=job_queue.JobKind.EXECUTE_RUN, run_id=run.id)

    logger.info("run %s sent back for revision by %s", run.id, user_id)
    return approval


async def assert_push_allowed(session: AsyncSession, *, run_id: str, diff_text: str) -> Approval:
    """The gate. Raises unless a valid approval exists for this exact diff.

    Called immediately before any branch push or pull-request creation. Deliberately raises
    rather than returning False: a caller cannot ignore an exception by forgetting to check a
    boolean.
    """
    approval = await get_approval(session, run_id)

    if approval is None:
        raise ApprovalRequiredError("No approval exists for this run")

    if approval.decision != Decision.approved:
        raise ApprovalRequiredError(
            f"Run was not approved (decision was '{approval.decision}')"
        )

    current_hash = hash_diff(diff_text)

    if approval.diff_hash != current_hash:
        # The workspace changed after review. The human approved different content, so this
        # approval does not apply.
        raise ApprovalRequiredError(
            "The change has been modified since it was approved; a new review is required"
        )

    return approval


async def record_pull_request(
    session: AsyncSession,
    *,
    run: AgentRun,
    repository_id: str,
    approval: Approval,
    branch: str,
    title: str,
    number: int | None,
    html_url: str | None,
    commit_sha: str | None,
) -> PullRequest:
    row = PullRequest(
        id=new_id(),
        run_id=run.id,
        repository_id=repository_id,
        approval_id=approval.id,
        branch=branch,
        number=number,
        title=title,
        html_url=html_url,
        commit_sha=commit_sha,
    )
    session.add(row)
    await session.flush()

    await event_service.transition(
        session,
        run.id,
        RunState.PR_CREATED,
        message=f"Pull request opened: {html_url or branch}",
        payload={"branch": branch, "number": number, "url": html_url},
    )

    logger.info("pull request recorded for run %s: %s", run.id, html_url or branch)
    return row


def branch_name(run_id: str, issue_number: int) -> str:
    """A predictable, collision-resistant branch name.

    Includes the run id so two runs on the same issue cannot fight over one branch.
    """
    return f"agent/issue-{issue_number}-{run_id[:8]}"


def pull_request_body(
    *,
    issue_number: int,
    plan: AgentPlan,
    diff: RunDiff,
    test_summary: str,
    iterations: int,
) -> str:
    """The PR description.

    States plainly that this was generated by an agent and reviewed by a human, reports the
    risk assessment as a heuristic, and repeats the reviewer warnings so they survive into the
    place other people will actually read.
    """
    warnings = diff.risk_warnings or []
    warning_block = (
        "\n".join(f"- {warning}" for warning in warnings) if warnings else "- none"
    )

    return f"""\
## Summary

{plan.problem_understanding}

Closes #{issue_number}

## Suspected root cause

{plan.suspected_root_cause}

Confidence: **{plan.root_cause_confidence}**

## Verification

{plan.verification_strategy}

Result: {test_summary}
Attempts required: {iterations}

## Changes

{diff.files_changed} file(s), +{diff.lines_added}/-{diff.lines_removed} lines.

## Risk assessment

**{diff.risk_level}** (heuristic score {diff.risk_score}, for reviewer attention, not a
security guarantee)

{warning_block}

---

Generated by an AI software engineering agent and approved by a human reviewer before this
pull request was opened. The root cause above is the agent's hypothesis; the test result is
the objective evidence.
"""


def now() -> datetime:
    return utc_now()
