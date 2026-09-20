"""Run lifecycle service.

The API never executes an agent run. It validates the request, creates the run row,
enqueues a job and returns 202. All long work happens in the worker process, which is
what keeps the API responsive on a two-core machine and what makes a run survive an
API restart.
"""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.states import FailureCategory, RunState, awaits_human, is_terminal
from app.db.base import new_id
from app.models.core import AgentRun, Issue, Repository
from app.services import event_service, job_queue

logger = logging.getLogger(__name__)


class RunConflictError(RuntimeError):
    """Raised when a run cannot be created or acted on in its current state."""


class NotFoundError(LookupError):
    """Raised when a resource does not exist or does not belong to the caller."""


async def create_run(
    session: AsyncSession,
    *,
    user_id: str,
    repository_id: str,
    issue_id: str,
    max_iterations: int,
    require_plan_approval: bool = True,
) -> AgentRun:
    repository = (
        await session.execute(
            select(Repository).where(
                Repository.id == repository_id,
                # Ownership is checked in the query itself. Filtering by id alone and
                # then comparing in Python is how cross-tenant leaks happen.
                Repository.user_id == user_id,
            )
        )
    ).scalar_one_or_none()

    if repository is None:
        raise NotFoundError("Repository not found")

    issue = (
        await session.execute(
            select(Issue).where(Issue.id == issue_id, Issue.repository_id == repository_id)
        )
    ).scalar_one_or_none()

    if issue is None:
        raise NotFoundError("Issue not found for this repository")

    active = (
        await session.execute(
            select(AgentRun).where(
                AgentRun.issue_id == issue_id,
                AgentRun.state.notin_(
                    [str(RunState.COMPLETED), str(RunState.FAILED), str(RunState.CANCELLED)]
                ),
            )
        )
    ).scalars().first()

    if active is not None:
        raise RunConflictError(f"Run {active.id} is already active for this issue")

    run = AgentRun(
        id=new_id(),
        user_id=user_id,
        repository_id=repository_id,
        issue_id=issue_id,
        state=str(RunState.CREATED),
        max_iterations=max_iterations,
        require_plan_approval=require_plan_approval,
    )
    session.add(run)
    await session.flush()

    await event_service.append_event(
        session,
        run.id,
        kind=event_service.EventKind.INFO,
        message=f"Run queued for issue #{issue.number}",
        payload={"repository": repository.full_name, "issue_number": issue.number},
    )

    await job_queue.enqueue(session, kind=job_queue.JobKind.EXECUTE_RUN, run_id=run.id)

    logger.info("created run %s for issue %s", run.id, issue.id)
    return run


async def get_run(session: AsyncSession, *, run_id: str, user_id: str) -> AgentRun:
    run = (
        await session.execute(
            select(AgentRun).where(AgentRun.id == run_id, AgentRun.user_id == user_id)
        )
    ).scalar_one_or_none()

    if run is None:
        raise NotFoundError("Run not found")

    return run


async def list_runs(
    session: AsyncSession, *, user_id: str, limit: int = 20, offset: int = 0
) -> list[AgentRun]:
    result = await session.execute(
        select(AgentRun)
        .where(AgentRun.user_id == user_id)
        .order_by(AgentRun.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars())


async def cancel_run(session: AsyncSession, *, run_id: str, user_id: str) -> AgentRun:
    run = await get_run(session, run_id=run_id, user_id=user_id)

    if is_terminal(RunState(run.state)):
        raise RunConflictError(f"Run already finished in state {run.state}")

    return await event_service.transition(
        session,
        run.id,
        RunState.CANCELLED,
        message="Cancelled by user",
        failure_category=FailureCategory.HUMAN_REJECTION,
    )


def is_awaiting_human(run: AgentRun) -> bool:
    return awaits_human(RunState(run.state))
