"""Review endpoints: see the plan and diff, then approve, reject or send back.

Every write endpoint checks ownership through `run_manager.get_run`, which filters by user id in
the query. The approval gate itself lives in `review_service`, not here — a route is the wrong
place for a security invariant, because a second route added later would have to remember to
repeat it.
"""

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.api.deps import CurrentUser, DbDep
from app.models.core import AgentRun
from app.schemas.api import (
    ApprovalResponse,
    PlanResponse,
    ReviewBundleResponse,
    RunDiffResponse,
)
from app.services import review_service, run_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["review"])


class DecisionRequest(BaseModel):
    comment: str | None = Field(default=None, max_length=2000)


class RevisionRequest(BaseModel):
    #: Required, unlike the optional comment on approve/reject. "Try again" without saying
    #: what was wrong is not feedback.
    comment: str = Field(min_length=3, max_length=2000)


async def _owned_run(session: DbDep, user: CurrentUser, run_id: str) -> AgentRun:
    try:
        return await run_manager.get_run(session, run_id=run_id, user_id=user.id)
    except run_manager.NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{run_id}/plan", response_model=PlanResponse)
async def get_plan(run_id: str, user: CurrentUser, session: DbDep) -> PlanResponse:
    await _owned_run(session, user, run_id)

    plan = await review_service.latest_plan(session, run_id)

    if plan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No plan has been produced yet"
        )

    return PlanResponse.model_validate(plan)


@router.get("/{run_id}/diff", response_model=RunDiffResponse)
async def get_diff(run_id: str, user: CurrentUser, session: DbDep) -> RunDiffResponse:
    await _owned_run(session, user, run_id)

    diff = await review_service.latest_diff(session, run_id)

    if diff is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No diff has been produced yet"
        )

    return RunDiffResponse.model_validate(diff)


@router.get("/{run_id}/review", response_model=ReviewBundleResponse)
async def get_review_bundle(
    run_id: str, user: CurrentUser, session: DbDep
) -> ReviewBundleResponse:
    """Everything the review screen needs, in one request.

    One round trip rather than four: on a slow connection, a reviewer should not watch four
    spinners resolve in an unpredictable order.
    """
    run = await _owned_run(session, user, run_id)

    plan = await review_service.latest_plan(session, run_id)
    diff = await review_service.latest_diff(session, run_id)
    approval = await review_service.get_approval(session, run_id)

    return ReviewBundleResponse(
        run_id=run.id,
        state=run.state,
        awaiting_approval=run_manager.is_awaiting_human(run),
        iterations_used=run.iteration,
        plan=PlanResponse.model_validate(plan) if plan else None,
        diff=RunDiffResponse.model_validate(diff) if diff else None,
        approval=ApprovalResponse.model_validate(approval) if approval else None,
        # Spelled out rather than relying on `bool(diff) and not diff.risk_blocking`
        # short-circuiting: the conditions are the approval rules, and they are worth reading
        # as three separate statements.
        can_approve=diff is not None and not diff.risk_blocking and approval is None,
    )


@router.post("/{run_id}/approve", response_model=ApprovalResponse)
async def approve_run(
    run_id: str, body: DecisionRequest, user: CurrentUser, session: DbDep
) -> ApprovalResponse:
    """Records approval. Does **not** create the pull request.

    Kept separate so the push happens in the worker: opening a pull request involves several
    GitHub calls and should not block an HTTP request or be lost if one times out.
    """
    run = await _owned_run(session, user, run_id)

    try:
        approval = await review_service.approve(
            session, run=run, user_id=user.id, comment=body.comment
        )
    except review_service.ReviewError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return ApprovalResponse.model_validate(approval)


@router.post("/{run_id}/reject", response_model=ApprovalResponse)
async def reject_run(
    run_id: str, body: DecisionRequest, user: CurrentUser, session: DbDep
) -> ApprovalResponse:
    run = await _owned_run(session, user, run_id)

    try:
        approval = await review_service.reject(
            session, run=run, user_id=user.id, comment=body.comment
        )
    except review_service.ReviewError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return ApprovalResponse.model_validate(approval)


@router.post("/{run_id}/request-revision", response_model=ApprovalResponse)
async def request_revision(
    run_id: str, body: RevisionRequest, user: CurrentUser, session: DbDep
) -> ApprovalResponse:
    run = await _owned_run(session, user, run_id)

    try:
        approval = await review_service.request_revision(
            session, run=run, user_id=user.id, comment=body.comment
        )
    except review_service.ReviewError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return ApprovalResponse.model_validate(approval)
