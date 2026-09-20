"""Run endpoints, including the resumable SSE event stream."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.agent.states import RunState, is_terminal
from app.api.deps import CurrentUser, DbDep, SettingsDep
from app.db.session import session_scope
from app.models.core import AgentRun
from app.schemas.api import CreateRunRequest, EventResponse, RunCreatedResponse, RunResponse
from app.services import event_service, run_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])


def _to_response(run: AgentRun) -> RunResponse:
    payload = RunResponse.model_validate(run)
    payload.awaiting_human = run_manager.is_awaiting_human(run)
    return payload


@router.post("", response_model=RunCreatedResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_run(
    body: CreateRunRequest, user: CurrentUser, session: DbDep, settings: SettingsDep
) -> RunCreatedResponse:
    """Creates a run and returns immediately.

    202 rather than 201 because the meaningful work has only been accepted, not
    completed. The worker executes it.
    """
    try:
        run = await run_manager.create_run(
            session,
            user_id=user.id,
            repository_id=body.repository_id,
            issue_id=body.issue_id,
            max_iterations=body.max_iterations or settings.max_iterations,
            require_plan_approval=body.require_plan_approval,
        )
    except run_manager.NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except run_manager.RunConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return RunCreatedResponse(run_id=run.id, state=run.state)


@router.get("", response_model=list[RunResponse])
async def list_runs(
    user: CurrentUser, session: DbDep, limit: int = 20, offset: int = 0
) -> list[RunResponse]:
    runs = await run_manager.list_runs(session, user_id=user.id, limit=limit, offset=offset)
    return [_to_response(run) for run in runs]


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(run_id: str, user: CurrentUser, session: DbDep) -> RunResponse:
    try:
        run = await run_manager.get_run(session, run_id=run_id, user_id=user.id)
    except run_manager.NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return _to_response(run)


@router.post("/{run_id}/cancel", response_model=RunResponse)
async def cancel_run(run_id: str, user: CurrentUser, session: DbDep) -> RunResponse:
    try:
        run = await run_manager.cancel_run(session, run_id=run_id, user_id=user.id)
    except run_manager.NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except run_manager.RunConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return _to_response(run)


@router.get("/{run_id}/events/history", response_model=list[EventResponse])
async def event_history(
    run_id: str, user: CurrentUser, session: DbDep, after: int = 0
) -> list[EventResponse]:
    try:
        await run_manager.get_run(session, run_id=run_id, user_id=user.id)
    except run_manager.NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    events = await event_service.list_events(session, run_id, after_sequence=after)
    return [EventResponse.model_validate(event) for event in events]


def _sse_frame(event: EventResponse) -> str:
    """Formats one SSE frame.

    The event id is the sequence number, which is what the browser sends back as
    Last-Event-ID after a dropped connection.
    """
    body = json.dumps(
        {
            "sequence": event.sequence,
            "kind": event.kind,
            "state": event.state,
            "message": event.message,
            "level": event.level,
            "payload": event.payload,
            "created_at": event.created_at.isoformat(),
        }
    )
    return f"id: {event.sequence}\nevent: run_event\ndata: {body}\n\n"


@router.get("/{run_id}/events")
async def stream_events(
    run_id: str,
    request: Request,
    user: CurrentUser,
    session: DbDep,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    after: int = 0,
) -> StreamingResponse:
    """Streams run events over SSE, resuming from Last-Event-ID when reconnecting.

    Authorisation is checked before the stream opens. Polling inside the generator uses
    its own short-lived sessions so the request-scoped session is not held open for the
    lifetime of the stream.
    """
    try:
        await run_manager.get_run(session, run_id=run_id, user_id=user.id)
    except run_manager.NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    start_after = after

    if last_event_id and last_event_id.isdigit():
        start_after = max(start_after, int(last_event_id))

    async def generator() -> AsyncIterator[str]:
        cursor = start_after
        idle_ticks = 0

        while True:
            if await request.is_disconnected():
                return

            async with session_scope() as stream_session:
                events = await event_service.list_events(
                    stream_session, run_id, after_sequence=cursor
                )
                run = await run_manager.get_run(
                    stream_session, run_id=run_id, user_id=user.id
                )

            for event in events:
                cursor = event.sequence
                yield _sse_frame(EventResponse.model_validate(event))

            if events:
                idle_ticks = 0
            else:
                idle_ticks += 1
                # Comment frame keeps proxies from closing an idle connection.
                if idle_ticks % 15 == 0:
                    yield ": keepalive\n\n"

            if is_terminal(RunState(run.state)) and not events:
                yield f"event: run_finished\ndata: {json.dumps({'state': run.state})}\n\n"
                return

            await asyncio.sleep(1.0)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
