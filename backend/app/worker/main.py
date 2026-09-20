"""Worker process.

Runs separately from the API. It claims jobs from Postgres, drives the orchestrator and
persists every transition. If this process is killed mid-run, the job lease expires, the
job is requeued and the run resumes from its last committed state rather than restarting.

Two job kinds reach the orchestrator:

* ``execute_run`` drives a run forward until it finishes, parks at a human gate, or fails;
* ``create_pull_request`` runs after a human approves, and is the only thing that pushes.

Start with:  python -m app.worker.main
"""

import asyncio
import contextlib
import logging
import os
import signal
import socket
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.orchestrator import Orchestrator, create_pull_request
from app.agent.runtime import AgentRuntime, build_runtime
from app.agent.states import FailureCategory, RunState, is_terminal
from app.config.settings import Settings, get_settings
from app.db.session import dispose_engine, init_engine, session_scope
from app.github import tokens
from app.models.core import AgentRun
from app.observability.logging_setup import configure_logging, request_id_var
from app.services import event_service, job_queue

logger = logging.getLogger(__name__)

_shutdown = asyncio.Event()


def _worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"


async def _load_run(session: AsyncSession, run_id: str) -> AgentRun:
    return (await session.execute(select(AgentRun).where(AgentRun.id == run_id))).scalar_one()


async def _handle_execute_run(runtime: AgentRuntime, run_id: str) -> None:
    async with session_scope() as session:
        run = await _load_run(session, run_id)

        if is_terminal(RunState(run.state)):
            logger.info("run %s is terminal, nothing to do", run_id)
            return

        try:
            final_state = await Orchestrator(runtime).advance(session, run)
            await session.commit()
            logger.info("run %s paused/finished at %s", run_id, final_state)
        except Exception as exc:
            await session.rollback()
            await _record_failure(run_id, exc)
            raise


async def _handle_create_pull_request(
    runtime: AgentRuntime, settings: Settings, run_id: str
) -> None:
    """Publishes an approved run.

    The approval gate is re-checked inside `create_pull_request` against the diff being
    pushed, so a retry of this job cannot push content nobody approved.
    """
    async with session_scope() as session:
        run = await _load_run(session, run_id)

        if is_terminal(RunState(run.state)):
            logger.info("run %s is terminal, not publishing", run_id)
            return

        if RunState(run.state) is RunState.PR_CREATED:
            # A retry after the pull request was already opened. Nothing more to do.
            logger.info("run %s already has a pull request", run_id)
            return

        github_client = await tokens.client_for_user(
            session, settings=settings, user_id=run.user_id
        )

        await create_pull_request(
            session, runtime=runtime, run=run, github_client=github_client
        )
        await session.commit()


async def _record_failure(run_id: str, exc: BaseException) -> None:
    """Marks the run failed so the UI shows a cause rather than a run that stops moving."""
    async with session_scope() as session:
        run = await _load_run(session, run_id)

        if is_terminal(RunState(run.state)):
            return

        await event_service.transition(
            session,
            run_id,
            RunState.FAILED,
            message="Run failed during execution",
            failure_category=FailureCategory.TOOL_FAILURE,
            failure_detail=f"{type(exc).__name__}: {exc}"[:2000],
        )
        await session.commit()


async def _process_one(
    runtime: AgentRuntime, settings: Settings, worker_id: str, lease_seconds: int
) -> bool:
    """Claims and runs at most one job. Returns True if work was done."""
    async with session_scope() as session:
        job = await job_queue.claim_next(session, worker_id=worker_id, lease_seconds=lease_seconds)

        if job is None:
            await session.commit()
            return False

        job_id, job_kind, run_id = job.id, job.kind, job.run_id
        await session.commit()

    request_id_var.set(f"job:{job_id[:8]}")
    logger.info("claimed job %s kind=%s run=%s", job_id, job_kind, run_id)

    try:
        if job_kind == job_queue.JobKind.EXECUTE_RUN and run_id:
            await _handle_execute_run(runtime, run_id)
        elif job_kind == job_queue.JobKind.CREATE_PULL_REQUEST and run_id:
            await _handle_create_pull_request(runtime, settings, run_id)
        else:
            raise RuntimeError(f"Unsupported job kind: {job_kind}")

        async with session_scope() as session:
            await job_queue.mark_succeeded(session, job_id)
            await session.commit()

    except Exception as exc:
        logger.exception("job %s failed", job_id)

        async with session_scope() as session:
            status = await job_queue.mark_failed(
                session, job_id, error=f"{type(exc).__name__}: {exc}"
            )
            await session.commit()

        logger.warning("job %s marked %s", job_id, status)

    finally:
        request_id_var.set("-")

    return True


async def run_worker() -> None:
    settings = get_settings()

    configure_logging(settings.log_level)
    settings.validate_for_runtime()
    init_engine(settings)

    worker_id = _worker_id()

    # Built once, at startup. A missing API key or an unimplemented sandbox backend should
    # stop the worker here rather than failing the first run that happens to need it.
    runtime = build_runtime(settings)

    logger.info(
        "worker %s started (sandbox=%s isolation=%s, model=%s)",
        worker_id,
        runtime.sandbox.name,
        runtime.sandbox.isolation,
        runtime.llm.model,
    )

    try:
        while not _shutdown.is_set():
            try:
                async with session_scope() as session:
                    await job_queue.reclaim_expired(session)
                    await session.commit()

                did_work = await _process_one(
                    runtime, settings, worker_id, settings.worker_lease_seconds
                )
            except Exception:
                # The loop must survive transient database failures; a crashed worker
                # stops the whole platform.
                logger.exception("worker loop error")
                did_work = False

            if not did_work:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        _shutdown.wait(), timeout=settings.worker_poll_interval_seconds
                    )
    finally:
        await dispose_engine()
        logger.info("worker %s stopped", worker_id)


def main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _request_shutdown(*_: object) -> None:
        logger.info("shutdown signal received")
        loop.call_soon_threadsafe(_shutdown.set)

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, AttributeError):
            signal.signal(sig, _request_shutdown)

    try:
        loop.run_until_complete(run_worker())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
