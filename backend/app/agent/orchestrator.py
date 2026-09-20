"""The orchestrator: drives a run through the state machine, doing real work at each state.

This replaces the Phase 1 stub. Every state now performs its actual step, and the structure is
deliberately boring — one method per state, each one small enough to read in full:

```
CREATED              → clone the repository, pin the commit
CLONING_REPOSITORY   → analyse the issue (comprehension only)
ANALYZING_ISSUE      → build the repository map
EXPLORING_REPOSITORY → chunk and embed into pgvector
INDEXING_REPOSITORY  → retrieve code for the analysed queries
RETRIEVING_CONTEXT   → produce a plan, store it
PLANNING             → park at the plan gate, or implement
IMPLEMENTING         → apply changes and run the repair loop
TESTING              → verify, diff, score risk
VERIFYING            → park for human approval
WAITING_FOR_APPROVAL → (a human decides; a separate job opens the pull request)
```

**Resumability.** The driver reads the run's persisted state and executes the step for it, then
loops. A worker crash mid-run therefore resumes at the state that was last committed rather than
starting again. This is why each step commits before the next begins, and why the loop is a
`while` over the stored state rather than a fixed sequence of calls.

Resuming needs more than the state name, though: the plan, the checkout and the recorded edits
all live in memory during a normal run. `_rehydrate` rebuilds them from durable sources — the
plan from `agent_plan`, the repository map by re-reading the checkout, the edits from the git
working tree. What cannot be rebuilt honestly is not guessed at: if the workspace is gone after
the agent has already written code, the run fails and says so instead of publishing a diff it
cannot verify.

**Bounds.** Wall-clock, tool calls, tokens and iterations are all capped. Hitting a cap is a
clean `FAILED` with a stated category, never a hang.
"""

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import diff as diff_builder
from app.agent import planner, repair, risk
from app.agent.loop import ObserverType
from app.agent.runtime import AgentRuntime, RetrieverType
from app.agent.schemas import ImplementationPlan, IssueAnalysis
from app.agent.states import FailureCategory, RunState, awaits_human, is_terminal
from app.agent.testing import TestOutcome, all_passed, run_tests
from app.agent.tools.base import ToolContext
from app.agent.tools.editing import FileChange
from app.db.base import new_id
from app.llm.base import QuotaExceededError
from app.models.core import AgentRun, Issue, Repository, RepositorySnapshot
from app.rag import indexer, repo_map, workspace
from app.rag.repo_map import RepositoryMap
from app.rag.retrieve import RetrievedChunk
from app.sandbox.base import SandboxSpec
from app.services import event_service, review_service

logger = logging.getLogger(__name__)

#: States after which the agent has written code. Losing the workspace at or beyond these is
#: unrecoverable, because the edits only exist there.
_STATES_WITH_EDITS: frozenset[RunState] = frozenset(
    {
        RunState.TESTING,
        RunState.ANALYZING_FAILURE,
        RunState.REVISING,
        RunState.VERIFYING,
        RunState.WAITING_FOR_APPROVAL,
    }
)


class RunAborted(RuntimeError):
    """Raised to stop a run with a specific failure category."""

    def __init__(self, category: FailureCategory, detail: str) -> None:
        super().__init__(detail)
        self.category = category
        self.detail = detail


@dataclass
class RunScratch:
    """Work in progress for one execution of a run.

    Intentionally **not** persisted wholesale. Only the outputs that a human or a later phase
    needs — the plan, the diff, the risk assessment — are written to the database. Keeping the
    rest in memory avoids inventing a serialisation format for intermediate state that nothing
    reads, and everything here can be rebuilt from a durable source by `_rehydrate`.
    """

    checkout: Path | None = None
    commit_sha: str | None = None
    snapshot_id: str | None = None
    mapping: RepositoryMap | None = None
    analysis: IssueAnalysis | None = None
    retrieved: list[RetrievedChunk] = field(default_factory=list)

    #: Distinguishes "retrieval found nothing" from "retrieval has not run", so a resumed run
    #: retrieves once and a genuinely empty result is not retried on every step.
    retrieval_attempted: bool = False

    plan: ImplementationPlan | None = None
    repair_result: repair.RepairResult | None = None

    #: The test evidence the approval gate and the risk score are based on.
    test_outcomes: list[TestOutcome] = field(default_factory=list)

    #: What the write tools actually wrote. The review diff is built from this, never from the
    #: model's account of what it did.
    changes: list[FileChange] = field(default_factory=list)

    #: One tool context for the whole run, so the tool-call budget is cumulative across states
    #: rather than silently resetting at every step boundary.
    context: ToolContext | None = None

    started_at: float = field(default_factory=time.perf_counter)


class Orchestrator:
    def __init__(self, runtime: AgentRuntime) -> None:
        self._runtime = runtime
        self._settings = runtime.settings

    # ------------------------------------------------------------------ driver

    async def advance(self, session: AsyncSession, run: AgentRun) -> RunState:
        """Drives the run until it finishes, parks at a human gate, or fails."""
        scratch = RunScratch()

        current = RunState(run.state)

        if is_terminal(current) or awaits_human(current):
            logger.info("run %s is already at %s", run.id, current)
            return current

        try:
            await self._rehydrate(session, run, scratch)

            while True:
                current = RunState(run.state)

                if is_terminal(current) or awaits_human(current):
                    return current

                self._check_budget(scratch)

                step = self._steps().get(current)

                if step is None:
                    raise RunAborted(
                        FailureCategory.TOOL_FAILURE, f"No handler for state {current}"
                    )

                await step(session, run, scratch)
                await session.commit()

        except RunAborted as aborted:
            await self._fail(session, run, aborted.category, aborted.detail)
            return RunState.FAILED

        except QuotaExceededError as exhausted:
            # Not a bug in the agent, and not retryable inside the run. Saying so plainly is
            # the difference between "wait until tomorrow" and an afternoon of debugging.
            logger.warning("run %s stopped: provider quota exhausted", run.id)
            await self._fail(
                session,
                run,
                FailureCategory.QUOTA_EXHAUSTED,
                str(exhausted)[:2000],
            )
            return RunState.FAILED

        except Exception as exc:
            logger.exception("run %s failed unexpectedly", run.id)
            await self._fail(
                session,
                run,
                FailureCategory.TOOL_FAILURE,
                f"{type(exc).__name__}: {exc}"[:2000],
            )
            return RunState.FAILED

        finally:
            await self._cleanup(session, run, scratch)

    def _steps(self) -> dict[RunState, Any]:
        return {
            RunState.CREATED: self._clone,
            RunState.CLONING_REPOSITORY: self._analyse_issue,
            RunState.ANALYZING_ISSUE: self._explore,
            RunState.EXPLORING_REPOSITORY: self._index,
            RunState.INDEXING_REPOSITORY: self._retrieve,
            RunState.RETRIEVING_CONTEXT: self._plan,
            RunState.PLANNING: self._after_plan,
            RunState.IMPLEMENTING: self._implement,
            RunState.TESTING: self._assess,
            RunState.VERIFYING: self._await_approval,
        }

    def _check_budget(self, scratch: RunScratch) -> None:
        elapsed = time.perf_counter() - scratch.started_at

        if elapsed > self._settings.max_run_seconds:
            raise RunAborted(
                FailureCategory.TIMEOUT,
                f"Run exceeded {self._settings.max_run_seconds}s",
            )

    async def _fail(
        self,
        session: AsyncSession,
        run: AgentRun,
        category: FailureCategory,
        detail: str,
    ) -> None:
        if is_terminal(RunState(run.state)):
            return

        await event_service.transition(
            session,
            run.id,
            RunState.FAILED,
            message=f"Run failed: {category}",
            failure_category=category,
            failure_detail=detail,
        )
        await session.commit()

    async def _cleanup(
        self, session: AsyncSession, run: AgentRun, scratch: RunScratch
    ) -> None:
        """Releases the sandbox, and the checkout only when it is genuinely finished with.

        The workspace survives a pause at a human gate on purpose: the pull request is created
        from the files on disk, and re-cloning would lose the edits. It is removed once the run
        is terminal, because a checkout is the largest thing a run leaves behind and this
        machine has little disk headroom.
        """
        await self._runtime.sandbox.cleanup()

        if scratch.checkout is None or not scratch.checkout.exists():
            return

        if not is_terminal(RunState(run.state)):
            logger.info("keeping workspace for run %s at %s", run.id, run.state)
            return

        shutil.rmtree(scratch.checkout, ignore_errors=True)
        logger.info("removed workspace for run %s", run.id)

    async def _observer(self, session: AsyncSession, run: AgentRun) -> ObserverType:
        """Turns component callbacks into timeline events.

        Only operational facts reach the timeline: which tool, on what, did it work. The
        model's reasoning is never surfaced.
        """

        async def observe(kind: str, payload: dict[str, Any]) -> None:
            message = payload.get("message") or payload.get("tool") or kind

            if kind == "tool_call":
                message = f"{payload.get('tool')} ({'ok' if payload.get('ok') else 'failed'})"
            elif kind == "test_run":
                message = payload.get("summary", "tests run")
            elif kind == "retrieval":
                message = f"Retrieved {payload.get('chunks_found', 0)} additional code sections"

            await event_service.append_event(
                session,
                run.id,
                kind=kind if kind in {"tool_call", "test_run", "retrieval"} else "info",
                message=str(message)[:500],
                level="warning" if kind == "warning" else "info",
                payload=payload,
            )

        return observe

    # -------------------------------------------------------------- rehydration

    async def _rehydrate(
        self, session: AsyncSession, run: AgentRun, scratch: RunScratch
    ) -> None:
        """Rebuilds in-memory state for a run that is resuming part-way through.

        A no-op for a fresh run. For a resumed one it restores, from durable sources only:
        the checkout, the repository map, the snapshot, the stored plan and the edits already
        written. Anything that cannot be rebuilt truthfully aborts the run instead.
        """
        state = RunState(run.state)

        if state is RunState.CREATED:
            return

        logger.info("rehydrating run %s from state %s", run.id, state)

        scratch.snapshot_id = run.snapshot_id
        destination = self._runtime.workspace_for(run.id)

        if destination.is_dir():
            scratch.checkout = destination
            scratch.commit_sha = await workspace.current_commit(destination)
        elif state in _STATES_WITH_EDITS:
            raise RunAborted(
                FailureCategory.TOOL_FAILURE,
                "The workspace holding this run's edits is gone, so the change cannot be "
                "reviewed or published. The run must be started again.",
            )
        else:
            # Nothing has been written yet, so a fresh clone of the same commit is equivalent.
            await self._checkout(session, run, scratch)

        if scratch.checkout is not None:
            scratch.mapping = repo_map.build(scratch.checkout, workspace.scan(scratch.checkout))
            await self._runtime.sandbox.prepare(SandboxSpec(workspace=scratch.checkout))

        stored_plan = await review_service.latest_plan(session, run.id)

        if stored_plan is not None:
            scratch.plan = ImplementationPlan.model_validate(stored_plan.payload)

        if state in _STATES_WITH_EDITS and scratch.checkout is not None:
            scratch.changes = [
                FileChange(path=path, action=action, original_content=original)
                for path, action, original in await workspace.uncommitted_changes(
                    scratch.checkout
                )
            ]

            await event_service.append_event(
                session,
                run.id,
                kind=event_service.EventKind.INFO,
                message=f"Resumed at {state} with {len(scratch.changes)} recovered change(s)",
                level="warning",
            )

    async def _ensure_analysis(
        self, session: AsyncSession, run: AgentRun, scratch: RunScratch
    ) -> IssueAnalysis:
        """The issue analysis, re-derived if it was lost to a restart.

        Not persisted as a table of its own: it is one cheap model call away and nothing but
        the agent reads it. Re-deriving costs a fraction of what a schema for it would.
        """
        if scratch.analysis is not None:
            return scratch.analysis

        issue = await self._load_issue(session, run)

        analysis, _usage = await planner.analyse_issue(
            provider=self._runtime.llm,
            issue_title=issue.title,
            issue_body=issue.body,
        )
        scratch.analysis = analysis
        return analysis

    # ------------------------------------------------------------------- steps

    async def _context(self, run: AgentRun, scratch: RunScratch) -> ToolContext:
        if scratch.checkout is None:
            raise RunAborted(FailureCategory.TOOL_FAILURE, "Workspace is not prepared")

        if scratch.context is None:
            scratch.context = ToolContext(
                workspace=scratch.checkout,
                run_id=run.id,
                snapshot_id=scratch.snapshot_id,
                max_tool_calls=self._settings.max_tool_calls,
                changes=scratch.changes,
            )
        else:
            scratch.context.snapshot_id = scratch.snapshot_id

        return scratch.context

    async def _load_issue(self, session: AsyncSession, run: AgentRun) -> Issue:
        issue = (
            await session.execute(select(Issue).where(Issue.id == run.issue_id))
        ).scalar_one_or_none()

        if issue is None:
            raise RunAborted(FailureCategory.REPOSITORY_UNDERSTANDING_FAILURE, "Issue not found")

        return issue

    async def _load_repository(self, session: AsyncSession, run: AgentRun) -> Repository:
        repository = (
            await session.execute(select(Repository).where(Repository.id == run.repository_id))
        ).scalar_one_or_none()

        if repository is None:
            raise RunAborted(
                FailureCategory.REPOSITORY_UNDERSTANDING_FAILURE, "Repository not found"
            )

        return repository

    async def _checkout(
        self, session: AsyncSession, run: AgentRun, scratch: RunScratch
    ) -> None:
        """Clones the repository into this run's workspace and prepares the sandbox."""
        repository = await self._load_repository(session, run)

        if not repository.clone_url:
            raise RunAborted(
                FailureCategory.REPOSITORY_UNDERSTANDING_FAILURE,
                "Repository has no clone URL",
            )

        try:
            checkout = await workspace.clone(
                clone_url=repository.clone_url,
                destination=self._runtime.workspace_for(run.id),
                branch=repository.default_branch,
            )
        except workspace.GitError as exc:
            raise RunAborted(
                FailureCategory.REPOSITORY_UNDERSTANDING_FAILURE, f"Clone failed: {exc}"
            ) from exc

        scratch.checkout = checkout.path
        scratch.commit_sha = checkout.commit_sha

        await self._runtime.sandbox.prepare(SandboxSpec(workspace=checkout.path))

        await event_service.append_event(
            session,
            run.id,
            kind=event_service.EventKind.INFO,
            message=f"Checked out {repository.full_name} at {checkout.commit_sha[:8]}",
            payload={"commit": checkout.commit_sha, "branch": checkout.branch},
        )

    async def _clone(self, session: AsyncSession, run: AgentRun, scratch: RunScratch) -> None:
        await event_service.transition(
            session, run.id, RunState.CLONING_REPOSITORY, message="Preparing repository workspace"
        )

        await self._checkout(session, run, scratch)

    async def _analyse_issue(
        self, session: AsyncSession, run: AgentRun, scratch: RunScratch
    ) -> None:
        await event_service.transition(
            session, run.id, RunState.ANALYZING_ISSUE, message="Reading the issue"
        )

        issue = await self._load_issue(session, run)

        analysis, usage = await planner.analyse_issue(
            provider=self._runtime.llm,
            issue_title=issue.title,
            issue_body=issue.body,
        )

        if not analysis.is_actionable:
            # Abstaining is a valid outcome. Guessing at a vague issue wastes a run and
            # produces a diff nobody asked for.
            raise RunAborted(
                FailureCategory.PLANNING_FAILURE,
                f"Issue is not actionable: {analysis.clarification_needed or 'too vague'}",
            )

        scratch.analysis = analysis

        await event_service.append_event(
            session,
            run.id,
            kind=event_service.EventKind.INFO,
            message=f"Issue understood: {analysis.problem_summary[:200]}",
            payload={
                "search_queries": analysis.search_queries,
                "referenced_symbols": analysis.referenced_symbols,
                "tokens": usage.total_tokens,
            },
        )

    async def _explore(self, session: AsyncSession, run: AgentRun, scratch: RunScratch) -> None:
        await event_service.transition(
            session, run.id, RunState.EXPLORING_REPOSITORY, message="Inspecting repository"
        )

        if scratch.checkout is None:
            raise RunAborted(FailureCategory.TOOL_FAILURE, "Workspace is missing")

        files = workspace.scan(scratch.checkout)
        scratch.mapping = repo_map.build(scratch.checkout, files)

        if not scratch.mapping.test_command:
            # Not fatal, but the run loses its objective signal, so the reviewer is warned now
            # rather than at the end.
            await event_service.append_event(
                session,
                run.id,
                kind=event_service.EventKind.WARNING,
                message="No test command detected; this change cannot be verified objectively",
                level="warning",
            )

        await event_service.append_event(
            session,
            run.id,
            kind=event_service.EventKind.INFO,
            message=(
                f"{scratch.mapping.primary_language or 'unknown'} project, "
                f"{scratch.mapping.file_count} indexable files, "
                f"tests: {scratch.mapping.test_command or 'none found'}"
            ),
            payload=scratch.mapping.to_dict(),
        )

    async def _index(self, session: AsyncSession, run: AgentRun, scratch: RunScratch) -> None:
        await event_service.transition(
            session, run.id, RunState.INDEXING_REPOSITORY, message="Indexing source code"
        )

        if scratch.checkout is None or scratch.commit_sha is None:
            raise RunAborted(FailureCategory.TOOL_FAILURE, "Workspace is missing")

        snapshot = (
            await session.execute(
                select(RepositorySnapshot).where(
                    RepositorySnapshot.repository_id == run.repository_id,
                    RepositorySnapshot.commit_sha == scratch.commit_sha,
                )
            )
        ).scalar_one_or_none()

        previous_snapshot_id: str | None = None

        if snapshot is None:
            # Reuse the most recent snapshot's hashes so only changed files are re-embedded.
            previous = (
                await session.execute(
                    select(RepositorySnapshot)
                    .where(RepositorySnapshot.repository_id == run.repository_id)
                    .order_by(RepositorySnapshot.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            previous_snapshot_id = previous.id if previous else None

            snapshot = RepositorySnapshot(
                id=new_id(),
                repository_id=run.repository_id,
                commit_sha=scratch.commit_sha,
                index_status="indexing",
            )
            session.add(snapshot)
            await session.flush()

        scratch.snapshot_id = snapshot.id
        run.snapshot_id = snapshot.id

        if snapshot.index_status == "ready" and snapshot.chunk_count:
            # This commit is already indexed. Re-embedding it would cost money for nothing.
            await event_service.append_event(
                session,
                run.id,
                kind=event_service.EventKind.INFO,
                message=f"Reusing existing index ({snapshot.chunk_count} chunks)",
            )
            return

        try:
            result = await indexer.index_snapshot(
                session,
                snapshot=snapshot,
                checkout_path=scratch.checkout,
                provider=self._runtime.embeddings,
                previous_snapshot_id=previous_snapshot_id,
            )
        except Exception as exc:
            raise RunAborted(
                FailureCategory.RETRIEVAL_FAILURE, f"Indexing failed: {type(exc).__name__}: {exc}"
            ) from exc

        await event_service.append_event(
            session,
            run.id,
            kind=event_service.EventKind.INFO,
            message=(
                f"Indexed {result.files_indexed} file(s) into {result.chunks_created} new "
                f"chunks; {result.chunks_reused} carried forward from the previous commit"
            ),
            payload={
                "chunks_total": result.total_chunks,
                "chunks_created": result.chunks_created,
                "chunks_reused": result.chunks_reused,
                "embeddings_created": result.embeddings_created,
                "embeddings_reused": result.embeddings_reused,
                "model": result.embedding_model,
            },
        )

    def _retriever(self, scratch: RunScratch) -> RetrieverType:
        from app.agent.runtime import build_retriever
        from app.db.session import get_session_factory

        if scratch.snapshot_id is None:
            raise RunAborted(FailureCategory.RETRIEVAL_FAILURE, "Repository is not indexed")

        if self._runtime.retriever_factory is not None:
            return self._runtime.retriever_factory(scratch.snapshot_id)

        return build_retriever(
            session_factory=get_session_factory(),
            embeddings=self._runtime.embeddings,
            snapshot_id=scratch.snapshot_id,
            limit=self._settings.max_retrieved_chunks,
        )

    async def _retrieve(self, session: AsyncSession, run: AgentRun, scratch: RunScratch) -> None:
        await event_service.transition(
            session, run.id, RunState.RETRIEVING_CONTEXT, message="Retrieving relevant code"
        )

        analysis = await self._ensure_analysis(session, run, scratch)
        retriever = self._retriever(scratch)

        scratch.retrieved = await retriever(analysis.search_queries, analysis.referenced_symbols)
        scratch.retrieval_attempted = True

        if not scratch.retrieved:
            await event_service.append_event(
                session,
                run.id,
                kind=event_service.EventKind.WARNING,
                message="No relevant code was retrieved; the plan may be poorly informed",
                level="warning",
            )

        await event_service.append_event(
            session,
            run.id,
            kind=event_service.EventKind.RETRIEVAL,
            message=f"Retrieved {len(scratch.retrieved)} relevant code sections",
            payload={
                "files": sorted({chunk.relative_path for chunk in scratch.retrieved})[:20],
                "queries": analysis.search_queries,
            },
        )

    async def _plan(self, session: AsyncSession, run: AgentRun, scratch: RunScratch) -> None:
        await event_service.transition(
            session, run.id, RunState.PLANNING, message="Drafting an implementation plan"
        )

        analysis = await self._ensure_analysis(session, run, scratch)
        observer = await self._observer(session, run)

        plan, _outcome = await planner.build_plan(
            provider=self._runtime.llm,
            registry=self._runtime.registry,
            context=await self._context(run, scratch),
            analysis=analysis,
            retrieved=scratch.retrieved,
            repository_summary=(
                f"{scratch.mapping.primary_language}, {', '.join(scratch.mapping.frameworks)}"
                if scratch.mapping
                else ""
            ),
            observer=observer,
        )

        scratch.plan = plan
        await review_service.store_plan(session, run_id=run.id, plan=plan)

        await event_service.append_event(
            session,
            run.id,
            kind=event_service.EventKind.INFO,
            message=planner.plan_summary(plan),
            payload={
                "files": [change.path for change in plan.proposed_changes],
                "confidence": str(plan.root_cause_confidence),
            },
        )

    async def _after_plan(
        self, session: AsyncSession, run: AgentRun, scratch: RunScratch
    ) -> None:
        """Either park for plan review, or go straight to implementing."""
        if run.require_plan_approval:
            await event_service.transition(
                session,
                run.id,
                RunState.WAITING_FOR_PLAN_REVIEW,
                message="Plan ready for review",
            )
            return

        await event_service.transition(
            session, run.id, RunState.IMPLEMENTING, message="Applying changes"
        )

    async def _implement(
        self, session: AsyncSession, run: AgentRun, scratch: RunScratch
    ) -> None:
        if scratch.plan is None or scratch.mapping is None:
            raise RunAborted(FailureCategory.IMPLEMENTATION_FAILURE, "Plan is missing")

        if not scratch.retrieval_attempted:
            # A resumed run starts with no retrieved context. Implementing blind would work,
            # badly; one retrieval pass is far cheaper than a wrong change.
            analysis = await self._ensure_analysis(session, run, scratch)
            scratch.retrieved = await self._retriever(scratch)(
                analysis.search_queries, analysis.referenced_symbols
            )
            scratch.retrieval_attempted = True

        context = await self._context(run, scratch)
        observer = await self._observer(session, run)

        result = await repair.implement_and_repair(
            provider=self._runtime.llm,
            registry=self._runtime.registry,
            context=context,
            sandbox=self._runtime.sandbox,
            mapping=scratch.mapping,
            plan=scratch.plan,
            retrieved=scratch.retrieved,
            retriever=self._retriever(scratch),
            max_iterations=run.max_iterations,
            observer=observer,
        )

        scratch.repair_result = result
        scratch.changes = context.changes
        run.iteration = result.iterations_used
        run.tool_call_count = context.tool_calls_used

        await event_service.transition(
            session, run.id, RunState.TESTING, message=result.describe()
        )

    async def _outcomes(self, scratch: RunScratch) -> list[TestOutcome]:
        """The test evidence for the current change.

        Normally carried over from the repair loop. After a restart there is none, so the
        tests are run again rather than assuming anything about a change we did not watch
        being made.
        """
        result = scratch.repair_result

        if result is not None and result.attempts:
            return result.attempts[-1].test_outcomes

        if scratch.mapping is None:
            return []

        logger.info("no test evidence in memory; re-running tests")
        return await run_tests(
            self._runtime.sandbox,
            scratch.mapping,
            test_paths=scratch.plan.tests_to_add_or_update if scratch.plan else None,
        )

    async def _assess(self, session: AsyncSession, run: AgentRun, scratch: RunScratch) -> None:
        """Verify, build the diff, score risk, and persist all three for review."""
        await event_service.transition(
            session, run.id, RunState.VERIFYING, message="Verifying the change"
        )

        if scratch.plan is None or scratch.checkout is None:
            raise RunAborted(FailureCategory.IMPLEMENTATION_FAILURE, "Plan is missing")

        change_set = diff_builder.build_change_set(scratch.checkout, scratch.changes)

        if change_set.file_count == 0:
            raise RunAborted(
                FailureCategory.IMPLEMENTATION_FAILURE,
                "No files were changed, so there is nothing to review",
            )

        outcomes = await self._outcomes(scratch)
        scratch.test_outcomes = outcomes
        verification = None

        # Only ask for a self-check when tests passed. Asking a model to verify a change that
        # is known to be broken produces confident noise.
        if all_passed(outcomes):
            try:
                verification, _usage = await repair.verify(
                    provider=self._runtime.llm,
                    registry=self._runtime.registry,
                    context=await self._context(run, scratch),
                    plan=scratch.plan,
                    diff_text=change_set.unified_diff(),
                    outcomes=outcomes,
                    observer=await self._observer(session, run),
                )
            except Exception as exc:
                # A failed self-check must not discard a working change; it becomes a warning.
                logger.warning("verification failed: %s", exc)
                await event_service.append_event(
                    session,
                    run.id,
                    kind=event_service.EventKind.WARNING,
                    message="Self-verification could not be completed",
                    level="warning",
                )

        result = scratch.repair_result

        assessment = risk.assess(
            change_set,
            outcomes,
            planned_files=[change.path for change in scratch.plan.proposed_changes],
            iterations_used=result.iterations_used if result else 1,
        )

        await review_service.store_diff(
            session,
            run_id=run.id,
            change_set=change_set,
            risk=assessment,
            verification=verification,
        )

        await event_service.append_event(
            session,
            run.id,
            kind=event_service.EventKind.INFO,
            message=f"{change_set.summary()}. {assessment.describe()}",
            payload={
                "risk_level": str(assessment.level),
                "risk_score": assessment.score,
                "warnings": assessment.warnings,
                "blocking": assessment.blocking,
            },
        )

    async def _await_approval(
        self, session: AsyncSession, run: AgentRun, scratch: RunScratch
    ) -> None:
        outcomes = scratch.test_outcomes or await self._outcomes(scratch)

        if not all_passed(outcomes):
            # Reaching a human with failing tests is allowed, but it is a failure of the run's
            # goal and is recorded as one rather than presented as a success awaiting sign-off.
            result = scratch.repair_result
            raise RunAborted(
                FailureCategory.TEST_FAILURE,
                f"Tests did not pass: {result.stopped_because if result else 'no evidence'}",
            )

        await event_service.transition(
            session,
            run.id,
            RunState.WAITING_FOR_APPROVAL,
            message="Waiting for human approval",
        )


async def create_pull_request(
    session: AsyncSession,
    *,
    runtime: AgentRuntime,
    run: AgentRun,
    github_client: Any,
) -> None:
    """Opens the pull request for an approved run.

    A separate entry point, executed as its own job after approval. Two reasons: opening a pull
    request is several GitHub calls and should not block the HTTP request a reviewer made, and
    if it fails it can be retried without re-running the agent.

    The gate is checked here, immediately before the push, against the diff being pushed. A
    refusal is recorded as a run failure and re-raised: this is not something to retry, and it
    must not be swallowed.
    """
    try:
        await _publish(session, runtime=runtime, run=run, github_client=github_client)
    except RunAborted as aborted:
        await _fail_run(session, run, aborted.category, aborted.detail)
        raise
    except review_service.ApprovalRequiredError as refused:
        # The gate said no. The run is finished, not retried.
        await _fail_run(session, run, FailureCategory.SECURITY_BLOCK, str(refused))
        raise


async def _fail_run(
    session: AsyncSession, run: AgentRun, category: FailureCategory, detail: str
) -> None:
    if is_terminal(RunState(run.state)):
        return

    await event_service.transition(
        session,
        run.id,
        RunState.FAILED,
        message=f"Run failed: {category}",
        failure_category=category,
        failure_detail=detail[:2000],
    )
    await session.commit()


async def _publish(
    session: AsyncSession,
    *,
    runtime: AgentRuntime,
    run: AgentRun,
    github_client: Any,
) -> None:
    from app.github.pr import PullRequestClient, publish_changes

    diff = await review_service.latest_diff(session, run.id)
    plan = await review_service.latest_plan(session, run.id)

    if diff is None or plan is None:
        raise RunAborted(FailureCategory.TOOL_FAILURE, "Run has no diff or plan to publish")

    # Raises ApprovalRequiredError unless an approval exists for exactly this diff.
    approval = await review_service.assert_push_allowed(
        session, run_id=run.id, diff_text=diff.diff_text
    )

    repository = (
        await session.execute(select(Repository).where(Repository.id == run.repository_id))
    ).scalar_one()
    issue = (await session.execute(select(Issue).where(Issue.id == run.issue_id))).scalar_one()

    checkout = runtime.workspace_for(run.id)

    if not checkout.is_dir():
        raise RunAborted(
            FailureCategory.TOOL_FAILURE,
            "The workspace is no longer available; the run must be repeated",
        )

    # Paths come from the reviewed diff, and whether each one still exists on disk is what
    # separates an edit from a deletion. Parsing the diff alone cannot tell them apart.
    paths = sorted(
        {
            line.removeprefix("+++ b/").strip()
            for line in diff.diff_text.splitlines()
            if line.startswith("+++ b/")
        }
    )
    changed = [path for path in paths if (checkout / path).is_file()]
    deleted = [path for path in paths if not (checkout / path).is_file()]

    if not changed and not deleted:
        raise RunAborted(
            FailureCategory.TOOL_FAILURE, "The reviewed diff named no files to publish"
        )

    branch = review_service.branch_name(run.id, issue.number)
    title = f"Fix: {issue.title}"[:200]

    created = await publish_changes(
        PullRequestClient(github_client, owner=repository.owner, repository=repository.name),
        workspace=checkout,
        changed_paths=changed,
        deleted_paths=deleted,
        branch=branch,
        base_branch=repository.default_branch,
        commit_message=f"Fix #{issue.number}: {issue.title}"[:72],
        title=title,
        body=review_service.pull_request_body(
            issue_number=issue.number,
            plan=plan,
            diff=diff,
            test_summary="Tests passed",
            iterations=run.iteration,
        ),
    )

    await review_service.record_pull_request(
        session,
        run=run,
        repository_id=repository.id,
        approval=approval,
        branch=created.branch,
        title=title,
        number=created.number,
        html_url=created.html_url,
        commit_sha=created.head_sha,
    )

    await event_service.transition(session, run.id, RunState.COMPLETED, message="Run complete")
    await session.commit()

    # The run is finished and the files are on the branch, so the checkout has no further use.
    shutil.rmtree(checkout, ignore_errors=True)

    logger.info("run %s completed with pull request %s", run.id, created.html_url)
