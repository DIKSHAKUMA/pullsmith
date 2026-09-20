"""Implementation and the test-driven repair loop.

```
implement  →  run tests  →  passed?  →  verify
                  │  failed
                  ▼
          analyse the failure
                  │
    new search queries from failing test names
                  │
          retrieve more context
                  │
              revise  →  run tests again      (bounded)
```

The loop is what makes this test-driven rather than hopeful: the model's opinion is replaced by
an objective signal each round, and the failure output is turned into the *input* for the next
attempt rather than just being logged.

Two things it must never do:

* **Run forever.** Bounded by iterations, tool calls, tokens and wall clock. Hitting a bound
  is a clean stop with a stated reason, not a crash.
* **Give up silently.** ``FailureAnalysis.is_recoverable`` lets the agent stop early and say
  why, which is more useful than five identical failed attempts.
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from app.agent.loop import LoopLimits, ObserverType, conclude_with_schema, run_tool_loop
from app.agent.safety import wrap_untrusted
from app.agent.schemas import FailureAnalysis, ImplementationPlan, VerificationResult
from app.agent.testing import TestOutcome, all_passed, run_tests
from app.agent.tools.base import ToolContext, ToolRegistry
from app.llm.base import LLMProvider, Message, Role, Usage
from app.rag.repo_map import RepositoryMap
from app.rag.retrieve import RetrievedChunk
from app.sandbox.base import SandboxRunner

logger = logging.getLogger(__name__)

#: Called with the failure's derived queries so the caller can run retrieval. Injected rather
#: than imported so this module does not depend on a database session.
RetrieverType = Callable[[list[str], list[str]], Awaitable[list[RetrievedChunk]]]

IMPLEMENTER_SYSTEM_PROMPT = """\
You are a senior software engineer implementing an approved plan.

Rules:
- Make the smallest change that satisfies the plan. Do not refactor anything else.
- Use replace_in_file for edits to existing code. Read the file first so your snippet matches
  exactly.
- Follow the conventions already in the file you are editing.
- Add or update tests when the plan says so.
- Do not touch files outside the plan unless it is strictly required, and say so if you do.

When you have finished editing, stop and say what you changed.
"""

REVISER_SYSTEM_PROMPT = """\
You are fixing a failed attempt.

The tests told you the truth: your previous change did not work. Read the failure carefully
before editing again. Do not repeat the same edit.

If the failure shows your root cause was wrong, say so and address the real cause instead of
patching the symptom.
"""


@dataclass
class Attempt:
    iteration: int
    test_outcomes: list[TestOutcome] = field(default_factory=list)
    failure_analysis: FailureAnalysis | None = None
    tool_calls: int = 0

    @property
    def passed(self) -> bool:
        return all_passed(self.test_outcomes)


@dataclass
class RepairResult:
    succeeded: bool
    attempts: list[Attempt] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stopped_because: str = "tests_passed"
    verification: VerificationResult | None = None

    @property
    def iterations_used(self) -> int:
        return len(self.attempts)

    def describe(self) -> str:
        if self.succeeded:
            return f"Tests passed after {self.iterations_used} attempt(s)"

        return f"Stopped after {self.iterations_used} attempt(s): {self.stopped_because}"


def _format_plan(plan: ImplementationPlan) -> str:
    changes = "\n".join(
        f"- {change.path}: {change.intent}" for change in plan.proposed_changes
    )
    tests = ", ".join(plan.tests_to_add_or_update) or "none specified"

    return (
        f"Suspected cause ({plan.root_cause_confidence} confidence): "
        f"{plan.suspected_root_cause}\n\n"
        f"Changes to make:\n{changes}\n\n"
        f"Tests to add or update: {tests}\n"
        f"Verification: {plan.verification_strategy}\n"
        f"Out of scope: {', '.join(plan.out_of_scope) or 'nothing stated'}"
    )


def _format_failures(outcomes: list[TestOutcome]) -> str:
    return "\n\n".join(outcome.output_summary for outcome in outcomes if not outcome.passed)


async def _apply_changes(
    *,
    provider: LLMProvider,
    registry: ToolRegistry,
    context: ToolContext,
    messages: list[Message],
    limits: LoopLimits | None,
    observer: ObserverType | None,
) -> tuple[int, Usage]:
    """Runs the tool loop with write access enabled. Returns tool calls made and usage."""
    outcome = await run_tool_loop(
        provider=provider,
        registry=registry,
        context=context,
        messages=messages,
        limits=limits,
        # The only phase where this is true. Planning and analysis are read-only.
        allow_mutating=True,
        observer=observer,
    )

    return len(outcome.tool_calls), outcome.usage


async def implement_and_repair(
    *,
    provider: LLMProvider,
    registry: ToolRegistry,
    context: ToolContext,
    sandbox: SandboxRunner,
    mapping: RepositoryMap,
    plan: ImplementationPlan,
    retrieved: list[RetrievedChunk],
    retriever: RetrieverType | None = None,
    max_iterations: int = 5,
    limits: LoopLimits | None = None,
    observer: ObserverType | None = None,
) -> RepairResult:
    """Implements the plan, then repairs until tests pass or a bound is reached."""
    result = RepairResult(succeeded=False)

    async def report(kind: str, payload: dict) -> None:
        if observer is not None:
            await observer(kind, payload)

    context_block = "\n\n".join(
        f"### {chunk.citation()}\n{chunk.content}" for chunk in retrieved[:10]
    )

    conversation = [
        Message(role=Role.system, content=IMPLEMENTER_SYSTEM_PROMPT),
        Message(
            role=Role.user,
            content=(
                f"Approved plan:\n{_format_plan(plan)}\n\n"
                f"{wrap_untrusted('relevant code', context_block, limit=24_000)}\n\n"
                "Implement this plan now."
            ),
        ),
    ]

    for iteration in range(1, max_iterations + 1):
        attempt = Attempt(iteration=iteration)
        result.attempts.append(attempt)

        await report(
            "info",
            {"message": f"Attempt {iteration} of {max_iterations}: applying changes"},
        )

        calls, usage = await _apply_changes(
            provider=provider,
            registry=registry,
            context=context,
            messages=conversation,
            limits=limits,
            observer=observer,
        )
        attempt.tool_calls = calls
        result.usage.add(usage)

        if context.tool_calls_used >= context.max_tool_calls:
            result.stopped_because = "tool_budget_exhausted"
            await report("warning", {"message": "Tool call budget exhausted"})
            return result

        # The objective signal. Everything before this was opinion.
        attempt.test_outcomes = await run_tests(
            sandbox, mapping, test_paths=plan.tests_to_add_or_update
        )

        await report(
            "test_run",
            {
                "iteration": iteration,
                "passed": attempt.passed,
                "summary": attempt.test_outcomes[-1].describe()
                if attempt.test_outcomes
                else "no tests were run",
            },
        )

        if attempt.passed:
            result.succeeded = True
            result.stopped_because = "tests_passed"
            return result

        if not attempt.test_outcomes:
            # Nothing was verified, and no amount of iterating will change that.
            result.stopped_because = "no_test_command"
            await report(
                "warning",
                {"message": "No test command detected; cannot verify the change objectively"},
            )
            return result

        if iteration == max_iterations:
            result.stopped_because = "max_iterations_reached"
            await report(
                "warning", {"message": f"Stopping after {max_iterations} attempts"}
            )
            return result

        # --- learn from the failure ---------------------------------------------------
        analysis, analysis_usage = await analyse_failure(
            provider=provider,
            plan=plan,
            outcomes=attempt.test_outcomes,
        )
        attempt.failure_analysis = analysis
        result.usage.add(analysis_usage)

        await report(
            "info",
            {
                "message": f"Failure analysed: {analysis.error_type}",
                "caused_by_our_change": analysis.caused_by_our_change,
                "recoverable": analysis.is_recoverable,
            },
        )

        if not analysis.is_recoverable:
            # An honest early stop beats four more identical failures.
            result.stopped_because = "analysis_says_unrecoverable"
            await report(
                "warning",
                {"message": f"Stopping: {analysis.likely_cause[:200]}"},
            )
            return result

        # Failing test names and error text are excellent retrieval queries: they name real
        # symbols, which is exactly what lexical search is good at.
        extra_context: list[RetrievedChunk] = []

        if retriever is not None and analysis.additional_search_queries:
            symbols = [
                name.split("::")[-1]
                for outcome in attempt.test_outcomes
                for name in outcome.failing_tests
            ]
            extra_context = await retriever(analysis.additional_search_queries, symbols)

            await report(
                "retrieval",
                {
                    "queries": analysis.additional_search_queries,
                    "chunks_found": len(extra_context),
                },
            )

        extra_block = "\n\n".join(
            f"### {chunk.citation()}\n{chunk.content}" for chunk in extra_context[:8]
        )

        conversation = [
            Message(role=Role.system, content=REVISER_SYSTEM_PROMPT),
            Message(
                role=Role.user,
                content=(
                    f"Original plan:\n{_format_plan(plan)}\n\n"
                    f"Attempt {iteration} failed.\n\n"
                    f"{wrap_untrusted('test output', _format_failures(attempt.test_outcomes))}\n\n"
                    f"Analysis: {analysis.likely_cause}\n"
                    f"Suggested fix: {analysis.suggested_fix}\n"
                    f"Caused by our change: {analysis.caused_by_our_change}\n\n"
                    + (
                        f"{wrap_untrusted('additional code', extra_block, limit=16_000)}\n\n"
                        if extra_block
                        else ""
                    )
                    + "Revise the implementation."
                ),
            ),
        ]

    result.stopped_because = "max_iterations_reached"
    return result


async def analyse_failure(
    *,
    provider: LLMProvider,
    plan: ImplementationPlan,
    outcomes: list[TestOutcome],
) -> tuple[FailureAnalysis, Usage]:
    """Reads test output and decides what to do next.

    A separate, focused call. Its most valuable outputs are the derived search queries and the
    ``is_recoverable`` judgement.
    """
    messages = [
        Message(
            role=Role.system,
            content=(
                "You are diagnosing a failed test run. Be concrete and specific. "
                "If the failure shows the original root cause was wrong, say so. "
                "Set is_recoverable to false if further attempts are unlikely to help, for "
                "example a missing dependency or an environment problem rather than a code bug."
            ),
        ),
        Message(
            role=Role.user,
            content=(
                f"We attempted this change:\n{_format_plan(plan)}\n\n"
                f"{wrap_untrusted('test output', _format_failures(outcomes))}\n\n"
                "Diagnose the failure."
            ),
        ),
    ]

    from app.llm.base import generate_structured

    analysis, usage = await generate_structured(provider, messages, FailureAnalysis)

    logger.info(
        "failure analysed: %s, ours=%s, recoverable=%s",
        analysis.error_type,
        analysis.caused_by_our_change,
        analysis.is_recoverable,
    )
    return analysis, usage


async def verify(
    *,
    provider: LLMProvider,
    registry: ToolRegistry,
    context: ToolContext,
    plan: ImplementationPlan,
    diff_text: str,
    outcomes: list[TestOutcome],
    limits: LoopLimits | None = None,
    observer: ObserverType | None = None,
) -> tuple[VerificationResult, Usage]:
    """A final self-check before a human is asked to look.

    Read-only: verification must not be able to "fix" what it is checking. Its job is to
    report, including reporting that something is wrong.
    """
    messages = [
        Message(
            role=Role.system,
            content=(
                "You are reviewing your own completed change before a human sees it. "
                "Be critical. Report unrelated changes, missing tests and anything you are "
                "unsure about. A concern raised now is far cheaper than one found in review."
            ),
        ),
        Message(
            role=Role.user,
            content=(
                f"Plan:\n{_format_plan(plan)}\n\n"
                f"Test result: "
                f"{outcomes[-1].describe() if outcomes else 'no tests were run'}\n\n"
                f"{wrap_untrusted('final diff', diff_text, limit=20_000)}\n\n"
                "Verify this change against the plan."
            ),
        ),
    ]

    outcome = await run_tool_loop(
        provider=provider,
        registry=registry,
        context=context,
        messages=messages,
        limits=limits or LoopLimits(max_steps=5),
        allow_mutating=False,
        observer=observer,
    )

    result, usage = await conclude_with_schema(
        provider=provider,
        outcome=outcome,
        schema=VerificationResult,
        instruction="Give your verification result now.",
    )

    logger.info(
        "verification: addressed=%s tests_pass=%s unrelated=%s confidence=%s",
        result.issue_addressed,
        result.tests_pass,
        result.unrelated_changes_detected,
        result.confidence,
    )
    return result, usage
