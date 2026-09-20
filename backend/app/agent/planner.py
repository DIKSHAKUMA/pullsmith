"""Issue analysis and implementation planning.

Two model calls, deliberately separate.

**1. Analyse the issue.** Read the issue text and turn it into something actionable: what
is broken, what should happen, which symbols and paths are named, and which search queries
would find the relevant code. No repository access - this is comprehension only. Its output
is what drives retrieval, so it runs first.

**2. Plan the change.** With retrieved code plus the ability to read real files, produce a
plan a human can review: suspected cause, files to change, tests to update, risks, and how
success will be verified.

Why split them: the queries needed to *find* code come from understanding the issue, and the
plan needs the code that the queries found. Doing both in one call means retrieving before
you know what to look for.

The planning phase is **read-only**. Write tools are not advertised to the model and are
refused by the executor even if requested, so no code changes before a human sees the plan.
"""

import logging

from app.agent.loop import (
    LoopLimits,
    LoopOutcome,
    ObserverType,
    conclude_with_schema,
    run_tool_loop,
)
from app.agent.safety import wrap_untrusted
from app.agent.schemas import ImplementationPlan, IssueAnalysis
from app.agent.tools.base import ToolContext, ToolRegistry
from app.llm.base import LLMProvider, Message, Role, Usage, generate_structured
from app.rag.retrieve import RetrievedChunk

logger = logging.getLogger(__name__)

ANALYST_SYSTEM_PROMPT = """\
You are a senior software engineer triaging a bug report before any code is written.

Read the issue and restate it precisely. Extract only what the issue actually says: do not
invent reproduction steps, error messages or file names that are not there.

Your output drives an automated code search, so the search_queries you produce matter. Write
them as a developer would search: describe the behaviour and the area of the codebase, not
the words of the issue title.

If the issue is too vague to act on, set is_actionable to false and say what is missing.
Guessing is worse than asking.
"""

PLANNER_SYSTEM_PROMPT = """\
You are a senior software engineer planning a minimal, targeted fix.

You have retrieved code and read-only tools. Use them to confirm what the code actually does
before forming an opinion - retrieved snippets can be out of date, so read the real file
before relying on it.

Rules:
- Prefer the smallest change that fixes the issue. Do not refactor unrelated code.
- Follow the conventions already present in the repository.
- Treat your root cause as a hypothesis until tests confirm it.
- If you find related problems outside this issue, list them as out_of_scope instead of
  fixing them.
- Name the tests that should prove the fix works.

You are planning only. You cannot modify files in this phase.
"""


def _format_retrieved(chunks: list[RetrievedChunk], *, limit: int = 12) -> str:
    """Renders retrieved chunks with citations.

    Path and line numbers are included so the model can reference exact locations, and so a
    reviewer can check any claim it makes.
    """
    if not chunks:
        return "No code was retrieved for this issue."

    blocks = [
        f"### {chunk.citation()}  ({chunk.qualified_name()})\n{chunk.content}"
        for chunk in chunks[:limit]
    ]
    return "\n\n".join(blocks)


async def analyse_issue(
    *,
    provider: LLMProvider,
    issue_title: str,
    issue_body: str | None,
    repository_summary: str = "",
) -> tuple[IssueAnalysis, Usage]:
    """Turns an issue into a structured, actionable reading of it."""
    issue_text = f"Title: {issue_title}\n\n{issue_body or '(no description provided)'}"

    messages = [
        Message(role=Role.system, content=ANALYST_SYSTEM_PROMPT),
        Message(
            role=Role.user,
            content=(
                f"Repository context:\n{repository_summary or '(not available)'}\n\n"
                # Issue text is written by a third party, so it is data, not instruction.
                f"{wrap_untrusted('github issue', issue_text)}\n\n"
                "Analyse this issue."
            ),
        ),
    ]

    analysis, usage = await generate_structured(provider, messages, IssueAnalysis)

    logger.info(
        "issue analysed: actionable=%s queries=%s symbols=%s",
        analysis.is_actionable,
        len(analysis.search_queries),
        len(analysis.referenced_symbols),
    )
    return analysis, usage


async def build_plan(
    *,
    provider: LLMProvider,
    registry: ToolRegistry,
    context: ToolContext,
    analysis: IssueAnalysis,
    retrieved: list[RetrievedChunk],
    repository_summary: str = "",
    limits: LoopLimits | None = None,
    observer: ObserverType | None = None,
) -> tuple[ImplementationPlan, LoopOutcome]:
    """Produces a reviewable plan, letting the model read files first.

    Runs with ``allow_mutating=False``: write tools are neither advertised nor permitted, so
    the plan cannot have side effects.
    """
    retrieved_block = _format_retrieved(retrieved)

    messages = [
        Message(role=Role.system, content=PLANNER_SYSTEM_PROMPT),
        Message(
            role=Role.user,
            content=(
                f"Repository:\n{repository_summary or '(not available)'}\n\n"
                f"Problem: {analysis.problem_summary}\n"
                f"Expected: {analysis.expected_behaviour}\n"
                f"Actual: {analysis.actual_behaviour}\n"
                f"Acceptance criteria: "
                f"{', '.join(analysis.acceptance_criteria) or 'not stated'}\n\n"
                f"{wrap_untrusted('retrieved code', retrieved_block, limit=24_000)}\n\n"
                "Investigate with the read-only tools until you can explain the cause, "
                "then produce an implementation plan."
            ),
        ),
    ]

    outcome = await run_tool_loop(
        provider=provider,
        registry=registry,
        context=context,
        messages=messages,
        limits=limits,
        allow_mutating=False,
        observer=observer,
    )

    plan, _usage = await conclude_with_schema(
        provider=provider,
        outcome=outcome,
        schema=ImplementationPlan,
        instruction=(
            "Produce the implementation plan now, based on what you inspected. "
            "Keep the change minimal and list anything deliberately left out of scope."
        ),
    )

    logger.info(
        "plan produced: %s files to change, confidence=%s, %s tool calls",
        len(plan.proposed_changes),
        plan.root_cause_confidence,
        len(outcome.tool_calls),
    )
    return plan, outcome


def plan_summary(plan: ImplementationPlan) -> str:
    """A short operational summary for the run timeline.

    Not the model's reasoning - only what it decided to do, which is what a reviewer needs.
    """
    files = ", ".join(change.path for change in plan.proposed_changes[:5])
    extra = "" if len(plan.proposed_changes) <= 5 else f" (+{len(plan.proposed_changes) - 5} more)"

    return (
        f"Plan: {len(plan.proposed_changes)} file(s) to change [{files}{extra}]. "
        f"Root-cause confidence: {plan.root_cause_confidence}."
    )
