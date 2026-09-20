"""Structured outputs the agent must produce.

Every one of these is parsed by code and acted on, so free-form prose is unusable. Defining
them as Pydantic models gives three things at once: the JSON Schema sent to the model,
validation of what comes back, and a typed object for the rest of the code.

Two conventions worth noting:

* **Hypotheses are labelled as hypotheses.** ``suspected_root_cause`` is what the model
  currently believes, not a finding. Naming it honestly keeps the UI honest too.
* **Confidence and abstention are first-class.** A model that cannot tell should be able to
  say so, rather than being forced to invent an answer.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class Confidence(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class IssueAnalysis(BaseModel):
    """The issue restated in terms the agent can act on.

    Derived from untrusted issue text, so it is a *reading* of the issue, not a fact.
    """

    problem_summary: str = Field(
        max_length=600, description="One or two sentences describing the actual problem"
    )
    expected_behaviour: str = Field(max_length=400)
    actual_behaviour: str = Field(max_length=400)

    reproduction_steps: list[str] = Field(
        default_factory=list, max_length=10, description="Steps to reproduce, if stated"
    )
    acceptance_criteria: list[str] = Field(
        default_factory=list, max_length=10, description="What must be true for this to be done"
    )

    error_messages: list[str] = Field(default_factory=list, max_length=10)
    referenced_symbols: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Function, class or file names named in the issue. Used for exact search.",
    )
    referenced_paths: list[str] = Field(default_factory=list, max_length=20)

    #: Drives the retrieval queries, which is why it is a list rather than prose.
    search_queries: list[str] = Field(
        min_length=1,
        max_length=8,
        description="Natural-language queries to find the relevant code",
    )

    is_actionable: bool = Field(
        description="False if the issue is too vague or lacks information to attempt"
    )
    clarification_needed: str | None = Field(
        default=None, max_length=400, description="What is missing, when not actionable"
    )


class ProposedChange(BaseModel):
    path: str = Field(max_length=1024)
    intent: str = Field(max_length=400, description="What changes in this file, and why")
    is_new_file: bool = False


class ImplementationPlan(BaseModel):
    """The plan a human reviews before any code is written."""

    problem_understanding: str = Field(max_length=800)

    relevant_files: list[str] = Field(
        min_length=1, max_length=20, description="Files inspected and judged relevant"
    )

    #: Explicitly a hypothesis. The tests are what turn it into a conclusion.
    suspected_root_cause: str = Field(
        max_length=800, description="Current best hypothesis, not a confirmed finding"
    )
    root_cause_confidence: Confidence

    proposed_changes: list[ProposedChange] = Field(min_length=1, max_length=15)

    tests_to_add_or_update: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Test files or cases that should cover this change",
    )
    risks: list[str] = Field(
        default_factory=list, max_length=10, description="What this change could break"
    )
    verification_strategy: str = Field(
        max_length=600, description="How success will be checked, including which tests to run"
    )

    #: Guards against scope creep, the most common failure of coding agents.
    out_of_scope: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Related problems deliberately not addressed",
    )


class FailureAnalysis(BaseModel):
    """Read from test output after a failed attempt."""

    failing_tests: list[str] = Field(default_factory=list, max_length=20)
    error_type: str = Field(max_length=200)
    error_summary: str = Field(max_length=800)

    likely_cause: str = Field(max_length=800)
    caused_by_our_change: bool = Field(
        description="True if the previous edit introduced this, false if pre-existing"
    )

    files_to_inspect: list[str] = Field(default_factory=list, max_length=10)

    #: Failure output is one of the best sources of retrieval queries: it names real symbols.
    additional_search_queries: list[str] = Field(default_factory=list, max_length=6)

    suggested_fix: str = Field(max_length=800)
    is_recoverable: bool = Field(
        description="False when the agent should stop rather than keep iterating"
    )


class VerificationResult(BaseModel):
    """The final self-check before a human is asked to look."""

    issue_addressed: bool
    tests_pass: bool
    unrelated_changes_detected: bool = Field(
        description="True if the diff touches files unrelated to the stated plan"
    )
    concerns: list[str] = Field(default_factory=list, max_length=10)
    summary: str = Field(max_length=800, description="What changed and why, for the reviewer")
    confidence: Confidence
