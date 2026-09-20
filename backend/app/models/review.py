"""Review artefacts: the plan, the diff, the approval decision and the pull request.

The important table here is ``approval``. It is the **only** thing that authorises a push, and
it is a persisted row rather than a flag in memory or a check in the UI. Two properties follow
from that:

* the gate survives a restart — an in-memory approval would be lost and, worse, might be
  treated as absent-but-assumed later;
* the decision is auditable — who approved what, when, and against which diff hash.

``diff_hash`` exists so an approval cannot be reused for different content. If the workspace
changed after approval, the hash no longer matches and the push is refused.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, IdMixin, TimestampMixin


class AgentPlan(Base, IdMixin, TimestampMixin):
    """The plan a human reviews, stored as produced."""

    __tablename__ = "agent_plan"
    __table_args__ = (Index("ix_plan_run", "run_id"),)

    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agent_run.id", ondelete="CASCADE"), nullable=False
    )

    #: Sequential per run, so a revision request produces version 2 rather than overwriting.
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    problem_understanding: Mapped[str] = mapped_column(Text, nullable=False)
    suspected_root_cause: Mapped[str] = mapped_column(Text, nullable=False)
    root_cause_confidence: Mapped[str] = mapped_column(String(16), nullable=False)
    verification_strategy: Mapped[str] = mapped_column(Text, nullable=False)

    #: The full structured plan. Kept whole so the review screen can show exactly what the
    #: agent committed to, not a lossy summary of it.
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RunDiff(Base, IdMixin, TimestampMixin):
    """The change set produced by a run, as reviewed."""

    __tablename__ = "run_diff"
    __table_args__ = (Index("ix_diff_run", "run_id"),)

    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agent_run.id", ondelete="CASCADE"), nullable=False
    )

    diff_text: Mapped[str] = mapped_column(Text, nullable=False)

    #: SHA-256 of diff_text. An approval is bound to this value.
    diff_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    files_changed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    lines_added: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    lines_removed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    risk_level: Mapped[str] = mapped_column(String(16), default="LOW", nullable=False)
    risk_score: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    risk_reasons: Mapped[list[str] | None] = mapped_column(JSON)
    risk_warnings: Mapped[list[str] | None] = mapped_column(JSON)

    #: True when a credential was detected. Blocks approval outright.
    risk_blocking: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    sensitive_paths: Mapped[list[str] | None] = mapped_column(JSON)
    verification: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class Approval(Base, IdMixin):
    """A recorded human decision. The only authorisation to push."""

    __tablename__ = "approval"
    __table_args__ = (
        # One decision per run. A second decision is a conflict, not an overwrite.
        UniqueConstraint("run_id", name="uq_approval_per_run"),
        Index("ix_approval_run", "run_id"),
    )

    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agent_run.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("app_user.id", ondelete="CASCADE"), nullable=False
    )

    #: "approved", "rejected" or "revision_requested".
    decision: Mapped[str] = mapped_column(String(24), nullable=False)

    #: Binds the decision to the exact content reviewed. A later change invalidates it.
    diff_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class PullRequest(Base, IdMixin, TimestampMixin):
    __tablename__ = "pull_request"
    __table_args__ = (Index("ix_pr_run", "run_id"),)

    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agent_run.id", ondelete="CASCADE"), nullable=False
    )
    repository_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("repository.id", ondelete="CASCADE"), nullable=False
    )

    #: Recorded so a push can never be retried without a fresh approval.
    approval_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("approval.id", ondelete="RESTRICT"), nullable=False
    )

    branch: Mapped[str] = mapped_column(String(255), nullable=False)
    number: Mapped[int | None] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    html_url: Mapped[str | None] = mapped_column(String(512))
    commit_sha: Mapped[str | None] = mapped_column(String(64))

    approval: Mapped[Approval] = relationship()
