"""Phase 1 ORM models: identity, GitHub linkage, repositories, issues, runs, events, jobs."""

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

from app.agent.states import RunState
from app.db.base import Base, IdMixin, TimestampMixin


class User(Base, IdMixin, TimestampMixin):
    __tablename__ = "app_user"

    github_login: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    github_user_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(255))
    avatar_url: Mapped[str | None] = mapped_column(String(512))

    connection: Mapped["GitHubConnection | None"] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )


class GitHubConnection(Base, IdMixin, TimestampMixin):
    """Stores the GitHub access token encrypted at rest.

    The plaintext token never touches this table, is never logged and is never
    returned by any API response.
    """

    __tablename__ = "github_connection"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("app_user.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    encrypted_token: Mapped[str] = mapped_column(Text, nullable=False)
    token_type: Mapped[str] = mapped_column(String(32), default="bearer", nullable=False)
    scopes: Mapped[str | None] = mapped_column(String(512))

    user: Mapped[User] = relationship(back_populates="connection")


class Repository(Base, IdMixin, TimestampMixin):
    __tablename__ = "repository"
    __table_args__ = (UniqueConstraint("user_id", "github_repo_id", name="uq_repo_per_user"),)

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("app_user.id", ondelete="CASCADE"), nullable=False
    )
    github_repo_id: Mapped[int] = mapped_column(Integer, nullable=False)
    owner: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(512), nullable=False)
    default_branch: Mapped[str] = mapped_column(String(255), default="main", nullable=False)
    primary_language: Mapped[str | None] = mapped_column(String(64))
    is_private: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    clone_url: Mapped[str | None] = mapped_column(String(512))

    snapshots: Mapped[list["RepositorySnapshot"]] = relationship(
        back_populates="repository", cascade="all, delete-orphan"
    )


class RepositorySnapshot(Base, IdMixin, TimestampMixin):
    """A repository pinned to one commit.

    Every indexed chunk and embedding hangs off a snapshot, which is what makes stale
    retrieval structurally impossible rather than merely unlikely.
    """

    __tablename__ = "repository_snapshot"
    __table_args__ = (
        UniqueConstraint("repository_id", "commit_sha", name="uq_snapshot_commit"),
        Index("ix_snapshot_repo", "repository_id"),
    )

    repository_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("repository.id", ondelete="CASCADE"), nullable=False
    )
    commit_sha: Mapped[str] = mapped_column(String(64), nullable=False)
    index_status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    file_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    repository_map: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    repository: Mapped[Repository] = relationship(back_populates="snapshots")


class Issue(Base, IdMixin, TimestampMixin):
    __tablename__ = "issue"
    __table_args__ = (UniqueConstraint("repository_id", "number", name="uq_issue_number"),)

    repository_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("repository.id", ondelete="CASCADE"), nullable=False
    )
    number: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(1024), nullable=False)
    body: Mapped[str | None] = mapped_column(Text)
    state: Mapped[str] = mapped_column(String(32), default="open", nullable=False)
    labels: Mapped[list[str] | None] = mapped_column(JSON)
    html_url: Mapped[str | None] = mapped_column(String(512))


class AgentRun(Base, IdMixin, TimestampMixin):
    __tablename__ = "agent_run"
    __table_args__ = (Index("ix_run_user_created", "user_id", "created_at"),)

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("app_user.id", ondelete="CASCADE"), nullable=False
    )
    repository_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("repository.id", ondelete="CASCADE"), nullable=False
    )
    issue_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("issue.id", ondelete="CASCADE"), nullable=False
    )
    snapshot_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("repository_snapshot.id", ondelete="SET NULL")
    )

    state: Mapped[str] = mapped_column(String(48), default=RunState.CREATED, nullable=False)
    failure_category: Mapped[str | None] = mapped_column(String(64))
    failure_detail: Mapped[str | None] = mapped_column(Text)

    iteration: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_iterations: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    tool_call_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    require_plan_approval: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    #: Highest event sequence written for this run. Incremented inside the same
    #: transaction as the event insert so sequences are gap-free and ordered.
    last_event_sequence: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    events: Mapped[list["AgentEvent"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class AgentEvent(Base, IdMixin):
    """Append-only operational timeline for a run.

    Never contains model reasoning. Only operational summaries the user is allowed to
    see, which is also what makes the timeline safe to expose in the UI.
    """

    __tablename__ = "agent_event"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence", name="uq_event_sequence"),
        Index("ix_event_run_sequence", "run_id", "sequence"),
    )

    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agent_run.id", ondelete="CASCADE"), nullable=False
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(48), nullable=False)
    state: Mapped[str | None] = mapped_column(String(48))
    message: Mapped[str] = mapped_column(Text, nullable=False)
    level: Mapped[str] = mapped_column(String(16), default="info", nullable=False)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    run: Mapped[AgentRun] = relationship(back_populates="events")


class Job(Base, IdMixin, TimestampMixin):
    """Durable queue row.

    Claimed with ``SELECT ... FOR UPDATE SKIP LOCKED`` so multiple workers can poll the
    same table without ever handing the same job to two workers.
    """

    __tablename__ = "job"
    __table_args__ = (Index("ix_job_claimable", "status", "run_after"),)

    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("agent_run.id", ondelete="CASCADE")
    )
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    status: Mapped[str] = mapped_column(String(24), default="queued", nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3, nullable=False)

    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    locked_by: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
