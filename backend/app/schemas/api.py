"""Request and response contracts.

Response models are separate from ORM models on purpose. Serialising a database row
directly is how internal fields such as encrypted tokens end up in an API response.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: str
    database: str
    app_env: str
    sandbox_backend: str

    #: Which sign-in methods are actually usable. The UI needs this to avoid offering a GitHub
    #: button that would 503, or a token form that is blocked in production. Both are
    #: capability flags, not secrets: neither reveals a credential.
    github_oauth_configured: bool = False
    dev_login_available: bool = False

    #: Queue depth, and how long the oldest queued job has been waiting. Inferring "is a worker
    #: running" from these is not perfect, but it is *truthful*: a job sitting queued for
    #: minutes means nothing is consuming the queue. A config flag would go stale the first time
    #: someone forgot to set it, which is the failure mode this avoids. The symptom it explains
    #: - a run stuck at CREATED forever - is otherwise completely silent.
    queued_jobs: int = 0
    oldest_queued_job_seconds: int | None = None


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    github_login: str
    display_name: str | None
    avatar_url: str | None


class RepositoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    owner: str
    name: str
    full_name: str
    default_branch: str
    primary_language: str | None
    is_private: bool


class GitHubRepositoryOption(BaseModel):
    """A repository available on GitHub but not necessarily linked here yet."""

    github_repo_id: int
    owner: str
    name: str
    full_name: str
    default_branch: str
    primary_language: str | None
    is_private: bool


class LinkRepositoryRequest(BaseModel):
    owner: str = Field(min_length=1, max_length=255)
    name: str = Field(min_length=1, max_length=255)


class IssueResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    number: int
    title: str
    body: str | None
    state: str
    labels: list[str] | None
    html_url: str | None


class SyncIssuesResponse(BaseModel):
    synced: int
    issues: list[IssueResponse]


class CreateRunRequest(BaseModel):
    repository_id: str
    issue_id: str
    require_plan_approval: bool = True
    max_iterations: int | None = Field(default=None, ge=1, le=20)


class RunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    repository_id: str
    issue_id: str
    state: str
    failure_category: str | None
    iteration: int
    max_iterations: int
    tool_call_count: int
    require_plan_approval: bool
    awaiting_human: bool = False
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    last_event_sequence: int


class RunCreatedResponse(BaseModel):
    run_id: str
    state: str


class EventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    sequence: int
    kind: str
    state: str | None
    message: str
    level: str
    payload: dict | None
    created_at: datetime


class PlanResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    version: int
    problem_understanding: str
    suspected_root_cause: str
    root_cause_confidence: str
    verification_strategy: str
    payload: dict


class RunDiffResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    diff_text: str
    diff_hash: str
    files_changed: int
    lines_added: int
    lines_removed: int
    risk_level: str
    risk_score: int
    risk_reasons: list[str] | None
    risk_warnings: list[str] | None
    risk_blocking: bool
    sensitive_paths: list[str] | None
    verification: dict | None


class ApprovalResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    decision: str
    diff_hash: str
    comment: str | None
    created_at: datetime


class ReviewBundleResponse(BaseModel):
    """Everything the review screen needs, in one round trip."""

    run_id: str
    state: str
    awaiting_approval: bool
    iterations_used: int

    plan: PlanResponse | None
    diff: RunDiffResponse | None
    approval: ApprovalResponse | None

    #: Precomputed so the UI does not re-derive the rule and get it wrong.
    can_approve: bool


class PullRequestResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    branch: str
    number: int | None
    title: str
    html_url: str | None
    commit_sha: str | None
