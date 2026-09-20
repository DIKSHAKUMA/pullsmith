"""Application configuration.

Configuration is read once from the environment and validated at import time so the
process fails fast on misconfiguration instead of failing on the first request that
happens to need a missing value.
"""

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(StrEnum):
    development = "development"
    test = "test"
    production = "production"


class SandboxBackend(StrEnum):
    github_actions = "github_actions"
    e2b = "e2b"
    local_unsafe = "local_unsafe"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: AppEnv = AppEnv.development
    log_level: str = "INFO"

    database_url: str = "sqlite+aiosqlite:///./local.db"

    # Deliberately obvious placeholder; validate_for_runtime() rejects it in production.
    session_secret: str = "insecure-development-secret"  # noqa: S105
    session_cookie_name: str = "ase_session"
    session_ttl_seconds: int = 60 * 60 * 12
    token_encryption_key: str | None = None

    github_client_id: str | None = None
    github_client_secret: str | None = None
    github_oauth_redirect_uri: str = "http://localhost:8000/auth/github/callback"
    github_api_base: str = "https://api.github.com"

    frontend_origin: str = "http://localhost:5173"

    max_iterations: int = Field(default=5, ge=1, le=20)
    max_tool_calls: int = Field(default=60, ge=1)
    max_run_seconds: int = Field(default=1800, ge=60)
    max_retrieved_chunks: int = Field(default=24, ge=1)

    sandbox_backend: SandboxBackend = SandboxBackend.github_actions

    #: Where repositories are checked out, one directory per run. Configurable because a
    #: checkout is the largest thing a run leaves on disk and it may need to live on a
    #: different volume from the application.
    agent_workspace_root: str = ".agent-workspaces"

    #: "gemini" or "fake". The fake provider is deterministic and offline, which is what
    #: lets the indexing and retrieval tests run with no network and no spend.
    embedding_provider: str = "fake"
    gemini_api_key: str | None = None
    embedding_model: str = "gemini-embedding-001"

    #: Configurable because model names are retired. `gemini-2.5-flash` stopped accepting new
    #: API keys and returned a 404 mid-run, so this is pinned here and changeable without a
    #: code edit. A `-latest` alias would avoid the rot at the cost of reproducibility.
    llm_model: str = "gemini-3.6-flash"

    #: Gemini 3 reasons before answering and charges those thinking tokens against this budget.
    #: 4096 was enough for the answer and not for the reasoning, so a large prompt produced an
    #: empty response and a failed run. Sized for reasoning plus a full ImplementationPlan.
    llm_max_output_tokens: int = Field(default=16_384, ge=1024, le=65_536)

    #: Must match models.rag.EMBEDDING_DIMENSIONS. The column dimension is fixed at
    #: migration time, so changing this alone would store vectors the database rejects.
    embedding_dimensions: int = Field(default=1536, ge=8, le=2000)
    embedding_batch_size: int = Field(default=32, ge=1, le=100)

    #: Client-side pacing for the embedding API. Free tiers rate-limit aggressively, and a
    #: rejected request still counts against quota, so pacing beats retrying. 0 disables.
    embedding_requests_per_minute: int = Field(default=0, ge=0, le=6000)

    worker_poll_interval_seconds: float = 1.0
    worker_lease_seconds: int = 120

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()

    @property
    def is_production(self) -> bool:
        return self.app_env is AppEnv.production

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    def validate_for_runtime(self) -> None:
        """Checks that only apply when actually serving traffic.

        Kept out of the validators so unit tests can build a Settings object without a
        full production environment.
        """
        problems: list[str] = []

        if self.is_production:
            if self.session_secret == "insecure-development-secret":  # noqa: S105
                problems.append("SESSION_SECRET must be set in production")
            if not self.token_encryption_key:
                problems.append("TOKEN_ENCRYPTION_KEY must be set in production")
            if self.is_sqlite:
                problems.append("DATABASE_URL must point at Postgres in production")
            if self.sandbox_backend is SandboxBackend.local_unsafe:
                problems.append(
                    "SANDBOX_BACKEND=local_unsafe provides no isolation and is "
                    "forbidden outside development"
                )

            if self.embedding_provider == "fake":
                problems.append(
                    "EMBEDDING_PROVIDER=fake returns meaningless vectors and is for tests "
                    "only"
                )

        if self.embedding_provider == "gemini" and not self.gemini_api_key:
            problems.append("EMBEDDING_PROVIDER=gemini requires GEMINI_API_KEY")

        # Catching this here turns a confusing database error during indexing into a clear
        # startup failure.
        from app.models.rag import EMBEDDING_DIMENSIONS

        if self.embedding_dimensions != EMBEDDING_DIMENSIONS:
            problems.append(
                f"EMBEDDING_DIMENSIONS={self.embedding_dimensions} does not match the "
                f"vector column dimension {EMBEDDING_DIMENSIONS}; a migration is required "
                f"to change it"
            )

        if problems:
            raise RuntimeError("Invalid configuration: " + "; ".join(problems))


@lru_cache
def get_settings() -> Settings:
    return Settings()
