"""Sandbox interface for running untrusted commands.

Both the repository's own code and any command the model proposes are untrusted. Running a
test suite executes arbitrary code from that repository, so it must not happen on the machine
holding our database credentials and GitHub tokens.

The interface has three implementations with genuinely different guarantees, and the
difference is stated rather than blurred:

| Backend | Isolation | Latency | Use |
|---|---|---|---|
| ``github_actions`` | ephemeral remote VM | 1-3 min | default |
| ``e2b`` | remote microVM | sub-second | fast repair loop |
| ``local`` | **process confinement only** | instant | development only |

The local runner is not a sandbox and is never described as one. Configuration validation
refuses it outside development.
"""

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

DEFAULT_COMMAND_TIMEOUT = 300


class IsolationLevel(StrEnum):
    """How strong the containment actually is. Surfaced in the UI, not hidden."""

    none = "none"
    process = "process"
    virtual_machine = "virtual_machine"


@dataclass
class ExecResult:
    argv: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def summary(self, *, limit: int = 4_000) -> str:
        """Trimmed output for prompts and logs.

        The **tail** of stdout is kept rather than the head: a test runner prints its
        failure summary last, so truncating from the front would discard the useful part.
        """
        command = " ".join(self.argv)
        status = "timed out" if self.timed_out else f"exit {self.exit_code}"

        stdout = self.stdout[-limit:] if len(self.stdout) > limit else self.stdout
        stderr = self.stderr[-2_000:] if len(self.stderr) > 2_000 else self.stderr

        parts = [f"$ {command}", f"({status}, {self.duration_ms}ms)"]

        if stdout.strip():
            parts.append(f"--- stdout ---\n{stdout.strip()}")
        if stderr.strip():
            parts.append(f"--- stderr ---\n{stderr.strip()}")

        return "\n".join(parts)


@dataclass
class SandboxSpec:
    """What the sandbox should provide for a run."""

    workspace: Path
    language: str | None = None
    install_command: str | None = None
    environment: dict[str, str] = field(default_factory=dict)

    #: No network during test execution by default. A test suite has no legitimate reason to
    #: reach the internet, and blocking it removes a whole class of exfiltration.
    network_enabled: bool = False

    cpu_limit: float = 1.0
    memory_limit_mb: int = 2048


class SandboxError(RuntimeError):
    pass


class SandboxRunner(Protocol):
    """The contract every backend implements."""

    @property
    def name(self) -> str: ...

    @property
    def isolation(self) -> IsolationLevel: ...

    async def prepare(self, spec: SandboxSpec) -> None:
        """Creates the environment and installs dependencies."""
        ...

    async def run(
        self,
        argv: list[str],
        *,
        # ASYNC109 prefers an external asyncio.timeout. Here the timeout must also kill the
        # process: abandoning the await would leave a test suite running in the background.
        timeout: int = DEFAULT_COMMAND_TIMEOUT,  # noqa: ASYNC109
    ) -> ExecResult:
        """Runs one command as an argument list. Never a shell string."""
        ...

    async def cleanup(self) -> None:
        """Releases whatever was created. Must be safe to call twice."""
        ...
