"""Local subprocess runner. **Development only — this is not isolation.**

It gives process confinement: a fixed working directory, a stripped environment, an
executable allowlist, argv-only execution and a hard timeout. That is genuinely useful for
developing the agent loop without a network round trip.

What it does **not** give: filesystem isolation, network isolation, or protection from
anything the executed code chooses to do. Code run here can read files outside the workspace
and open sockets. It is labelled ``IsolationLevel.process`` everywhere it appears, and
``Settings.validate_for_runtime`` refuses it outside development.

Anything that would be described to a user as "the sandbox" uses a remote backend.
"""

import asyncio
import logging
import os
import time
from pathlib import Path

from app.agent.safety import assert_safe_command
from app.sandbox.base import (
    DEFAULT_COMMAND_TIMEOUT,
    ExecResult,
    IsolationLevel,
    SandboxError,
    SandboxSpec,
)

logger = logging.getLogger(__name__)

#: Environment variables passed through. Everything else is dropped, so the child cannot
#: read DATABASE_URL, GEMINI_API_KEY or a GitHub token out of its own environment.
ENVIRONMENT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "PATH",
        "SYSTEMROOT",
        "COMSPEC",
        "TEMP",
        "TMP",
        "HOME",
        "USERPROFILE",
        "LANG",
        "LC_ALL",
        "PYTHONIOENCODING",
        "PYTHONUNBUFFERED",
    }
)

MAX_CAPTURED_BYTES = 1_000_000


class LocalSubprocessSandbox:
    """Runs commands in a subprocess. Not isolated; see the module docstring."""

    def __init__(self) -> None:
        self._spec: SandboxSpec | None = None

    @property
    def name(self) -> str:
        return "local_unsafe"

    @property
    def isolation(self) -> IsolationLevel:
        # Reported honestly so the UI can label it and config validation can refuse it.
        return IsolationLevel.process

    def _environment(self) -> dict[str, str]:
        base = {key: value for key, value in os.environ.items() if key in ENVIRONMENT_ALLOWLIST}

        # Keeps test output deterministic and unbuffered so a timeout still yields logs.
        base["PYTHONUNBUFFERED"] = "1"
        base["PYTHONIOENCODING"] = "utf-8"
        base["CI"] = "1"

        if self._spec:
            base.update(self._spec.environment)

        return base

    async def prepare(self, spec: SandboxSpec) -> None:
        if not spec.workspace.is_dir():
            raise SandboxError(f"Workspace does not exist: {spec.workspace}")

        self._spec = spec

        logger.warning(
            "local sandbox prepared at %s - process confinement only, NOT isolation",
            spec.workspace,
        )

        if spec.install_command:
            # Dependency installation is the one step that legitimately needs the network.
            result = await self.run(spec.install_command.split(), timeout=600)

            if not result.ok:
                raise SandboxError(f"Dependency installation failed: {result.summary(limit=800)}")

    async def run(
        self,
        argv: list[str],
        *,
        timeout: int = DEFAULT_COMMAND_TIMEOUT,  # noqa: ASYNC109 - must kill the process
    ) -> ExecResult:
        if self._spec is None:
            raise SandboxError("prepare() must be called before run()")

        # Same allowlist the tool layer uses: only known executables, no shell metacharacters.
        assert_safe_command(argv)

        started = time.perf_counter()

        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(self._spec.workspace),
            env=self._environment(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            timed_out = False
        except TimeoutError:
            # Kill rather than abandon: an orphaned test process would hold the workspace
            # and keep consuming CPU on a machine that has very little.
            process.kill()
            stdout, stderr = await process.communicate()
            timed_out = True

            logger.warning("command timed out after %ss: %s", timeout, argv[0])

        duration_ms = int((time.perf_counter() - started) * 1000)

        return ExecResult(
            argv=list(argv),
            exit_code=process.returncode if process.returncode is not None else -1,
            stdout=stdout.decode("utf-8", errors="replace")[:MAX_CAPTURED_BYTES],
            stderr=stderr.decode("utf-8", errors="replace")[:MAX_CAPTURED_BYTES],
            duration_ms=duration_ms,
            timed_out=timed_out,
        )

    async def cleanup(self) -> None:
        # Nothing to tear down: no container, no remote resource. The workspace is owned by
        # the run, not by the sandbox.
        self._spec = None


def build_local_sandbox(_workspace: Path | None = None) -> LocalSubprocessSandbox:
    return LocalSubprocessSandbox()
