"""Repository checkout and file scanning.

Two ideas drive this module.

**Commit pinning.** Everything indexed is tied to one commit SHA. Retrieval is then
scoped to that commit, so the agent can never be handed a chunk of code that no longer
exists in the version it is editing. Stale context becomes structurally impossible
rather than merely unlikely.

**Content hashing.** Each file's SHA-256 is recorded. Re-indexing a later commit only
re-parses and re-embeds files whose hash changed, which is what makes incremental
indexing possible.
"""

import asyncio
import hashlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from app.rag.exclusions import is_test_path, looks_minified, should_index

logger = logging.getLogger(__name__)

#: Git operations are network-bound and can hang; never wait forever.
CLONE_TIMEOUT_SECONDS = 300
GIT_TIMEOUT_SECONDS = 60

#: 0xC0000142, Windows STATUS_DLL_INIT_FAILED. A process that cannot even start produces no
#: stderr, so without special-casing it the failure looks like a git error with no message.
WINDOWS_DLL_INIT_FAILED = 3221225794


class GitError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScannedFile:
    relative_path: str
    absolute_path: Path
    language: str | None
    size_bytes: int
    line_count: int
    content_hash: str
    is_test: bool


@dataclass(frozen=True)
class Checkout:
    path: Path
    commit_sha: str
    branch: str


#: Extension to tree-sitter language name.
LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".kt": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "bash",
    ".sql": "sql",
    ".vue": "vue",
    ".svelte": "svelte",
}


def language_for(relative_path: str) -> str | None:
    return LANGUAGE_BY_EXTENSION.get(Path(relative_path).suffix.lower())


async def _run_git(
    *args: str,
    cwd: Path | None = None,
    # ASYNC109 suggests an external asyncio.timeout instead. Here the timeout must also
    # kill the subprocess: abandoning the await would leave an orphaned git process.
    timeout: int = GIT_TIMEOUT_SECONDS,  # noqa: ASYNC109
    strip: bool = True,
) -> str:
    """Runs git with an argument list, never a shell string.

    Passing a list means a branch or path containing shell metacharacters cannot be
    interpreted as a command. This matters more once repository names and branches come
    from user input or from a language model.
    """
    process = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        raise GitError(f"git {args[0]} timed out after {timeout}s") from exc

    if process.returncode != 0:
        # Git error text can contain a token embedded in a remote URL, so it is
        # truncated and the logging redaction filter strips credentials.
        detail = stderr.decode(errors="replace").strip()[:500]

        # The exit code is included because a failed *launch* produces no stderr at all, and
        # "git clone failed: " with nothing after it is impossible to act on.
        if process.returncode == WINDOWS_DLL_INIT_FAILED:
            raise GitError(
                f"git {args[0]} could not start (exit 0xC0000142): the machine could not "
                f"allocate memory for a new process. Close other applications and retry."
            )

        raise GitError(
            f"git {args[0]} failed (exit {process.returncode}): {detail or 'no error output'}"
        )

    output = stdout.decode(errors="replace")

    # File contents must survive verbatim: stripping a trailing newline off `git show`
    # output would make an unchanged final line look edited in the diff.
    return output.strip() if strip else output


async def clone(
    *,
    clone_url: str,
    destination: Path,
    branch: str | None = None,
    depth: int = 1,
) -> Checkout:
    """Shallow-clones a repository and resolves the exact commit.

    ``depth=1`` fetches only the tip commit. Full history can be hundreds of megabytes
    and the agent needs the current state of the code, not its past. Later phases can
    deepen the clone on demand if history is genuinely required.
    """
    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)

    destination.parent.mkdir(parents=True, exist_ok=True)

    args = ["clone", "--depth", str(depth), "--single-branch"]
    if branch:
        args += ["--branch", branch]
    args += [clone_url, str(destination)]

    await _run_git(*args, timeout=CLONE_TIMEOUT_SECONDS)

    commit_sha = await _run_git("rev-parse", "HEAD", cwd=destination)
    resolved_branch = branch or await _run_git(
        "rev-parse", "--abbrev-ref", "HEAD", cwd=destination
    )

    logger.info("cloned into %s at %s", destination.name, commit_sha[:8])
    return Checkout(path=destination, commit_sha=commit_sha, branch=resolved_branch)


async def current_commit(repository_path: Path) -> str:
    return await _run_git("rev-parse", "HEAD", cwd=repository_path)


#: git status letters mapped to the action names the diff builder uses.
_STATUS_ACTION = {"?": "created", "A": "created", "D": "deleted", "M": "modified", "R": "created"}


async def uncommitted_changes(repository_path: Path) -> list[tuple[str, str, str | None]]:
    """Every working-tree change, as ``(relative_path, action, content_at_HEAD)``.

    Used to recover a change set after a worker crash. The edits an agent makes live in the
    working tree, so the original content recorded in memory by the write tools is lost if the
    process dies — but git still has it at ``HEAD``, and the checkout is pinned to one commit.
    Reading it back from git therefore reconstructs a truthful diff rather than guessing.
    """
    # strip=False matters: porcelain status codes are two columns and an unstaged
    # modification begins with a space, so trimming the output shifts every path by one.
    porcelain = await _run_git("status", "--porcelain", cwd=repository_path, strip=False)

    if not porcelain.strip():
        return []

    recovered: list[tuple[str, str, str | None]] = []

    for raw in porcelain.splitlines():
        line = raw.rstrip("\r\n")

        if len(line) < 4:
            continue

        status = line[:2]
        path = line[3:].strip().strip('"')

        # A rename is reported as "old -> new"; only the new path exists on disk.
        if " -> " in path:
            path = path.split(" -> ", 1)[1]

        letter = status.strip()[:1] or "M"
        action = _STATUS_ACTION.get(letter, "modified")

        original: str | None = None

        if action != "created":
            try:
                original = await _run_git(
                    "show", f"HEAD:{path}", cwd=repository_path, strip=False
                )
            except GitError:
                # Present in the index but not at HEAD. Treated as a new file.
                action = "created"

        recovered.append((path, action, original))

    logger.info("recovered %s uncommitted change(s) from git", len(recovered))
    return recovered


def hash_content(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_bytes(path: Path) -> bytes | None:
    """Reads a file, or returns None if it is binary or unreadable.

    A NUL byte in the first block is the standard heuristic for binary content. Binary
    files must be skipped rather than crashing the scan.
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        logger.warning("could not read %s: %s", path.name, type(exc).__name__)
        return None

    if b"\x00" in data[:8192]:
        return None

    return data


def scan(checkout_path: Path) -> list[ScannedFile]:
    """Walks the checkout and returns the files worth indexing.

    Pruning excluded directories during the walk rather than filtering afterwards means
    we never descend into ``node_modules``, which on a real repository can be tens of
    thousands of files.
    """
    from app.rag.exclusions import EXCLUDED_DIRECTORIES

    results: list[ScannedFile] = []
    skipped = 0

    for directory, subdirectories, filenames in checkout_path.walk():
        # Mutating the list in place tells walk() not to descend into these.
        subdirectories[:] = [name for name in subdirectories if name not in EXCLUDED_DIRECTORIES]

        for filename in filenames:
            absolute = directory / filename
            relative = absolute.relative_to(checkout_path).as_posix()

            try:
                size = absolute.stat().st_size
            except OSError:
                continue

            if not should_index(relative, size):
                skipped += 1
                continue

            data = read_bytes(absolute)
            if data is None:
                skipped += 1
                continue

            text = data.decode("utf-8", errors="replace")

            if looks_minified(text):
                skipped += 1
                continue

            results.append(
                ScannedFile(
                    relative_path=relative,
                    absolute_path=absolute,
                    language=language_for(relative),
                    size_bytes=size,
                    line_count=text.count("\n") + 1,
                    content_hash=hash_content(data),
                    is_test=is_test_path(relative),
                )
            )

    logger.info("scanned %s indexable files, skipped %s", len(results), skipped)
    return sorted(results, key=lambda item: item.relative_path)


def changed_files(
    previous: dict[str, str], current: list[ScannedFile]
) -> tuple[list[ScannedFile], list[str], list[str]]:
    """Compares content hashes between two snapshots.

    Returns (added or modified, unchanged paths, deleted paths). This is the whole basis
    of incremental indexing: unchanged files keep their existing chunks and embeddings,
    so re-indexing a repository after a one-line change costs one file, not thousands.
    """
    current_by_path = {item.relative_path: item for item in current}

    modified = [
        item
        for path, item in current_by_path.items()
        if previous.get(path) != item.content_hash
    ]
    unchanged = [
        path for path, item in current_by_path.items() if previous.get(path) == item.content_hash
    ]
    deleted = [path for path in previous if path not in current_by_path]

    return modified, unchanged, deleted
