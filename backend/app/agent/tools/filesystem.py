"""Read-only repository tools.

These exist alongside Code RAG rather than instead of it. Retrieval finds *candidates*;
these tools let the agent verify them against the file actually on disk. Retrieval can be
stale or wrong, so before editing anything the agent reads the real file. Trusting a
retrieved snippet as ground truth is how an agent edits code that no longer exists.

Every path goes through ``resolve_in_workspace`` and every output is wrapped as untrusted
data.
"""

import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field

from app.agent.safety import (
    MAX_READ_BYTES,
    assert_readable,
    resolve_in_workspace,
    wrap_untrusted,
)
from app.agent.tools.base import Tool, ToolContext, ToolRegistry, ToolResult
from app.rag.exclusions import EXCLUDED_DIRECTORIES

logger = logging.getLogger(__name__)

MAX_LIST_ENTRIES = 200
MAX_SEARCH_MATCHES = 40
MAX_TREE_ENTRIES = 300


class ListDirectoryArgs(BaseModel):
    path: str = Field(default=".", description="Directory path relative to the repository root")


class ReadFileArgs(BaseModel):
    path: str = Field(description="File path relative to the repository root")
    start_line: int | None = Field(
        default=None, ge=1, description="First line to read, 1-based"
    )
    end_line: int | None = Field(default=None, ge=1, description="Last line to read, inclusive")


class SearchCodeArgs(BaseModel):
    pattern: str = Field(min_length=2, max_length=200, description="Text or regex to find")
    path: str = Field(default=".", description="Directory to search within")
    is_regex: bool = Field(default=False)
    case_sensitive: bool = Field(default=False)


class FindSymbolArgs(BaseModel):
    symbol: str = Field(min_length=1, max_length=200, description="Function or class name")


class TreeArgs(BaseModel):
    path: str = Field(default=".")
    max_depth: int = Field(default=3, ge=1, le=6)


async def list_directory(context: ToolContext, args: ListDirectoryArgs) -> ToolResult:
    target = resolve_in_workspace(context.workspace, args.path)

    if not target.is_dir():
        return ToolResult.failure("NOT_A_DIRECTORY", f"Not a directory: {args.path}")

    entries: list[str] = []

    for child in sorted(target.iterdir(), key=lambda item: (item.is_file(), item.name)):
        if child.name in EXCLUDED_DIRECTORIES:
            continue

        entries.append(f"{child.name}/" if child.is_dir() else child.name)

        if len(entries) >= MAX_LIST_ENTRIES:
            entries.append(f"... truncated at {MAX_LIST_ENTRIES} entries")
            break

    return ToolResult.success(
        wrap_untrusted(f"listing of {args.path}", "\n".join(entries)),
        entry_count=len(entries),
    )


async def read_file(context: ToolContext, args: ReadFileArgs) -> ToolResult:
    assert_readable(args.path)
    target = resolve_in_workspace(context.workspace, args.path)

    if not target.is_file():
        return ToolResult.failure("NOT_FOUND", f"No such file: {args.path}")

    if target.stat().st_size > MAX_READ_BYTES:
        return ToolResult.failure(
            "FILE_TOO_LARGE",
            f"{args.path} exceeds {MAX_READ_BYTES} bytes; read a line range instead",
        )

    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")

    start = args.start_line or 1
    end = args.end_line or len(lines)

    if start > len(lines):
        return ToolResult.failure(
            "RANGE_OUT_OF_BOUNDS", f"{args.path} has {len(lines)} lines; start_line was {start}"
        )

    selected = lines[start - 1 : end]

    # Line numbers are included because the agent must cite exact locations, and because
    # its next action is often an edit at a specific line.
    numbered = "\n".join(
        f"{number:>5} | {line}" for number, line in enumerate(selected, start=start)
    )

    return ToolResult.success(
        wrap_untrusted(f"{args.path} lines {start}-{start + len(selected) - 1}", numbered),
        path=args.path,
        total_lines=len(lines),
        returned_lines=len(selected),
    )


def _iter_source_files(root: Path) -> list[Path]:
    files: list[Path] = []

    for directory, subdirectories, filenames in root.walk():
        subdirectories[:] = [name for name in subdirectories if name not in EXCLUDED_DIRECTORIES]

        for filename in filenames:
            path = directory / filename

            try:
                if path.stat().st_size <= MAX_READ_BYTES:
                    files.append(path)
            except OSError:
                continue

    return files


async def search_code(context: ToolContext, args: SearchCodeArgs) -> ToolResult:
    """Literal or regex search across the repository, returning file:line matches."""
    root = resolve_in_workspace(context.workspace, args.path)

    if not root.exists():
        return ToolResult.failure("NOT_FOUND", f"No such path: {args.path}")

    flags = 0 if args.case_sensitive else re.IGNORECASE

    try:
        # A model-supplied regex can be invalid; that is a normal error to report back.
        pattern = re.compile(args.pattern if args.is_regex else re.escape(args.pattern), flags)
    except re.error as exc:
        return ToolResult.failure("INVALID_REGEX", f"Pattern rejected: {exc}")

    search_root = root if root.is_dir() else root.parent
    candidates = [root] if root.is_file() else _iter_source_files(search_root)

    matches: list[str] = []
    files_matched: set[str] = set()

    for path in candidates:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        if "\x00" in content[:2048]:
            continue

        for number, line in enumerate(content.split("\n"), start=1):
            if pattern.search(line):
                relative = path.relative_to(context.workspace.resolve()).as_posix()
                files_matched.add(relative)
                matches.append(f"{relative}:{number}: {line.strip()[:200]}")

                if len(matches) >= MAX_SEARCH_MATCHES:
                    break

        if len(matches) >= MAX_SEARCH_MATCHES:
            matches.append(f"... truncated at {MAX_SEARCH_MATCHES} matches")
            break

    if not matches:
        return ToolResult.success(f"No matches for {args.pattern!r}", match_count=0)

    return ToolResult.success(
        wrap_untrusted(f"matches for {args.pattern!r}", "\n".join(matches)),
        match_count=len(matches),
        files_matched=sorted(files_matched),
    )


async def find_symbol(context: ToolContext, args: FindSymbolArgs) -> ToolResult:
    """Finds where a function or class is *defined*, as opposed to merely mentioned.

    A plain text search for a common name returns every call site. Matching definition
    keywords narrows it to the declaration, which is almost always what is wanted when a
    stack trace names a symbol.
    """
    root = context.workspace.resolve()
    escaped = re.escape(args.symbol)

    definition = re.compile(
        rf"(?:def|class|function|func|interface|type|struct|trait|impl)\s+{escaped}\b"
        rf"|(?:const|let|var)\s+{escaped}\s*[=:]"
        rf"|{escaped}\s*[:=]\s*(?:async\s*)?(?:function|\()",
    )

    findings: list[str] = []

    for path in _iter_source_files(root):
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for number, line in enumerate(content.split("\n"), start=1):
            if definition.search(line):
                relative = path.relative_to(root).as_posix()
                findings.append(f"{relative}:{number}: {line.strip()[:200]}")

    if not findings:
        return ToolResult.success(
            f"No definition found for {args.symbol!r}. It may be imported or dynamic.",
            match_count=0,
        )

    return ToolResult.success(
        wrap_untrusted(f"definitions of {args.symbol}", "\n".join(findings[:MAX_SEARCH_MATCHES])),
        match_count=len(findings),
    )


async def repository_tree(context: ToolContext, args: TreeArgs) -> ToolResult:
    """A depth-limited directory tree, for orientation rather than exhaustive listing."""
    root = resolve_in_workspace(context.workspace, args.path)

    if not root.is_dir():
        return ToolResult.failure("NOT_A_DIRECTORY", f"Not a directory: {args.path}")

    lines: list[str] = []

    def walk(directory: Path, depth: int, prefix: str) -> None:
        if depth > args.max_depth or len(lines) >= MAX_TREE_ENTRIES:
            return

        children = [
            child
            for child in sorted(directory.iterdir(), key=lambda item: (item.is_file(), item.name))
            if child.name not in EXCLUDED_DIRECTORIES and not child.name.startswith(".")
        ]

        for child in children:
            if len(lines) >= MAX_TREE_ENTRIES:
                lines.append("... truncated")
                return

            lines.append(f"{prefix}{child.name}{'/' if child.is_dir() else ''}")

            if child.is_dir():
                walk(child, depth + 1, prefix + "  ")

    walk(root, 1, "")

    return ToolResult.success(
        wrap_untrusted(f"tree of {args.path}", "\n".join(lines)), entry_count=len(lines)
    )


def register_filesystem_tools(registry: ToolRegistry) -> None:
    """Adds the read-only repository tools to a registry."""
    registry.register(
        Tool(
            name="list_directory",
            description=(
                "List files and folders in a repository directory. Dependency and build "
                "directories are omitted."
            ),
            arguments=ListDirectoryArgs,
            handler=list_directory,
        )
    )
    registry.register(
        Tool(
            name="read_file",
            description=(
                "Read a file from the repository, optionally a line range. Output is "
                "line-numbered. Use this to verify retrieved code before editing it."
            ),
            arguments=ReadFileArgs,
            handler=read_file,
        )
    )
    registry.register(
        Tool(
            name="search_code",
            description=(
                "Search repository files for text or a regular expression. Returns "
                "file:line matches. Good for error messages and exact strings."
            ),
            arguments=SearchCodeArgs,
            handler=search_code,
            timeout_seconds=45,
        )
    )
    registry.register(
        Tool(
            name="find_symbol",
            description=(
                "Find where a function, class or constant is defined, ignoring call sites. "
                "Use this when a stack trace names a symbol."
            ),
            arguments=FindSymbolArgs,
            handler=find_symbol,
            timeout_seconds=45,
        )
    )
    registry.register(
        Tool(
            name="repository_tree",
            description="Show a depth-limited directory tree, for orienting in a new repository.",
            arguments=TreeArgs,
            handler=repository_tree,
        )
    )
