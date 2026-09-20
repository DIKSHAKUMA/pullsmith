"""Write tools. Every tool here is marked ``mutating`` and is refused in read-only phases.

Two decisions shape this module.

**Targeted replacement over whole-file rewrites.** ``replace_in_file`` requires the exact
existing text and refuses when it appears zero times or more than once. That forces the agent
to have actually read the file, and it makes an accidental mass rewrite impossible. A model
asked to "change one line" will often regenerate the whole file from memory, silently dropping
anything it did not remember - this is the guard against that.

**Every change is recorded.** Each edit appends to ``ToolContext.changes`` so the diff shown
to a human reviewer is built from what was actually written, not from what the model claimed.
"""

import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.agent.safety import (
    MAX_READ_BYTES,
    assert_readable,
    is_sensitive_path,
    resolve_in_workspace,
)
from app.agent.tools.base import Tool, ToolContext, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

#: A single edit larger than this is almost certainly a regeneration rather than a fix.
MAX_WRITE_BYTES = 100_000


@dataclass
class FileChange:
    """One recorded modification, used to build the review diff."""

    path: str
    action: str
    lines_added: int = 0
    lines_removed: int = 0
    is_sensitive: bool = False
    original_content: str | None = field(default=None, repr=False)


class WriteFileArgs(BaseModel):
    path: str = Field(max_length=1024, description="File path relative to the repository root")
    content: str = Field(description="Complete new contents of the file")
    reason: str = Field(max_length=300, description="Why this file is being written")


class ReplaceInFileArgs(BaseModel):
    path: str = Field(max_length=1024)
    find: str = Field(
        min_length=1,
        description=(
            "Exact existing text to replace. Must appear exactly once in the file. "
            "Include surrounding lines if needed to make it unique."
        ),
    )
    replace: str = Field(description="Replacement text")
    reason: str = Field(max_length=300)


class DeleteFileArgs(BaseModel):
    path: str = Field(max_length=1024)
    reason: str = Field(max_length=300)


def _record(context: ToolContext, change: FileChange) -> None:
    changes = getattr(context, "changes", None)

    if changes is None:
        # Older contexts may not carry the list; attach one rather than losing the record.
        context.changes = [change]
    else:
        changes.append(change)


async def write_file(context: ToolContext, args: WriteFileArgs) -> ToolResult:
    """Creates a file, or replaces one entirely.

    Whole-file writes are allowed because new files need them, but ``replace_in_file`` is
    preferred for edits and the tool description says so.
    """
    assert_readable(args.path)
    target = resolve_in_workspace(context.workspace, args.path)

    if len(args.content.encode("utf-8")) > MAX_WRITE_BYTES:
        return ToolResult.failure(
            "CONTENT_TOO_LARGE", f"Refusing to write more than {MAX_WRITE_BYTES} bytes at once"
        )

    existed = target.is_file()
    original = target.read_text(encoding="utf-8", errors="replace") if existed else None

    if existed and len(original or "") > MAX_READ_BYTES:
        return ToolResult.failure(
            "FILE_TOO_LARGE", f"{args.path} is too large to rewrite; use replace_in_file"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(args.content, encoding="utf-8", newline="\n")

    removed = len((original or "").splitlines()) if existed else 0
    added = len(args.content.splitlines())
    sensitive = is_sensitive_path(args.path)

    _record(
        context,
        FileChange(
            path=args.path,
            action="modified" if existed else "created",
            lines_added=added,
            lines_removed=removed,
            is_sensitive=sensitive,
            original_content=original,
        ),
    )

    logger.info("%s %s (+%s/-%s)", "modified" if existed else "created", args.path, added, removed)

    warning = " NOTE: this path is flagged for human review." if sensitive else ""

    return ToolResult.success(
        f"{'Modified' if existed else 'Created'} {args.path} "
        f"({added} lines written, {removed} replaced).{warning}",
        path=args.path,
        created=not existed,
        is_sensitive=sensitive,
    )


async def replace_in_file(context: ToolContext, args: ReplaceInFileArgs) -> ToolResult:
    """Replaces one exact occurrence of text. The preferred way to edit.

    Refusing on zero or multiple matches is the point: it proves the agent read the real file,
    and it prevents a one-line intention from becoming a many-line accident.
    """
    assert_readable(args.path)
    target = resolve_in_workspace(context.workspace, args.path)

    if not target.is_file():
        return ToolResult.failure("NOT_FOUND", f"No such file: {args.path}")

    original = target.read_text(encoding="utf-8", errors="replace")
    occurrences = original.count(args.find)

    if occurrences == 0:
        # Usually means the agent is working from a stale retrieved snippet.
        return ToolResult.failure(
            "TEXT_NOT_FOUND",
            f"The text to replace was not found in {args.path}. "
            f"Read the file again - it may differ from what you expected.",
        )

    if occurrences > 1:
        return ToolResult.failure(
            "TEXT_NOT_UNIQUE",
            f"The text appears {occurrences} times in {args.path}. "
            f"Include more surrounding context so exactly one location matches.",
        )

    updated = original.replace(args.find, args.replace, 1)
    target.write_text(updated, encoding="utf-8", newline="\n")

    added = len(args.replace.splitlines())
    removed = len(args.find.splitlines())
    sensitive = is_sensitive_path(args.path)

    _record(
        context,
        FileChange(
            path=args.path,
            action="modified",
            lines_added=added,
            lines_removed=removed,
            is_sensitive=sensitive,
            original_content=original,
        ),
    )

    logger.info("edited %s (+%s/-%s)", args.path, added, removed)

    return ToolResult.success(
        f"Replaced {removed} line(s) with {added} in {args.path}.",
        path=args.path,
        is_sensitive=sensitive,
    )


async def delete_file(context: ToolContext, args: DeleteFileArgs) -> ToolResult:
    """Deletes a file.

    Deliberately narrow: one file, no directories, no globs. Recursive deletion is the kind
    of capability that turns a confused agent into an incident.
    """
    assert_readable(args.path)
    target = resolve_in_workspace(context.workspace, args.path)

    if not target.exists():
        return ToolResult.failure("NOT_FOUND", f"No such file: {args.path}")

    if target.is_dir():
        return ToolResult.failure(
            "IS_A_DIRECTORY", "Directories cannot be deleted; delete individual files"
        )

    original = target.read_text(encoding="utf-8", errors="replace")
    target.unlink()

    _record(
        context,
        FileChange(
            path=args.path,
            action="deleted",
            lines_removed=len(original.splitlines()),
            is_sensitive=is_sensitive_path(args.path),
            original_content=original,
        ),
    )

    logger.info("deleted %s", args.path)

    return ToolResult.success(f"Deleted {args.path}.", path=args.path)


def summarise_changes(changes: list[FileChange]) -> str:
    """A short operational summary of what was written, for the timeline."""
    if not changes:
        return "No files changed."

    # One file edited twice is one changed file, not two.
    by_path: dict[str, FileChange] = {}
    for change in changes:
        by_path[change.path] = change

    added = sum(change.lines_added for change in by_path.values())
    removed = sum(change.lines_removed for change in by_path.values())
    flagged = [change.path for change in by_path.values() if change.is_sensitive]

    summary = f"{len(by_path)} file(s) changed, +{added}/-{removed} lines"

    if flagged:
        summary += f". Flagged for review: {', '.join(flagged)}"

    return summary


def register_editing_tools(registry: ToolRegistry) -> None:
    registry.register(
        Tool(
            name="replace_in_file",
            description=(
                "Replace an exact snippet of text in a file. The snippet must appear exactly "
                "once. This is the preferred way to edit existing code, because it changes "
                "only what you name."
            ),
            arguments=ReplaceInFileArgs,
            handler=replace_in_file,
            mutating=True,
        )
    )
    registry.register(
        Tool(
            name="write_file",
            description=(
                "Write the complete contents of a file, creating it if needed. Use this for "
                "new files. For editing existing code prefer replace_in_file."
            ),
            arguments=WriteFileArgs,
            handler=write_file,
            mutating=True,
        )
    )
    registry.register(
        Tool(
            name="delete_file",
            description="Delete a single file. Directories cannot be deleted.",
            arguments=DeleteFileArgs,
            handler=delete_file,
            mutating=True,
        )
    )
