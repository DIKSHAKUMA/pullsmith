"""Building the diff a human reviews.

The diff is produced from the original content captured *before* each write, compared against
what is on disk now. It is never assembled from the model's description of what it did: an
agent that says "I made a small change" and actually rewrote two hundred lines must still show
two hundred lines.

Deliberately uses Python's ``difflib`` rather than shelling out to ``git diff``. The changes may
not be committed yet, and reading the workspace directly avoids depending on repository state.
"""

import difflib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.agent.safety import is_sensitive_path
from app.agent.tools.editing import FileChange

logger = logging.getLogger(__name__)

#: Beyond this a diff stops being reviewable and becomes a wall of text. The reviewer is told
#: it was trimmed rather than being shown a partial diff silently.
MAX_DIFF_LINES = 2_000
MAX_LINES_PER_FILE = 400


@dataclass
class FileDiff:
    path: str
    action: str
    lines_added: int
    lines_removed: int
    is_sensitive: bool
    diff_text: str
    truncated: bool = False


@dataclass
class ChangeSet:
    """Everything that changed in a run, as the reviewer will see it."""

    files: list[FileDiff] = field(default_factory=list)
    truncated: bool = False

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def lines_added(self) -> int:
        return sum(item.lines_added for item in self.files)

    @property
    def lines_removed(self) -> int:
        return sum(item.lines_removed for item in self.files)

    @property
    def sensitive_paths(self) -> list[str]:
        return [item.path for item in self.files if item.is_sensitive]

    def unified_diff(self) -> str:
        return "\n".join(item.diff_text for item in self.files)

    def summary(self) -> str:
        if not self.files:
            return "No changes."

        parts = [f"{self.file_count} file(s), +{self.lines_added}/-{self.lines_removed}"]

        if self.sensitive_paths:
            parts.append(f"flagged: {', '.join(self.sensitive_paths)}")

        return ". ".join(parts)


def _collapse(changes: list[FileChange]) -> dict[str, FileChange]:
    """One entry per path, keeping the *earliest* original content.

    A file edited three times is one changed file, and the diff must be against how it looked
    before the agent touched it - not before its last edit.
    """
    collapsed: dict[str, FileChange] = {}

    for change in changes:
        existing = collapsed.get(change.path)

        if existing is None:
            collapsed[change.path] = change
            continue

        # Keep the first-seen original, but let the latest action win (a create then delete
        # is a delete).
        collapsed[change.path] = FileChange(
            path=change.path,
            action=change.action,
            lines_added=change.lines_added,
            lines_removed=change.lines_removed,
            is_sensitive=change.is_sensitive or existing.is_sensitive,
            original_content=existing.original_content,
        )

    return collapsed


def _diff_for(path: str, before: str, after: str) -> tuple[str, int, int, bool]:
    """Unified diff plus accurate added/removed counts."""
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)

    lines = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        )
    )

    added = sum(
        1 for line in lines if line.startswith("+") and not line.startswith("+++")
    )
    removed = sum(
        1 for line in lines if line.startswith("-") and not line.startswith("---")
    )

    truncated = len(lines) > MAX_LINES_PER_FILE

    if truncated:
        lines = lines[:MAX_LINES_PER_FILE]
        lines.append(f"... diff truncated at {MAX_LINES_PER_FILE} lines\n")

    return "".join(lines).rstrip("\n"), added, removed, truncated


def build_change_set(workspace: Path, changes: list[FileChange]) -> ChangeSet:
    """Compares recorded originals against the current workspace."""
    change_set = ChangeSet()

    for path, change in sorted(_collapse(changes).items()):
        before = change.original_content or ""
        target = workspace / path

        after = (
            target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
        )

        if before == after:
            # The agent wrote the file back unchanged. Not worth a reviewer's attention.
            continue

        diff_text, added, removed, truncated = _diff_for(path, before, after)

        change_set.files.append(
            FileDiff(
                path=path,
                action=change.action,
                lines_added=added,
                lines_removed=removed,
                is_sensitive=change.is_sensitive or is_sensitive_path(path),
                diff_text=diff_text,
                truncated=truncated,
            )
        )

    total_lines = sum(item.diff_text.count("\n") for item in change_set.files)

    if total_lines > MAX_DIFF_LINES:
        change_set.truncated = True
        logger.warning("change set is large: %s diff lines", total_lines)

    return change_set
