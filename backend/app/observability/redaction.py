"""Secret redaction for logs.

This platform holds GitHub write tokens and model API keys. A single unredacted log
line is a credential leak, so redaction is applied centrally in the logging pipeline
rather than trusted to every call site.
"""

import logging
import re

_PATTERNS: tuple[re.Pattern[str], ...] = (
    # GitHub tokens: classic, fine-grained, OAuth, app, refresh
    re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    # Bearer / token headers
    re.compile(r"(?i)(authorization\s*[:=]\s*)(bearer\s+)?\S+"),
    # Common provider key shapes
    re.compile(r"sk-[A-Za-z0-9\-_]{16,}"),
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
    # Google AI Studio also issues keys prefixed "AQ." which the AIza pattern misses.
    re.compile(r"\bAQ\.[A-Za-z0-9\-_]{20,}"),
    # key=value style secrets in free text
    re.compile(
        r"(?i)\b(api[_-]?key|secret|password|passwd|token|private[_-]?key)\b"
        r"(\s*[:=]\s*)(\"|')?[^\s\"',;]+"
    ),
    # Postgres URLs with inline credentials
    re.compile(r"(?i)(postgres(?:ql)?(?:\+\w+)?://)[^:\s]+:[^@\s]+@"),
)

REDACTED = "[REDACTED]"


def redact(text: str) -> str:
    """Replaces anything that looks like a credential with a placeholder."""
    if not text:
        return text

    result = text
    # Simple whole-match replacements.
    for index in (0, 1, 3, 4, 5):
        result = _PATTERNS[index].sub(REDACTED, result)

    # Patterns that keep a prefix so the log still says *what* was redacted.
    result = _PATTERNS[2].sub(lambda m: f"{m.group(1)}{REDACTED}", result)
    result = _PATTERNS[6].sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", result)
    result = _PATTERNS[7].sub(lambda m: f"{m.group(1)}{REDACTED}:{REDACTED}@", result)
    return result


class RedactingFilter(logging.Filter):
    """Applies redaction to the formatted message and to string args."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)

        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    key: redact(value) if isinstance(value, str) else value
                    for key, value in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    redact(value) if isinstance(value, str) else value for value in record.args
                )

        return True
