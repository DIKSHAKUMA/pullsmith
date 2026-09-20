"""Throwaway pre-push secret scan.

Runs over exactly the files git would commit, not the whole working tree, because the whole
point is to catch something that slipped past .gitignore.
"""

import re
import subprocess
import sys
from pathlib import Path

PATTERNS = {
    "GitHub PAT (classic)": re.compile(r"ghp_[A-Za-z0-9]{30,}"),
    "GitHub PAT (fine-grained)": re.compile(r"github_pat_[A-Za-z0-9_]{50,}"),
    "GitHub OAuth token": re.compile(r"gho_[A-Za-z0-9]{30,}"),
    "Gemini key (AIza)": re.compile(r"AIza[A-Za-z0-9_\-]{30,}"),
    "Gemini key (AQ.)": re.compile(r"\bAQ\.[A-Za-z0-9_\-]{20,}"),
    "Neon password": re.compile(r"npg_[A-Za-z0-9]{10,}"),
    "Postgres URL with password": re.compile(r"postgres(?:ql)?(?:\+\w+)?://[^:\s]+:[^@\s]+@"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "Private key block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "Fernet-looking key assignment": re.compile(
        r"TOKEN_ENCRYPTION_KEY\s*=\s*[A-Za-z0-9_\-]{40,}"
    ),
}

#: Strings that legitimately look like secrets: placeholders and test fixtures.
ALLOWED = (
    "change-me",
    "your_pat",
    "github_pat_...",
    "placeholder-not-a-real-token",
    "user:password@",
    "user:pass@",
    "u:p@",
    "neondb_owner:npg_secret@",
    "AQ.Fake000Example",
    "ghp_abcdefghijklmnopqrstuvwxyz1234",
    "sk-abcdefghijklmnopqrstuvwx",
    "ghp_fake_token_value_1234567890",
    "ghp_a_valid_looking_token",
    "ghp_not_a_real_token",
    "ghp_worker_token_000",
    "ghp_secret_value_0987654321",
    "gho_exchanged",
    "u:p@host",
)


def tracked_files() -> list[Path]:
    """Files git would actually put in the commit."""
    result = subprocess.run(  # noqa: S603
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],  # noqa: S607
        capture_output=True,
        check=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    findings: list[str] = []
    files = tracked_files()

    for path in files:
        if not path.is_file():
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                hit = match.group(0)

                if any(allowed in hit for allowed in ALLOWED):
                    continue

                line = text[: match.start()].count("\n") + 1
                findings.append(f"{path}:{line}  {label}  ->  {hit[:22]}...")

    print(f"scanned {len(files)} file(s) git would commit")

    if findings:
        print(f"\nFOUND {len(findings)} possible secret(s). Do not push:\n")
        for finding in findings:
            print(f"  {finding}")
        return 1

    print("no secrets found in committable files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
