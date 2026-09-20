"""Which files are worth indexing.

Indexing the wrong files is expensive twice over: embeddings cost money and irrelevant
chunks crowd out relevant ones at retrieval time. A repository's real source code is
usually a small fraction of the files on disk.
"""

from pathlib import PurePosixPath

#: Directories never worth indexing. Matched on any path segment.
EXCLUDED_DIRECTORIES: frozenset[str] = frozenset(
    {
        # dependencies (installed code, not this repository's code)
        "node_modules",
        "bower_components",
        "vendor",
        "venv",
        ".venv",
        "env",
        "site-packages",
        # version control and tooling metadata
        ".git",
        ".hg",
        ".svn",
        ".idea",
        ".vscode",
        # build output (generated from the source we already index)
        "dist",
        "build",
        "out",
        "target",
        ".next",
        ".nuxt",
        ".output",
        ".svelte-kit",
        # caches
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".cache",
        ".turbo",
        ".parcel-cache",
        # coverage and reports
        "coverage",
        "htmlcov",
        ".nyc_output",
        # migrations are generated and highly repetitive; they add noise
        "migrations",
        "alembic/versions",
    }
)

#: Extensions we can parse and that carry real logic.
SOURCE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".pyi",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".mts",
        ".cts",
        ".go",
        ".rs",
        ".java",
        ".rb",
        ".php",
        ".cs",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".cc",
        ".kt",
        ".swift",
        ".scala",
        ".sh",
        ".sql",
        ".vue",
        ".svelte",
    }
)

#: Config and docs. Indexed because issues often reference them, but they are not parsed
#: into symbols.
SUPPORTING_FILENAMES: frozenset[str] = frozenset(
    {
        "package.json",
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "setup.py",
        "setup.cfg",
        "go.mod",
        "cargo.toml",
        "pom.xml",
        "build.gradle",
        "gemfile",
        "composer.json",
        "dockerfile",
        "docker-compose.yml",
        "makefile",
        "readme.md",
        "tsconfig.json",
        "vite.config.ts",
        "next.config.js",
        "pytest.ini",
        "tox.ini",
        "jest.config.js",
        "vitest.config.ts",
    }
)

#: Generated files that happen to have source extensions.
GENERATED_SUFFIXES: tuple[str, ...] = (
    ".min.js",
    ".min.css",
    ".bundle.js",
    ".chunk.js",
    "-lock.json",
    ".lock",
    ".map",
    "_pb2.py",
    ".pb.go",
    ".g.dart",
    ".generated.ts",
    ".d.ts",  # type declarations: signatures only, no logic
)

#: Files above this size are almost always generated or vendored data.
MAX_FILE_BYTES = 400_000

#: Files with very long lines are almost always minified.
MAX_LINE_LENGTH = 5_000


def is_excluded_path(relative_path: str) -> bool:
    """True if any directory in the path is on the exclusion list."""
    parts = PurePosixPath(relative_path).parts

    if any(part in EXCLUDED_DIRECTORIES for part in parts):
        return True

    # Handles multi-segment entries such as "alembic/versions".
    joined = "/".join(parts)
    return any("/" in excluded and excluded in joined for excluded in EXCLUDED_DIRECTORIES)


def is_generated(relative_path: str) -> bool:
    lowered = relative_path.lower()
    return any(lowered.endswith(suffix) for suffix in GENERATED_SUFFIXES)


def is_source_file(relative_path: str) -> bool:
    return PurePosixPath(relative_path).suffix.lower() in SOURCE_EXTENSIONS


def is_supporting_file(relative_path: str) -> bool:
    return PurePosixPath(relative_path).name.lower() in SUPPORTING_FILENAMES


def should_index(relative_path: str, size_bytes: int) -> bool:
    """The single decision point for whether a file enters the index."""
    if is_excluded_path(relative_path) or is_generated(relative_path):
        return False

    if size_bytes > MAX_FILE_BYTES:
        return False

    return is_source_file(relative_path) or is_supporting_file(relative_path)


def looks_minified(content: str) -> bool:
    """Catches minified files that passed the name and size checks."""
    lines = content.split("\n", 200)[:200]
    return any(len(line) > MAX_LINE_LENGTH for line in lines)


def is_test_path(relative_path: str) -> bool:
    """Test files are indexed and flagged.

    The flag matters both ways: a bug fix usually needs the existing tests as context,
    but tests should not dominate retrieval when the question is about behaviour.
    """
    lowered = relative_path.lower()
    name = PurePosixPath(lowered).name

    if any(segment in ("test", "tests", "spec", "__tests__", "e2e") for segment in
           PurePosixPath(lowered).parts):
        return True

    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or ".test." in name
        or ".spec." in name
        or name.endswith("test.go")
    )
