"""Repository map: what kind of project is this, and how do I test it?

The agent needs this before it can do anything useful. Knowing the test command is what
turns "the model thinks this is fixed" into "the test suite says this is fixed", which is
the only objective signal the loop has.

Detection reads manifest files rather than asking a language model. It is deterministic,
free, instant, and cannot hallucinate a test command that does not exist.
"""

import json
import logging
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.rag.workspace import ScannedFile

logger = logging.getLogger(__name__)


@dataclass
class RepositoryMap:
    languages: dict[str, int] = field(default_factory=dict)
    primary_language: str | None = None
    package_managers: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    test_framework: str | None = None
    test_command: str | None = None
    lint_command: str | None = None
    typecheck_command: str | None = None
    build_command: str | None = None
    install_command: str | None = None
    entry_points: list[str] = field(default_factory=list)
    test_paths: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    file_count: int = 0
    test_file_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


#: Import or dependency name → framework label.
FRAMEWORK_MARKERS: dict[str, str] = {
    "fastapi": "FastAPI",
    "flask": "Flask",
    "django": "Django",
    "starlette": "Starlette",
    "sqlalchemy": "SQLAlchemy",
    "pydantic": "Pydantic",
    "react": "React",
    "next": "Next.js",
    "vue": "Vue",
    "svelte": "Svelte",
    "express": "Express",
    "nestjs": "NestJS",
    "@nestjs/core": "NestJS",
    "vite": "Vite",
    "webpack": "webpack",
    "tailwindcss": "Tailwind CSS",
    "prisma": "Prisma",
    "gin-gonic/gin": "Gin",
    "spring-boot": "Spring Boot",
    "rails": "Rails",
    "laravel": "Laravel",
}

#: Test dependency → (framework label, command).
PYTHON_TEST_RUNNERS: dict[str, tuple[str, str]] = {
    "pytest": ("pytest", "pytest -q"),
    "nose2": ("nose2", "nose2"),
}

JS_TEST_RUNNERS: dict[str, tuple[str, str]] = {
    "vitest": ("vitest", "npx vitest run"),
    "jest": ("jest", "npx jest --ci"),
    "mocha": ("mocha", "npx mocha"),
    "@playwright/test": ("playwright", "npx playwright test"),
}


def _detect_languages(files: list[ScannedFile]) -> tuple[dict[str, int], str | None]:
    """Counts files per language.

    Counting files rather than bytes avoids one enormous generated file deciding that a
    Python project is really JSON.
    """
    counter = Counter(item.language for item in files if item.language)

    if not counter:
        return {}, None

    return dict(counter.most_common()), counter.most_common(1)[0][0]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not parse %s: %s", path.name, type(exc).__name__)
        return {}


def _read_toml(path: Path) -> dict:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("could not parse %s: %s", path.name, type(exc).__name__)
        return {}


def _inspect_package_json(root: Path, result: RepositoryMap) -> None:
    manifest_path = root / "package.json"
    if not manifest_path.is_file():
        return

    manifest = _read_json(manifest_path)
    if not manifest:
        return

    result.package_managers.append(
        "pnpm"
        if (root / "pnpm-lock.yaml").is_file()
        else "yarn"
        if (root / "yarn.lock").is_file()
        else "npm"
    )
    result.config_files.append("package.json")
    result.install_command = f"{result.package_managers[-1]} install"

    dependencies = {
        **manifest.get("dependencies", {}),
        **manifest.get("devDependencies", {}),
    }

    for name in dependencies:
        label = FRAMEWORK_MARKERS.get(name)
        if label and label not in result.frameworks:
            result.frameworks.append(label)

    scripts = manifest.get("scripts", {})

    # A declared script is more reliable than a guessed command: it encodes the
    # project's own conventions, including any required flags.
    if "test" in scripts:
        result.test_command = "npm test -- --run" if "vitest" in dependencies else "npm test"
    if "lint" in scripts:
        result.lint_command = "npm run lint"
    if "typecheck" in scripts:
        result.typecheck_command = "npm run typecheck"
    elif (root / "tsconfig.json").is_file():
        result.typecheck_command = "npx tsc --noEmit"
    if "build" in scripts:
        result.build_command = "npm run build"

    for name, (label, command) in JS_TEST_RUNNERS.items():
        if name in dependencies:
            result.test_framework = label
            result.test_command = result.test_command or command
            break

    for candidate in ("src/main.tsx", "src/main.ts", "src/index.tsx", "src/index.ts", "index.js"):
        if (root / candidate).is_file():
            result.entry_points.append(candidate)


def _inspect_python(root: Path, result: RepositoryMap) -> None:
    pyproject = root / "pyproject.toml"
    requirements = root / "requirements.txt"

    dependency_text = ""

    if pyproject.is_file():
        result.config_files.append("pyproject.toml")
        data = _read_toml(pyproject)

        project = data.get("project", {})
        dependency_text = " ".join(
            [
                *project.get("dependencies", []),
                *[
                    entry
                    for group in project.get("optional-dependencies", {}).values()
                    for entry in group
                ],
            ]
        ).lower()

        if "poetry" in data.get("tool", {}):
            result.package_managers.append("poetry")
            result.install_command = "poetry install"
        else:
            result.package_managers.append("pip")
            result.install_command = "pip install -e ."

        tools = data.get("tool", {})
        if "pytest" in tools or "pytest.ini_options" in str(tools):
            result.test_framework = "pytest"
            result.test_command = "pytest -q"
        if "ruff" in tools:
            result.lint_command = "ruff check ."
        if "mypy" in tools:
            result.typecheck_command = "mypy ."

    if requirements.is_file():
        result.config_files.append("requirements.txt")
        dependency_text += " " + requirements.read_text(
            encoding="utf-8", errors="replace"
        ).lower()
        if "pip" not in result.package_managers:
            result.package_managers.append("pip")
            result.install_command = "pip install -r requirements.txt"

    if not dependency_text:
        return

    for marker, label in FRAMEWORK_MARKERS.items():
        if marker in dependency_text and label not in result.frameworks:
            result.frameworks.append(label)

    for name, (label, command) in PYTHON_TEST_RUNNERS.items():
        if name in dependency_text:
            result.test_framework = result.test_framework or label
            result.test_command = result.test_command or command
            break

    if (root / "pytest.ini").is_file() or (root / "tests").is_dir():
        result.test_framework = result.test_framework or "pytest"
        result.test_command = result.test_command or "pytest -q"

    for candidate in ("app/main.py", "main.py", "manage.py", "src/main.py", "wsgi.py"):
        if (root / candidate).is_file():
            result.entry_points.append(candidate)


def _inspect_other_manifests(root: Path, result: RepositoryMap) -> None:
    if (root / "go.mod").is_file():
        result.config_files.append("go.mod")
        result.package_managers.append("go modules")
        result.install_command = "go mod download"
        result.test_command = result.test_command or "go test ./..."
        result.build_command = result.build_command or "go build ./..."
        result.test_framework = result.test_framework or "go test"

    if (root / "Cargo.toml").is_file():
        result.config_files.append("Cargo.toml")
        result.package_managers.append("cargo")
        result.install_command = "cargo fetch"
        result.test_command = result.test_command or "cargo test"
        result.build_command = result.build_command or "cargo build"
        result.test_framework = result.test_framework or "cargo test"


def build(root: Path, files: list[ScannedFile]) -> RepositoryMap:
    """Builds the repository map from manifests plus the scanned file list."""
    result = RepositoryMap()

    languages, primary = _detect_languages(files)
    result.languages = languages
    result.primary_language = primary
    result.file_count = len(files)
    result.test_file_count = sum(1 for item in files if item.is_test)

    _inspect_package_json(root, result)
    _inspect_python(root, result)
    _inspect_other_manifests(root, result)

    # Distinct top-level directories that hold tests, so the agent knows where to look.
    test_directories = {
        item.relative_path.split("/")[0] for item in files if item.is_test and "/" in
        item.relative_path
    }
    result.test_paths = sorted(test_directories)

    logger.info(
        "repository map: language=%s frameworks=%s test=%s",
        result.primary_language,
        result.frameworks,
        result.test_command,
    )
    return result
