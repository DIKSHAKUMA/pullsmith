"""Scanning, exclusion and repository-map tests.

Exclusions are worth testing carefully because a mistake is expensive in both directions:
indexing ``node_modules`` wastes embedding spend and floods retrieval with irrelevant
library code, while wrongly excluding a source directory makes the agent blind to it.
"""

import json
from pathlib import Path

import pytest

from app.rag import repo_map, workspace
from app.rag.exclusions import is_test_path, should_index


def write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """A small repository resembling a real full-stack project."""
    write(tmp_path, "app/main.py", "from fastapi import FastAPI\n\napp = FastAPI()\n")
    write(
        tmp_path,
        "app/services/profile.py",
        "class ProfileService:\n    def update(self, email):\n        return email\n",
    )
    write(tmp_path, "tests/test_profile.py", "def test_update():\n    assert True\n")
    write(tmp_path, "src/App.tsx", "export function App() {\n  return null\n}\n")

    write(
        tmp_path,
        "pyproject.toml",
        '[project]\nname = "demo"\ndependencies = ["fastapi", "sqlalchemy"]\n'
        '[project.optional-dependencies]\ndev = ["pytest", "ruff"]\n'
        "[tool.ruff]\nline-length = 100\n",
    )
    write(
        tmp_path,
        "package.json",
        json.dumps(
            {
                "name": "demo-ui",
                "scripts": {"test": "vitest", "build": "vite build", "lint": "eslint ."},
                "dependencies": {"react": "19.0.0"},
                "devDependencies": {"vitest": "3.0.0", "vite": "7.0.0"},
            }
        ),
    )
    write(tmp_path, "tsconfig.json", "{}")

    # Noise that must be excluded.
    write(tmp_path, "node_modules/react/index.js", "module.exports = {}\n")
    write(tmp_path, "dist/bundle.js", "console.log('built')\n")
    write(tmp_path, "app/__pycache__/main.cpython-312.pyc", "binary-ish")
    write(tmp_path, "coverage/report.js", "// coverage\n")
    write(tmp_path, "static/vendor.min.js", "var a=1;var b=2;\n")
    write(tmp_path, "app/schema_pb2.py", "# generated\n")

    return tmp_path


class TestExclusions:
    def test_source_files_are_indexed(self) -> None:
        assert should_index("app/main.py", 1_000)
        assert should_index("src/App.tsx", 1_000)
        assert should_index("cmd/server/main.go", 1_000)

    def test_dependency_and_build_directories_are_skipped(self) -> None:
        assert not should_index("node_modules/react/index.js", 1_000)
        assert not should_index("dist/bundle.js", 1_000)
        assert not should_index(".next/server/page.js", 1_000)
        assert not should_index("venv/lib/site-packages/x.py", 1_000)

    def test_generated_files_are_skipped(self) -> None:
        assert not should_index("static/app.min.js", 1_000)
        assert not should_index("app/schema_pb2.py", 1_000)
        assert not should_index("package-lock.json", 1_000)
        # Type declarations carry signatures but no logic.
        assert not should_index("types/index.d.ts", 1_000)

    def test_huge_files_are_skipped(self) -> None:
        """Anything this large is almost certainly generated or vendored data."""
        assert not should_index("app/data.py", 900_000)

    def test_config_and_docs_are_indexed(self) -> None:
        # Issues frequently reference these, so they must be searchable.
        assert should_index("package.json", 500)
        assert should_index("pyproject.toml", 500)
        assert should_index("README.md", 500)

    def test_unrelated_files_are_skipped(self) -> None:
        assert not should_index("logo.png", 500)
        assert not should_index("notes.txt", 500)

    def test_test_paths_are_recognised(self) -> None:
        assert is_test_path("tests/test_profile.py")
        assert is_test_path("app/profile_test.go")
        assert is_test_path("src/App.test.tsx")
        assert is_test_path("src/__tests__/App.tsx")
        assert not is_test_path("app/services/profile.py")


class TestScan:
    def test_scan_finds_source_and_skips_noise(self, sample_repo: Path) -> None:
        found = {item.relative_path for item in workspace.scan(sample_repo)}

        assert "app/main.py" in found
        assert "app/services/profile.py" in found
        assert "src/App.tsx" in found
        assert "package.json" in found

        assert not any(path.startswith("node_modules/") for path in found)
        assert not any(path.startswith("dist/") for path in found)
        assert not any("__pycache__" in path for path in found)
        assert "static/vendor.min.js" not in found
        assert "app/schema_pb2.py" not in found

    def test_scan_detects_language_and_flags_tests(self, sample_repo: Path) -> None:
        by_path = {item.relative_path: item for item in workspace.scan(sample_repo)}

        assert by_path["app/main.py"].language == "python"
        assert by_path["src/App.tsx"].language == "tsx"
        assert by_path["tests/test_profile.py"].is_test
        assert not by_path["app/main.py"].is_test

    def test_scan_records_a_content_hash(self, sample_repo: Path) -> None:
        first = {item.relative_path: item.content_hash for item in workspace.scan(sample_repo)}
        second = {item.relative_path: item.content_hash for item in workspace.scan(sample_repo)}

        # Stable across runs, which is what makes it usable as a change detector.
        assert first == second
        assert len(first["app/main.py"]) == 64

    def test_scan_results_are_ordered(self, sample_repo: Path) -> None:
        paths = [item.relative_path for item in workspace.scan(sample_repo)]
        assert paths == sorted(paths)

    def test_binary_files_are_skipped_not_crashed_on(self, tmp_path: Path) -> None:
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "blob.py").write_bytes(b"\x00\x01\x02binary")

        assert workspace.scan(tmp_path) == []


class TestIncrementalIndexing:
    def test_only_changed_files_are_reported(self, sample_repo: Path) -> None:
        first = workspace.scan(sample_repo)
        previous = {item.relative_path: item.content_hash for item in first}

        write(sample_repo, "app/main.py", "from fastapi import FastAPI\n\napp = FastAPI(1)\n")

        modified, unchanged, deleted = workspace.changed_files(
            previous, workspace.scan(sample_repo)
        )

        assert [item.relative_path for item in modified] == ["app/main.py"]
        assert "app/services/profile.py" in unchanged
        assert deleted == []

    def test_unchanged_repository_reports_no_work(self, sample_repo: Path) -> None:
        """The point of hashing: re-indexing an untouched repo must re-embed nothing."""
        first = workspace.scan(sample_repo)
        previous = {item.relative_path: item.content_hash for item in first}

        modified, unchanged, deleted = workspace.changed_files(
            previous, workspace.scan(sample_repo)
        )

        assert modified == []
        assert len(unchanged) == len(first)
        assert deleted == []

    def test_deleted_files_are_reported(self, sample_repo: Path) -> None:
        scanned = workspace.scan(sample_repo)
        previous = {item.relative_path: item.content_hash for item in scanned}
        (sample_repo / "app" / "services" / "profile.py").unlink()

        _, _, deleted = workspace.changed_files(previous, workspace.scan(sample_repo))

        assert deleted == ["app/services/profile.py"]


class TestRepositoryMap:
    def test_detects_languages_and_primary(self, sample_repo: Path) -> None:
        result = repo_map.build(sample_repo, workspace.scan(sample_repo))

        assert result.primary_language == "python"
        assert "tsx" in result.languages

    def test_detects_frameworks_from_manifests(self, sample_repo: Path) -> None:
        result = repo_map.build(sample_repo, workspace.scan(sample_repo))

        assert "FastAPI" in result.frameworks
        assert "React" in result.frameworks

    def test_detects_test_command(self, sample_repo: Path) -> None:
        """The test command is the agent's only objective success signal."""
        result = repo_map.build(sample_repo, workspace.scan(sample_repo))

        assert result.test_command is not None
        assert result.test_framework is not None

    def test_detects_supporting_commands(self, sample_repo: Path) -> None:
        result = repo_map.build(sample_repo, workspace.scan(sample_repo))

        assert result.lint_command is not None
        assert result.typecheck_command == "npx tsc --noEmit"
        assert result.build_command == "npm run build"
        assert result.install_command is not None

    def test_finds_entry_points_and_counts_tests(self, sample_repo: Path) -> None:
        result = repo_map.build(sample_repo, workspace.scan(sample_repo))

        assert "app/main.py" in result.entry_points
        assert result.test_file_count == 1
        assert "tests" in result.test_paths

    def test_go_project_detection(self, tmp_path: Path) -> None:
        write(tmp_path, "go.mod", "module demo\n\ngo 1.22\n")
        write(tmp_path, "main.go", "package main\n\nfunc main() {}\n")

        result = repo_map.build(tmp_path, workspace.scan(tmp_path))

        assert result.test_command == "go test ./..."
        assert result.primary_language == "go"

    def test_empty_repository_does_not_crash(self, tmp_path: Path) -> None:
        result = repo_map.build(tmp_path, workspace.scan(tmp_path))

        assert result.primary_language is None
        assert result.file_count == 0

    def test_map_serialises_for_storage(self, sample_repo: Path) -> None:
        """The map is persisted as JSON on the snapshot row."""
        result = repo_map.build(sample_repo, workspace.scan(sample_repo))

        payload = result.to_dict()

        assert json.loads(json.dumps(payload))["primary_language"] == "python"
