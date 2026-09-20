"""Scans and chunks a local directory, printing what the indexer would store.

A development tool for inspecting chunk quality before any embedding spend. Point it at
a repository and read the output: if the chunks look wrong here, retrieval will be wrong
later, and no amount of embedding tuning will fix it.

Usage:  python -m scripts.index_preview <path> [--show N]
"""

import argparse
from collections import Counter
from pathlib import Path

from app.rag import repo_map, workspace
from app.rag.chunk import chunk_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--show", type=int, default=8, help="sample chunks to print")
    args = parser.parse_args()

    root: Path = args.path.resolve()

    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    files = workspace.scan(root)
    mapping = repo_map.build(root, files)

    print(f"\nrepository: {root.name}")
    print(f"  primary language : {mapping.primary_language}")
    print(f"  languages        : {mapping.languages}")
    print(f"  frameworks       : {', '.join(mapping.frameworks) or '-'}")
    print(f"  test framework   : {mapping.test_framework}")
    print(f"  test command     : {mapping.test_command}")
    print(f"  lint command     : {mapping.lint_command}")
    print(f"  entry points     : {', '.join(mapping.entry_points) or '-'}")
    print(f"  indexable files  : {mapping.file_count} ({mapping.test_file_count} test files)")

    chunks = []
    for item in files:
        data = workspace.read_bytes(item.absolute_path)
        if data is None:
            continue

        chunks.extend(
            chunk_file(
                content=data.decode("utf-8", errors="replace"),
                relative_path=item.relative_path,
                language=item.language,
                is_test=item.is_test,
            )
        )

    strategies = Counter(chunk.strategy for chunk in chunks)
    kinds = Counter(chunk.symbol_kind for chunk in chunks if chunk.symbol_kind)
    named = sum(1 for chunk in chunks if chunk.symbol)
    lines = [chunk.line_count for chunk in chunks] or [0]

    print(f"\nchunks: {len(chunks)}")
    print(f"  named symbols    : {named} ({named * 100 // max(len(chunks), 1)}%)")
    print(f"  strategies       : {dict(strategies)}")
    print(f"  kinds            : {dict(kinds.most_common(8))}")
    average = sum(lines) // len(lines)
    print(f"  lines per chunk  : min {min(lines)} / avg {average} / max {max(lines)}")

    print(f"\nsample of {args.show}:")
    for chunk in chunks[: args.show]:
        label = chunk.qualified_name()
        print(
            f"  {chunk.relative_path}:{chunk.start_line}-{chunk.end_line}"
            f"  [{chunk.symbol_kind or chunk.strategy}] {label}"
        )


if __name__ == "__main__":
    main()
