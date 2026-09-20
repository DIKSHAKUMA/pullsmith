"""Chunking tests.

Chunk quality is the single biggest lever on retrieval quality, so the properties that
matter are pinned here: whole declarations, correct symbol names, correct parent classes,
accurate line numbers, and a labelled fallback when no parser exists.
"""

from app.rag.chunk import chunk_file

PYTHON_SOURCE = '''\
import json
from typing import Any


def load_config(path: str) -> dict[str, Any]:
    """Reads a config file."""
    with open(path) as handle:
        return json.load(handle)


class ProfileService:
    """Handles profile updates."""

    def __init__(self, repository):
        self.repository = repository

    def update(self, user_id: str, email: str) -> dict:
        if not email:
            raise ValueError("email is required")

        return self.repository.save(user_id, email)

    def delete(self, user_id: str) -> None:
        self.repository.delete(user_id)
'''

TYPESCRIPT_SOURCE = """\
import { useState } from 'react'

export interface Profile {
  id: string
  email: string
}

export function validateEmail(email: string): boolean {
  if (!email) {
    return false
  }
  return email.includes('@')
}

export class ProfileClient {
  async update(id: string, email: string): Promise<Profile> {
    const response = await fetch(`/api/profiles/${id}`, { method: 'PATCH' })
    return response.json()
  }
}
"""


def test_python_functions_and_methods_become_separate_chunks() -> None:
    chunks = chunk_file(
        content=PYTHON_SOURCE, relative_path="app/profile_service.py", language="python"
    )

    symbols = {chunk.symbol for chunk in chunks}

    assert "load_config" in symbols
    assert "update" in symbols
    assert "delete" in symbols


def test_method_records_its_parent_class() -> None:
    """This is what makes an exact lookup for ProfileService.update possible."""
    chunks = chunk_file(
        content=PYTHON_SOURCE, relative_path="app/profile_service.py", language="python"
    )

    update = next(chunk for chunk in chunks if chunk.symbol == "update")

    assert update.parent_symbol == "ProfileService"
    assert update.qualified_name() == "ProfileService.update"


def test_chunk_contains_the_whole_function_not_a_fragment() -> None:
    chunks = chunk_file(
        content=PYTHON_SOURCE, relative_path="app/profile_service.py", language="python"
    )

    update = next(chunk for chunk in chunks if chunk.symbol == "update")

    # Signature, the guard clause and the return must all be present. A fixed-size
    # window would routinely cut one of them off.
    assert "def update" in update.content
    assert 'raise ValueError("email is required")' in update.content
    assert "return self.repository.save" in update.content


def test_line_numbers_point_at_the_real_location() -> None:
    chunks = chunk_file(
        content=PYTHON_SOURCE, relative_path="app/profile_service.py", language="python"
    )

    update = next(chunk for chunk in chunks if chunk.symbol == "update")
    source_lines = PYTHON_SOURCE.split("\n")

    assert "def update" in source_lines[update.start_line - 1]
    assert update.end_line > update.start_line


def test_imports_are_attached_to_every_chunk() -> None:
    chunks = chunk_file(
        content=PYTHON_SOURCE, relative_path="app/profile_service.py", language="python"
    )

    assert chunks
    for chunk in chunks:
        joined = " ".join(chunk.imports)
        assert "import json" in joined


def test_typescript_functions_interfaces_and_methods() -> None:
    chunks = chunk_file(
        content=TYPESCRIPT_SOURCE, relative_path="src/profile.ts", language="typescript"
    )

    symbols = {chunk.symbol for chunk in chunks}

    assert "validateEmail" in symbols
    assert "Profile" in symbols or any(chunk.symbol_kind == "interface" for chunk in chunks)
    assert "update" in symbols


def test_tsx_reuses_the_typescript_grammar() -> None:
    chunks = chunk_file(
        content=TYPESCRIPT_SOURCE, relative_path="src/Profile.tsx", language="tsx"
    )

    assert any(chunk.symbol == "validateEmail" for chunk in chunks)
    assert all(chunk.strategy == "syntax" for chunk in chunks)


def test_unknown_language_falls_back_to_labelled_line_windows() -> None:
    content = "\n".join(f"line {number}" for number in range(1, 200))

    chunks = chunk_file(content=content, relative_path="notes.txt", language=None)

    assert chunks
    # Labelled honestly so degraded retrieval can be attributed to chunking, not blamed
    # on the embedding model.
    assert all(chunk.strategy == "lines" for chunk in chunks)


def test_fallback_windows_overlap() -> None:
    content = "\n".join(f"line {number}" for number in range(1, 200))

    chunks = chunk_file(content=content, relative_path="notes.txt", language=None)

    assert len(chunks) >= 2
    # Overlap means a construct sitting on a boundary still appears whole somewhere.
    assert chunks[1].start_line < chunks[0].end_line


def test_oversized_declaration_is_split_and_labelled() -> None:
    body = "\n".join(f"    value_{number} = {number}" for number in range(400))
    content = f"def enormous():\n{body}\n"

    chunks = chunk_file(content=content, relative_path="big.py", language="python")

    assert len(chunks) > 1
    assert all(chunk.strategy == "syntax-split" for chunk in chunks)
    assert all(chunk.line_count <= 221 for chunk in chunks)


def test_empty_file_produces_no_chunks() -> None:
    assert chunk_file(content="   \n\n", relative_path="empty.py", language="python") == []


def test_module_of_top_level_statements_is_still_indexed() -> None:
    """A settings module has no functions but issues still reference it."""
    content = "DEBUG = True\nDATABASE_URL = 'postgres://'\nALLOWED = ['a', 'b']\n"

    chunks = chunk_file(content=content, relative_path="settings.py", language="python")

    assert chunks
    assert chunks[0].strategy == "lines"


def test_test_flag_is_carried_through() -> None:
    chunks = chunk_file(
        content=PYTHON_SOURCE,
        relative_path="tests/test_profile.py",
        language="python",
        is_test=True,
    )

    assert chunks
    assert all(chunk.is_test for chunk in chunks)


def test_broken_syntax_does_not_crash_the_indexer() -> None:
    """Real repositories contain files that do not parse. Indexing must survive them."""
    content = "def broken(:\n  this is not valid python !!!\n"

    chunks = chunk_file(content=content, relative_path="broken.py", language="python")

    assert isinstance(chunks, list)


CONST_SOURCE = """\
const NAV = [
  { to: '/', label: 'Dashboard' },
  { to: '/repositories', label: 'Repositories' },
]

export const TONE: Record<string, string> = {
  running: 'blue',
  gate: 'amber',
}

export const formatUser = (name: string) => name.trim().toLowerCase()
"""


def test_const_symbol_name_excludes_the_initialiser() -> None:
    """A symbol name must be an identifier, not the whole value it was assigned.

    Getting this wrong poisons exact-symbol retrieval: a stack trace mentioning `NAV`
    would never match a "symbol" that is actually a multi-line array literal.
    """
    chunks = chunk_file(
        content=CONST_SOURCE, relative_path="src/constants.ts", language="typescript"
    )

    symbols = {chunk.symbol for chunk in chunks}

    assert "NAV" in symbols
    assert "TONE" in symbols
    assert "formatUser" in symbols

    for chunk in chunks:
        if chunk.symbol:
            assert "\n" not in chunk.symbol
            assert "=" not in chunk.symbol
            assert len(chunk.symbol) < 60


def test_exported_declaration_keeps_the_export_keyword() -> None:
    """Whether a symbol is exported is part of its meaning, so it stays in the chunk."""
    chunks = chunk_file(
        content=CONST_SOURCE, relative_path="src/constants.ts", language="typescript"
    )

    tone = next(chunk for chunk in chunks if chunk.symbol == "TONE")

    assert tone.content.startswith("export")


def test_python_decorator_stays_attached_to_the_function() -> None:
    """`@app.post("/runs")` is often the most searchable line in the function."""
    source = (
        "@app.post('/runs')\n"
        "async def create_run(body: CreateRunRequest) -> RunResponse:\n"
        "    return await service.create(body)\n"
    )

    chunks = chunk_file(content=source, relative_path="app/api/runs.py", language="python")

    created = next(chunk for chunk in chunks if chunk.symbol == "create_run")

    assert "@app.post('/runs')" in created.content
    assert created.start_line == 1
