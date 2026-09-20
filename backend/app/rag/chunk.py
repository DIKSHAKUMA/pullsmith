"""Syntax-aware code chunking.

Why not fixed-size chunks
-------------------------
The default RAG approach splits text every N characters. For prose that is acceptable.
For code it is actively harmful: a 500-character window cuts a function in half, so one
chunk holds a signature with no body and the next holds a body with no name. Neither
chunk answers "where is email validation handled?" because neither is a complete thought.

What we do instead
------------------
tree-sitter parses the file into a real syntax tree, and we cut at declaration
boundaries: functions, methods, classes, interfaces. Every chunk is therefore a unit a
developer would recognise, and it carries its symbol name, kind, parent class, line span
and imports as metadata.

Metadata is what makes retrieval precise. Knowing a chunk is the method
``ProfileService.update`` at lines 40-72 of ``profile_service.py`` lets us filter by
symbol name for a stack trace, filter out tests, and cite exact line numbers.
"""

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, cast

logger = logging.getLogger(__name__)

#: Files with no parser fall back to line windows rather than being dropped.
FALLBACK_WINDOW_LINES = 80
FALLBACK_OVERLAP_LINES = 10

#: A declaration larger than this is split, so one giant class cannot swamp the context
#: budget at retrieval time.
MAX_CHUNK_LINES = 220

#: Unnamed fragments below this size are dropped as noise. Named declarations are always
#: kept regardless of size: a two-line ``delete`` method is still a real symbol someone
#: will search for by name, whereas a two-line anonymous fragment is not.
MIN_UNNAMED_CHUNK_LINES = 3


@dataclass
class CodeChunk:
    relative_path: str
    language: str | None
    symbol: str | None
    symbol_kind: str | None
    parent_symbol: str | None
    start_line: int
    end_line: int
    content: str
    imports: list[str] = field(default_factory=list)
    is_test: bool = False
    strategy: str = "syntax"

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    def qualified_name(self) -> str:
        """The name used for exact-symbol lookups, e.g. ``ProfileService.update``."""
        if self.symbol and self.parent_symbol:
            return f"{self.parent_symbol}.{self.symbol}"
        return self.symbol or self.relative_path


#: tree-sitter node types that represent a declaration worth isolating, per language.
DECLARATION_NODES: dict[str, dict[str, str]] = {
    "python": {
        "function_definition": "function",
        "class_definition": "class",
        "decorated_definition": "decorated",
    },
    "javascript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "lexical_declaration": "binding",
        "export_statement": "export",
    },
    "typescript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
        "enum_declaration": "enum",
        "lexical_declaration": "binding",
        "export_statement": "export",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_declaration": "type",
    },
    "rust": {
        "function_item": "function",
        "impl_item": "impl",
        "struct_item": "struct",
        "trait_item": "trait",
        "enum_item": "enum",
    },
    "java": {
        "class_declaration": "class",
        "method_declaration": "method",
        "interface_declaration": "interface",
    },
    "ruby": {"method": "method", "class": "class", "module": "module"},
}

DECLARATION_NODES["tsx"] = DECLARATION_NODES["typescript"]

#: Nodes whose text is an import statement, used to attach imports to every chunk.
IMPORT_NODES: frozenset[str] = frozenset(
    {
        "import_statement",
        "import_from_statement",
        "import_declaration",
        "use_declaration",
    }
)

#: Declaration kinds that hold other declarations. We descend into these and record the
#: container's name as each inner chunk's ``parent_symbol``.
CONTAINER_KINDS: frozenset[str] = frozenset({"class", "impl", "struct", "trait", "module"})

#: Declaration kinds that wrap exactly one inner declaration, such as a Python decorator
#: or a TypeScript ``export``. The wrapper is emitted as the chunk so the decorator lines
#: and the ``export`` keyword are preserved, but the name comes from the inner node.
#: Keeping decorators matters: ``@app.post("/runs")`` is often the most searchable line
#: in the whole function.
WRAPPER_KINDS: frozenset[str] = frozenset({"decorated", "export"})

#: Guards against pathological nesting in generated code.
MAX_WALK_DEPTH = 14


@lru_cache(maxsize=32)
def _parser_for(language: str) -> Any | None:
    """Loads and caches a tree-sitter parser.

    Cached because building a parser is comparatively expensive and a repository has many
    files of the same language.
    """
    try:
        from tree_sitter_language_pack import get_parser

        # get_parser is typed with a Literal of every supported language. Ours comes from a
        # file extension at runtime, so it cannot be narrowed statically. An unsupported name
        # raises and is handled below, which is the check that actually matters.
        return get_parser(cast("Any", language))
    except Exception as exc:
        logger.info("no tree-sitter parser for %s (%s)", language, type(exc).__name__)
        return None


#: Node types that are themselves a bare name.
IDENTIFIER_NODES: frozenset[str] = frozenset(
    {
        "identifier",
        "type_identifier",
        "property_identifier",
        "field_identifier",
        "constant",
        "shorthand_property_identifier_pattern",
    }
)


def _text_of(node: Any, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _node_name(node: Any, source: bytes, depth: int = 0) -> str | None:
    """Extracts a declaration's identifier.

    Recursion is needed because grammars nest names differently. A JavaScript
    ``const NAV = [...]`` is a ``lexical_declaration`` whose name sits two levels down
    inside a ``variable_declarator``. Naively taking the declarator's text would return
    the entire array literal as the "symbol name", which would then poison exact-symbol
    retrieval.
    """
    if depth > 3:
        return None

    if node.type in IDENTIFIER_NODES:
        return _text_of(node, source).strip() or None

    for field_name in ("name", "declarator", "pattern"):
        child = node.child_by_field_name(field_name)

        if child is None:
            continue

        if child.type in IDENTIFIER_NODES:
            return _text_of(child, source).strip() or None

        # e.g. lexical_declaration -> variable_declarator -> identifier
        nested = _node_name(child, source, depth + 1)
        if nested:
            return nested

    for child in node.named_children:
        if child.type in IDENTIFIER_NODES:
            return _text_of(child, source).strip() or None

        if child.type in ("variable_declarator", "declarator", "init_declarator"):
            nested = _node_name(child, source, depth + 1)
            if nested:
                return nested

    return None


def _collect_imports(root: Any, source: bytes, limit: int = 40) -> list[str]:
    """Gathers import lines from the top of the file.

    Imports are attached to every chunk from that file because they answer "what does
    this code depend on?", which a chunk cut from the middle of a file cannot otherwise
    show. They also feed structural retrieval in the next step.
    """
    imports: list[str] = []

    for child in root.named_children:
        if child.type in IMPORT_NODES:
            text = source[child.start_byte : child.end_byte].decode(
                "utf-8", errors="replace"
            ).strip()
            imports.append(text)

            if len(imports) >= limit:
                break

    return imports


def _split_oversized(chunk: CodeChunk) -> list[CodeChunk]:
    """Splits a declaration that is too large to be one chunk.

    A 900-line class would otherwise consume the entire prompt budget, and most of it
    would be irrelevant to the question asked.
    """
    if chunk.line_count <= MAX_CHUNK_LINES:
        return [chunk]

    lines = chunk.content.split("\n")
    parts: list[CodeChunk] = []

    for offset in range(0, len(lines), MAX_CHUNK_LINES):
        window = lines[offset : offset + MAX_CHUNK_LINES]
        if not any(line.strip() for line in window):
            continue

        parts.append(
            CodeChunk(
                relative_path=chunk.relative_path,
                language=chunk.language,
                symbol=chunk.symbol,
                symbol_kind=chunk.symbol_kind,
                parent_symbol=chunk.parent_symbol,
                start_line=chunk.start_line + offset,
                end_line=chunk.start_line + offset + len(window) - 1,
                content="\n".join(window),
                imports=chunk.imports,
                is_test=chunk.is_test,
                strategy="syntax-split",
            )
        )

    return parts


def _make_chunk(
    node: Any,
    source: bytes,
    context: dict[str, Any],
    *,
    symbol: str | None,
    kind: str | None,
    parent_symbol: str | None,
) -> list[CodeChunk]:
    chunk = CodeChunk(
        relative_path=context["relative_path"],
        language=context["language"],
        symbol=symbol,
        symbol_kind=kind,
        parent_symbol=parent_symbol,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        content=source[node.start_byte : node.end_byte].decode("utf-8", errors="replace"),
        imports=context["imports"],
        is_test=context["is_test"],
    )
    return _split_oversized(chunk)


def _inner_declaration(node: Any, declarations: dict[str, str]) -> tuple[Any, str] | None:
    """Finds the declaration a wrapper node wraps, e.g. inside a decorator or export."""
    for child in node.named_children:
        kind = declarations.get(child.type)
        if kind and kind not in WRAPPER_KINDS:
            return child, kind
    return None


def _walk(
    node: Any,
    source: bytes,
    declarations: dict[str, str],
    context: dict[str, Any],
    parent_symbol: str | None = None,
    depth: int = 0,
) -> list[CodeChunk]:
    """Depth-first walk emitting one chunk per declaration.

    Three cases, in order:

    1. A wrapper (decorator, export) is emitted whole, named after the declaration inside
       it, so decorators and the export keyword stay attached.
    2. A container (class, impl, trait) is descended into so each method becomes its own
       chunk while inheriting the container's name as ``parent_symbol``. An empty
       container is emitted as itself rather than disappearing.
    3. Any other declaration is emitted as a chunk, and we do **not** descend into it, so
       a nested helper stays inside its enclosing function rather than being torn out.

    Nodes that are not declarations (a Python ``block``, a JS ``class_body``) are simply
    traversed through: they carry no meaning on their own but hold the declarations we
    want.
    """
    if depth > MAX_WALK_DEPTH:
        return []

    chunks: list[CodeChunk] = []

    for child in node.named_children:
        kind = declarations.get(child.type)

        if kind in WRAPPER_KINDS:
            inner = _inner_declaration(child, declarations)

            if inner is None:
                continue

            inner_node, inner_kind = inner

            if inner_kind in CONTAINER_KINDS:
                # A decorated or exported class: descend so its methods are separate.
                container_name = _node_name(inner_node, source)
                inner_chunks = _walk(
                    inner_node, source, declarations, context,
                    container_name or parent_symbol, depth + 1,
                )
                chunks.extend(
                    inner_chunks
                    or _make_chunk(
                        child, source, context,
                        symbol=container_name, kind=inner_kind, parent_symbol=parent_symbol,
                    )
                )
            else:
                # Emit the wrapper so decorators and `export` are kept in the chunk.
                chunks.extend(
                    _make_chunk(
                        child, source, context,
                        symbol=_node_name(inner_node, source),
                        kind=inner_kind,
                        parent_symbol=parent_symbol,
                    )
                )
            continue

        if kind in CONTAINER_KINDS:
            container_name = _node_name(child, source)
            inner_chunks = _walk(
                child, source, declarations, context,
                container_name or parent_symbol, depth + 1,
            )
            chunks.extend(
                inner_chunks
                or _make_chunk(
                    child, source, context,
                    symbol=container_name, kind=kind, parent_symbol=parent_symbol,
                )
            )
            continue

        if kind:
            chunks.extend(
                _make_chunk(
                    child, source, context,
                    symbol=_node_name(child, source), kind=kind, parent_symbol=parent_symbol,
                )
            )
            continue

        # Structural node (block, class_body, program): traverse through it.
        if child.named_child_count:
            chunks.extend(
                _walk(child, source, declarations, context, parent_symbol, depth + 1)
            )

    return chunks


def _fallback_windows(
    content: str, relative_path: str, language: str | None, is_test: bool
) -> list[CodeChunk]:
    """Line-window chunking for files with no parser.

    Overlapping windows mean a construct sitting on a boundary still appears whole in one
    of them. This is the degraded path and is labelled as such, so retrieval quality can
    be compared against syntax chunking rather than silently blamed on the embeddings.
    """
    lines = content.split("\n")
    chunks: list[CodeChunk] = []
    step = FALLBACK_WINDOW_LINES - FALLBACK_OVERLAP_LINES

    for offset in range(0, max(len(lines), 1), step):
        window = lines[offset : offset + FALLBACK_WINDOW_LINES]

        if not any(line.strip() for line in window):
            continue

        chunks.append(
            CodeChunk(
                relative_path=relative_path,
                language=language,
                symbol=None,
                symbol_kind=None,
                parent_symbol=None,
                start_line=offset + 1,
                end_line=offset + len(window),
                content="\n".join(window),
                is_test=is_test,
                strategy="lines",
            )
        )

        if offset + FALLBACK_WINDOW_LINES >= len(lines):
            break

    return chunks


def chunk_file(
    *,
    content: str,
    relative_path: str,
    language: str | None,
    is_test: bool = False,
) -> list[CodeChunk]:
    """Chunks one file, using syntax boundaries when a parser exists."""
    if not content.strip():
        return []

    declarations = DECLARATION_NODES.get(language or "", {})
    parser = _parser_for(language) if language and declarations else None

    if parser is None:
        return _fallback_windows(content, relative_path, language, is_test)

    source = content.encode("utf-8")

    try:
        tree = parser.parse(source)
    except Exception as exc:
        logger.warning("parse failed for %s: %s", relative_path, type(exc).__name__)
        return _fallback_windows(content, relative_path, language, is_test)

    imports = _collect_imports(tree.root_node, source)

    chunks = _walk(
        tree.root_node,
        source,
        declarations,
        {
            "relative_path": relative_path,
            "language": language,
            "imports": imports,
            "is_test": is_test,
        },
    )

    # A file of pure top-level statements (a config module, a script) yields no
    # declarations. Falling back keeps it searchable instead of invisible.
    if not chunks:
        return _fallback_windows(content, relative_path, language, is_test)

    kept = [
        chunk
        for chunk in chunks
        if chunk.symbol or chunk.line_count >= MIN_UNNAMED_CHUNK_LINES
    ]
    return kept or chunks
