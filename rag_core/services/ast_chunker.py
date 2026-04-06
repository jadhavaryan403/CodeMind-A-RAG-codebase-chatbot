"""
ast_chunker.py

AST-based structured code chunking.
Splits Python source into meaningful semantic units:
  - Top-level functions
  - Classes (whole class body)
  - Methods inside classes

Each chunk carries:
  symbol_name, code_text, start_line, end_line, chunk_type, file_path
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CodeChunk:
    symbol_name: str        # e.g. "MyClass", "my_function", "MyClass.my_method"
    code_text: str          # The actual source code of this chunk
    start_line: int         # 1-indexed
    end_line: int
    chunk_type: str         # "function" | "class" | "method"
    file_path: str          # Relative path of the source file
    original_path: str = ''   # original_path is the path of the file when it was uploaded, which may differ from file_path if the file was renamed after upload


def _extract_source(source_lines: list[str], node: ast.AST) -> str:
    """Return the source lines corresponding to `node`."""
    start = node.lineno - 1          # ast is 1-indexed
    end = node.end_lineno            # end_lineno is inclusive
    return "".join(source_lines[start:end])


def chunk_file(file_path: str | Path) -> list[CodeChunk]:
    """
    Parse a .py file with the AST and return a list of CodeChunks.
    Falls back gracefully if parsing fails (e.g. syntax errors).
    """
    file_path = Path(file_path)
    source = file_path.read_text(encoding="utf-8", errors="replace")
    source_lines = source.splitlines(keepends=True)
    chunks: list[CodeChunk] = []

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        # Degrade gracefully: treat whole file as one chunk
        chunks.append(CodeChunk(
            symbol_name=file_path.stem,
            code_text=source,
            start_line=1,
            end_line=len(source_lines),
            chunk_type="module",
            file_path=str(file_path),
        ))
        return chunks

    for node in ast.walk(tree):
        # Top-level functions
        if isinstance(node, ast.FunctionDef) and not _is_method(node, tree):
            chunks.append(CodeChunk(
                symbol_name=node.name,
                code_text=_extract_source(source_lines, node),
                start_line=node.lineno,
                end_line=node.end_lineno,
                chunk_type="function",
                file_path=str(file_path),
            ))

        # Classes (whole class + all methods)
        elif isinstance(node, ast.ClassDef):
            chunks.append(CodeChunk(
                symbol_name=node.name,
                code_text=_extract_source(source_lines, node),
                start_line=node.lineno,
                end_line=node.end_lineno,
                chunk_type="class",
                file_path=str(file_path),
            ))
            # Individual methods within the class
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    chunks.append(CodeChunk(
                        symbol_name=f"{node.name}.{item.name}",
                        code_text=_extract_source(source_lines, item),
                        start_line=item.lineno,
                        end_line=item.end_lineno,
                        chunk_type="method",
                        file_path=str(file_path),
                    ))

    return chunks


def _is_method(func_node: ast.FunctionDef, tree: ast.AST) -> bool:
    """Return True if func_node is nested inside a ClassDef."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if item is func_node:
                    return True
    return False