"""Size-bounded AST parsing shared by static analyzers."""
from __future__ import annotations

import ast

MAX_SOURCE_BYTES = 256 * 1024
MAX_AST_NODES = 20000

def parse_source(source: str, filename: str) -> ast.Module:
    if len(source.encode()) > MAX_SOURCE_BYTES:
        raise ValueError(f"{filename}: source exceeds {MAX_SOURCE_BYTES} bytes")
    try:
        tree = ast.parse(source, filename=filename)
    except (SyntaxError, RecursionError, ValueError) as exc:
        raise ValueError(f"{filename}: cannot parse Python source: {exc}") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ValueError(f"{filename}: AST exceeds {MAX_AST_NODES} nodes")
    return tree
