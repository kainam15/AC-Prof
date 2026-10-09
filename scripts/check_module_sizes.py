"""Report large Python modules for human architectural review, without a line-count gate.

Counts physical lines, including comments and data declarations. Branch points and
static imports are review hints, not formal complexity or coupling scores.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOFT_TARGET = 500
REVIEW_THRESHOLD = 800
LONG_CALLABLE = 100


@dataclass(frozen=True)
class ModuleSize:
    path: str
    lines: int
    status: str
    data_catalog: bool
    functions: int
    classes: int
    longest_callable: str | None
    longest_callable_lines: int
    longest_callable_branches: int
    long_callables: int
    internal_imports: int


def _branch_points(function: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Count decision nodes as a *hint*, not McCabe cyclomatic complexity."""
    # Do not attribute decisions in nested definitions to their parent.
    stack: list[ast.AST] = [function]
    points = 0
    while stack:
        node = stack.pop()
        if node is not function and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, (ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While,
                             ast.ExceptHandler, ast.Assert, ast.comprehension)):
            points += 1
        elif isinstance(node, ast.BoolOp):
            points += len(node.values) - 1
        elif isinstance(node, ast.match_case):
            points += 1
        stack.extend(ast.iter_child_nodes(node))
    return points


def _catalog_ratio(tree: ast.Module, lines: int) -> float:
    """Detect very long top-level dict/list literals without path-based exemptions."""
    covered = 0
    for node in tree.body:
        value = None
        if isinstance(node, ast.Assign):
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            value = node.value
        if isinstance(value, (ast.Dict, ast.List, ast.Tuple)):
            covered += node.end_lineno - node.lineno + 1
    return covered / max(lines, 1)


def inspect_module(path: Path, root: Path) -> ModuleSize:
    source = path.read_text(encoding="utf-8")
    lines = len(source.splitlines())
    tree = ast.parse(source, filename=str(path))
    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    classes = sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
    longest = max(functions, key=lambda node: node.end_lineno - node.lineno + 1, default=None)
    long_callables = sum(node.end_lineno - node.lineno + 1 >= LONG_CALLABLE for node in functions)
    internal_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            internal_imports.update(alias.name for alias in node.names if alias.name.startswith("acprof."))
        elif isinstance(node, ast.ImportFrom):
            # Count static import declarations; resolution and runtime coupling
            # remain the responsibility of check_import_boundaries.py.
            if node.level or (node.module or "").startswith("acprof."):
                internal_imports.add(f"{node.level}:{node.module}")
    # Only a mostly-static module without long functions gets catalog treatment.
    data_catalog = (
        _catalog_ratio(tree, lines) >= 0.60
        and long_callables == 0
        and (longest is None or longest.end_lineno - longest.lineno + 1 <= 60)
    )
    status = "review" if lines > REVIEW_THRESHOLD else "target" if lines > SOFT_TARGET else "ok"
    if data_catalog and status == "review":
        status = "data-review"  # Still visible, but not automatically refactor-priority.
    return ModuleSize(
        path=path.relative_to(root).as_posix(),
        lines=lines,
        status=status,
        data_catalog=data_catalog,
        functions=len(functions),
        classes=classes,
        longest_callable=longest.name if longest else None,
        longest_callable_lines=longest.end_lineno - longest.lineno + 1 if longest else 0,
        longest_callable_branches=_branch_points(longest) if longest else 0,
        long_callables=long_callables,
        internal_imports=len(internal_imports),
    )


def scan_modules(root: Path) -> list[ModuleSize]:
    source_dir = root / "acprof"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing Python package directory: {source_dir}")
    return sorted(
        (inspect_module(path, root) for path in source_dir.rglob("*.py")),
        key=lambda item: (-item.lines, item.path),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root")
    parser.add_argument("--top", type=int, default=25, help="Max highlighted modules (0 = all)")
    parser.add_argument("--json", action="store_true", help="Emit full machine-readable report")
    args = parser.parse_args(argv)
    if args.top < 0:
        parser.error("--top must be >= 0")
    try:
        modules = scan_modules(args.root)
    except (OSError, SyntaxError, UnicodeError) as exc:
        parser.exit(1, f"Module review failed: {exc}\n")
    if args.json:
        print(json.dumps({"soft_target": SOFT_TARGET, "review_threshold": REVIEW_THRESHOLD,
                          "modules": [asdict(module) for module in modules]}, ensure_ascii=False, indent=2))
        return 0
    review = [item for item in modules if item.status == "review"]
    data = [item for item in modules if item.status == "data-review"]
    target = [item for item in modules if item.status == "target"]
    print(f"Module review: {len(modules)} Python files; >{SOFT_TARGET}: "
          f"{len(review) + len(data) + len(target)}; >{REVIEW_THRESHOLD}: "
          f"{len(review) + len(data)} ({len(data)} data catalogs)")
    print("Size is advisory. Investigate responsibilities, branches, dependencies and test coverage.")
    flagged = review + data + target
    for item in flagged[:args.top or None]:
        callable_hint = (f"{item.longest_callable}({item.longest_callable_lines} lines, "
                         f"{item.longest_callable_branches} decisions)"
                         if item.longest_callable else "none")
        print(f"[{item.status:11}] {item.lines:5} {item.path} | "
              f"largest={callable_hint}; >=100-line funcs={item.long_callables}; "
              f"classes={item.classes}; imports={item.internal_imports}")
    return 0  # No hard line-count cap, including in CI.


if __name__ == "__main__":
    raise SystemExit(main())
