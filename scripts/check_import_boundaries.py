"""Reject reverse dependencies between AC-Prof implementation layers.

Only runtime imports are checked; imports guarded by TYPE_CHECKING do not cause
runtime coupling. A two-edge exception documents existing monitor command use.
"""
from __future__ import annotations

import argparse
import ast
from importlib.util import resolve_name
from pathlib import Path

FORBIDDEN = {
    "analysis": frozenset({"host", "container", "monitors", "plotting", "tui", "cli"}),
    "plotting": frozenset({"host", "container", "monitors", "tui", "cli"}),
    "host": frozenset({"cli", "tui"}),
    "container": frozenset({"host", "cli", "tui", "plotting", "analysis"}),
    "monitors": frozenset({"host", "cli", "tui", "plotting", "analysis", "container"}),
    "workloads": frozenset({"host", "cli", "tui"}),
    "packet": frozenset({"cli", "tui"}),
}
# Monitor discovery currently uses the shared host subprocess runner.
# Keep this exception narrow until command ownership is independently migrated.
EXISTING_EXCEPTIONS = {
    ("acprof.monitors.common", "acprof.host.command"),
    ("acprof.monitors.perf_mips", "acprof.host.command"),
}


def _type_checking_guard(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"
        or isinstance(test, ast.Attribute)
        and isinstance(test.value, ast.Name)
        and test.value.id == "typing"
        and test.attr == "TYPE_CHECKING"
    )


class ImportChecker(ast.NodeVisitor):
    def __init__(self, module: str, *, is_package: bool = False):
        self.module = module
        self.package = module if is_package else module.rpartition(".")[0]
        self.layer = module.split(".")[1] if module.count(".") >= 1 else ""
        self.violations: list[str] = []

    def _check(self, imported: str, line: int) -> None:
        parts = imported.split(".")
        if len(parts) < 3 or parts[0] != "acprof":
            return
        if parts[1] not in FORBIDDEN.get(self.layer, ()):
            return
        for source, target in EXISTING_EXCEPTIONS:
            if self.module == source and (imported == target or imported.startswith(target + ".")):
                return
        self.violations.append(f"{self.module}:{line}: forbidden import {imported}")

    def visit_If(self, node: ast.If) -> None:
        if _type_checking_guard(node.test):
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check(alias.name, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = node.module or ""
        if node.level:
            base = resolve_name("." * node.level + base, self.package)
        self._check(base, node.lineno)
        for alias in node.names:
            self._check(f"{base}.{alias.name}", node.lineno)


def violations_in_text(module: str, source: str, *, is_package: bool = False) -> list[str]:
    checker = ImportChecker(module, is_package=is_package)
    checker.visit(ast.parse(source))
    return sorted(set(checker.violations))


def scan_import_violations(root: Path) -> list[str]:
    violations = []
    for path in sorted((root / "acprof").rglob("*.py")):
        parts = path.relative_to(root).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        for violation in violations_in_text(
            module, path.read_text(encoding="utf-8"), is_package=path.name == "__init__.py"
        ):
            violations.append(f"{path.relative_to(root)}: {violation}")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        violations = scan_import_violations(args.root)
    except (OSError, SyntaxError, ValueError, ImportError) as exc:
        print(f"import boundary check failed: {exc}")
        return 1
    for entry in violations:
        print(entry)
    if violations:
        print(f"Rejected {len(violations)} forbidden import(s)")
        return 1
    print("Import boundaries: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
