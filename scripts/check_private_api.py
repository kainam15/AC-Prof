"""Check explicit cross-module private dependencies without importing application code."""
from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from dataclasses import asdict, dataclass
from importlib.util import resolve_name
from pathlib import Path


@dataclass(frozen=True, order=True)
class Dependency:
    source: str
    target: str
    symbol: str


def module_name(path: Path) -> str:
    parts = path.with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def is_private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


class Bindings(ast.NodeVisitor):
    """Collect one lexical scope; nested function locals must not leak out."""

    def __init__(self, package: str, modules: set[str]):
        self.package = package
        self.modules = modules
        self.names: dict[str, str | None] = {}

    def import_target(self, node: ast.ImportFrom) -> str:
        return resolve_name("." * node.level + (node.module or ""), self.package)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.name if alias.asname else alias.name.split(".")[0]
            self.names[alias.asname or name] = name if name in self.modules else None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        target = self.import_target(node)
        for alias in node.names:
            candidate = f"{target}.{alias.name}"
            self.names[alias.asname or alias.name] = candidate if candidate in self.modules else None

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names[node.id] = None

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.names[node.name] = None

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names[node.name] = None

    def visit_Lambda(self, node: ast.Lambda) -> None:
        pass

    def visit_ListComp(self, node: ast.ListComp) -> None:
        pass

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp


class References(ast.NodeVisitor):
    def __init__(self, path: Path, modules: set[str]):
        self.path = path.as_posix()
        self.source = module_name(path)
        self.package = self.source if path.name == "__init__.py" else self.source.rpartition(".")[0]
        self.modules = modules
        self.scopes: list[dict[str, str | None]] = []
        self.references: dict[Dependency, list[str]] = {}

    def record(self, target: str, symbol: str, line: int) -> None:
        if target in self.modules and target != self.source and is_private(symbol):
            dependency = Dependency(self.source, target, symbol)
            self.references.setdefault(dependency, []).append(f"{self.path}:{line}")

    def visit_scope(self, body: list[ast.stmt], arguments: ast.arguments | None = None) -> None:
        bindings = Bindings(self.package, self.modules)
        for node in body:
            bindings.visit(node)
        if arguments:
            for node in ast.walk(arguments):
                if isinstance(node, ast.arg):
                    bindings.names[node.arg] = None
        self.scopes.append(bindings.names)
        for node in body:
            self.visit(node)
        self.scopes.pop()

    def visit_Module(self, node: ast.Module) -> None:
        self.visit_scope(node.body)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for expression in [*node.decorator_list, *node.args.defaults, *node.args.kw_defaults, node.returns]:
            if expression is not None:
                self.visit(expression)
        for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs, node.args.vararg, node.args.kwarg]:
            if arg is not None and arg.annotation is not None:
                self.visit(arg.annotation)
        self.visit_scope(node.body, node.args)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in [*node.decorator_list, *node.bases, *node.keywords]:
            self.visit(expression)
        self.visit_scope(node.body)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        self.visit_scope([ast.Expr(value=node.body)], node.args)

    def visit_ListComp(self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp) -> None:
        self.visit(node.generators[0].iter)
        bindings = Bindings(self.package, self.modules)
        self.scopes.append(bindings.names)
        for index, generator in enumerate(node.generators):
            if index:
                self.visit(generator.iter)
            bindings.visit(generator.target)
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.scopes.pop()

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            parts = alias.name.split(".")
            for index, part in enumerate(parts[1:], 1):
                self.record(".".join(parts[:index]), part, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        target = Bindings(self.package, self.modules).import_target(node)
        for alias in node.names:
            self.record(target, alias.name, node.lineno)

    def resolve(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            for scope in reversed(self.scopes):
                if node.id in scope:
                    return scope[node.id]
        elif isinstance(node, ast.Attribute):
            parent = self.resolve(node.value)
            if parent:
                candidate = f"{parent}.{node.attr}"
                return candidate if candidate in self.modules else None
        return None

    def visit_Attribute(self, node: ast.Attribute) -> None:
        owner = self.resolve(node.value)
        if owner:
            self.record(owner, node.attr, node.lineno)
        self.generic_visit(node)


def scan_sources(root: Path, directories: tuple[str, ...] = ("acprof",)) -> dict[Dependency, list[str]]:
    modules = {module_name(path.relative_to(root)) for path in (root / "acprof").rglob("*.py")}
    if not modules:
        raise ValueError(f"no production Python modules found under {root / 'acprof'}")
    references: dict[Dependency, list[str]] = {}
    for directory in directories:
        for path in sorted((root / directory).rglob("*.py")):
            visitor = References(path.relative_to(root), modules)
            visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
            references.update(visitor.references)
    return references


def read_baseline(path: Path) -> set[Dependency]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("private API baseline requires schema_version=1")
    if not isinstance(payload.get("dependencies"), list):
        raise ValueError("private API baseline requires a dependency list")
    entries = [Dependency(**entry) for entry in payload["dependencies"]]
    if any(not all(isinstance(value, str) and value for value in (item.source, item.target, item.symbol))
           or not is_private(item.symbol) for item in entries):
        raise ValueError("baseline entries require source, target and a private symbol")
    if entries != sorted(set(entries)):
        raise ValueError("private API baseline must be sorted and contain no duplicates")
    return set(entries)


def write_baseline(path: Path, dependencies: set[Dependency]) -> None:
    payload = {"schema_version": 1, "dependencies": [asdict(item) for item in sorted(dependencies)]}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline", type=Path, default=Path("tests/private_api_baseline.json"))
    parser.add_argument("--prune", action="store_true", help="Remove obsolete entries; never approve new dependencies")
    args = parser.parse_args(argv)
    baseline_path = args.root / args.baseline
    try:
        baseline = read_baseline(baseline_path)
        production = scan_sources(args.root)
        tests = scan_sources(args.root, ("tests",))
    except (OSError, ValueError, TypeError, KeyError, SyntaxError) as exc:
        print(f"private API check failed: {exc}")
        return 1
    actual = set(production)
    if args.prune:
        baseline &= actual
        write_baseline(baseline_path, baseline)
    print(f"Production: {sum(map(len, production.values()))} references, {len(actual)} dependencies")
    print(f"Tests (informational): {sum(map(len, tests.values()))} references, {len(tests)} dependencies")
    for target, count in Counter(item.target for item in production).most_common():
        print(f"  {target}: {count}")
    for item in sorted(actual - baseline):
        print(f"NEW {item.source} -> {item.target}:{item.symbol} ({', '.join(production[item])})")
    for item in sorted(baseline - actual):
        print(f"STALE {item.source} -> {item.target}:{item.symbol} (remove from baseline or run --prune)")
    return int(actual != baseline)


if __name__ == "__main__":
    raise SystemExit(main())
