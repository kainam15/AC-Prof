#!/usr/bin/env python3
"""Check TUI copy without importing the application.

Unused entries are candidates: arbitrary runtime keys are not statically provable.
"""
from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter

_KEY = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){2,}\Z")
_CJK = re.compile(r"[\u3400-\u9fff]")
_SUPPORT = {"__init__.py", "entry.py"}
_I18N_MODULE = "acprof.tui.i18n"


@dataclass(frozen=True)
class Issue:
    path: Path
    line: int
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.code}: {self.detail}"


@dataclass(frozen=True)
class CatalogEntry:
    key: str
    zh: str
    en: str
    path: Path
    line: int


@dataclass
class Report:
    entries: dict[str, CatalogEntry] = field(default_factory=dict)
    errors: list[Issue] = field(default_factory=list)
    unused: list[CatalogEntry] = field(default_factory=list)
    dynamic: list[Issue] = field(default_factory=list)


def _parse(path: Path, root: Path, report: Report) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        report.errors.append(Issue(path.relative_to(root), getattr(error, "lineno", 1) or 1,
                                   "parse-error", str(error)))
        return None


def _assignment(node: ast.AST) -> tuple[list[str], ast.expr | None]:
    if isinstance(node, ast.Assign):
        return [target.id for target in node.targets if isinstance(target, ast.Name)], node.value
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id], node.value
    return [], None


def _fields(template: str) -> list[tuple[str, str, str]]:
    result = []
    for _, name, spec, conversion in Formatter().parse(template):
        if name is None:
            continue
        if not name.split(".")[0].split("[")[0].isdigit():
            raise ValueError("use explicit positional fields such as {0}")
        if conversion not in {None, "r", "s", "a"}:
            raise ValueError(f"unsupported conversion !{conversion}")
        result.append((name, spec, conversion or ""))
        if spec:
            result.extend(_fields(spec))
    return sorted(result)


def _read_catalog(root: Path, report: Report) -> None:
    directory = root / "acprof" / "tui" / "catalog"
    if not directory.is_dir():
        report.errors.append(Issue(Path("acprof/tui/catalog"), 1, "missing-catalog",
                                   "catalog directory does not exist"))
        return
    source_entries: dict[str, CatalogEntry] = {}
    for path in sorted(directory.glob("*.py")):
        if path.name in _SUPPORT:
            continue
        tree = _parse(path, root, report)
        if tree is None:
            continue
        relative = path.relative_to(root)
        declarations = [value for node in tree.body for names, value in [_assignment(node)]
                        if "MESSAGES" in names]
        if len(declarations) != 1 or not isinstance(declarations[0], ast.Dict):
            report.errors.append(Issue(relative, 1, "catalog-shape",
                                       "define exactly one literal MESSAGES dictionary"))
            continue
        keys = []
        for key_node, value in zip(declarations[0].keys, declarations[0].values, strict=True):
            line = getattr(key_node, "lineno", value.lineno)
            if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                report.errors.append(Issue(relative, line, "invalid-key",
                                           "catalog keys must be semantic string literals"))
                continue
            key = key_node.value
            keys.append(key)
            if not _KEY.fullmatch(key):
                report.errors.append(Issue(relative, line, "invalid-key", key))
            if key in report.entries:
                previous = report.entries[key]
                report.errors.append(Issue(relative, line, "duplicate-key",
                                           f"{key}; first defined at {previous.path}:{previous.line}"))
            if (not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name)
                    or value.func.id != "Entry"):
                report.errors.append(Issue(relative, line, "entry-shape",
                                           f"{key}: use Entry(zh=..., en=...)"))
                continue
            keywords = {item.arg: item.value for item in value.keywords}
            if value.args or set(keywords) != {"zh", "en"} or len(value.keywords) != 2:
                report.errors.append(Issue(relative, line, "entry-shape",
                                           f"{key}: supply exactly zh and en"))
                continue
            if any(not isinstance(node, ast.Constant) or not isinstance(node.value, str)
                   for node in keywords.values()):
                report.errors.append(Issue(relative, line, "entry-shape",
                                           f"{key}: translations must be string literals"))
                continue
            zh = keywords["zh"].value
            en = keywords["en"].value
            entry = CatalogEntry(key, zh, en, relative, line)
            report.entries.setdefault(key, entry)
            if not zh.strip() or not en.strip():
                report.errors.append(Issue(relative, line, "empty-translation", key))
            try:
                if _fields(zh) != _fields(en):
                    report.errors.append(Issue(relative, line, "format-mismatch", key))
            except ValueError as error:
                report.errors.append(Issue(relative, line, "invalid-format", f"{key}: {error}"))
            previous_source = source_entries.setdefault(zh, entry)
            if previous_source.en != en:
                report.errors.append(Issue(relative, line, "source-conflict",
                                           f"{key}: same Chinese source has different English at "
                                           f"{previous_source.path}:{previous_source.line}"))
        if keys != sorted(keys):
            report.errors.append(Issue(relative, 1, "unsorted-keys",
                                       "sort MESSAGES by semantic key"))
    if not report.entries:
        report.errors.append(Issue(Path("acprof/tui/catalog"), 1, "empty-catalog",
                                   "no catalog entries were found"))


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else ""
    return ""


def _bindings(scope: ast.AST) -> dict[str, ast.expr | str | None]:
    """Track one lexical scope; repeated writes remain explicitly unresolved."""
    result: dict[str, ast.expr | str | None] = {}

    def record(name: str, value: ast.expr | str | None) -> None:
        if isinstance(value, str) and result.get(name) == value:
            return
        result[name] = None if name in result else value

    class Bindings(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            record(node.name, None)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            record(node.name, None)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            pass

        def visit_ListComp(self, node: ast.ListComp) -> None:
            pass

        visit_SetComp = visit_ListComp
        visit_DictComp = visit_ListComp
        visit_GeneratorExp = visit_ListComp

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                record(alias.asname or alias.name.split(".")[0],
                       alias.name if alias.asname else alias.name.split(".")[0])

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for alias in node.names:
                target = f"{node.module}.{alias.name}" if not node.level else None
                record(alias.asname or alias.name, target)

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                record(node.id, None)

        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    record(target.id, node.value)
                else:
                    self.visit(target)
            self.visit(node.value)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if isinstance(node.target, ast.Name):
                record(node.target.id, node.value)
            else:
                self.visit(node.target)
            if node.value is not None:
                self.visit(node.value)

        def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
            if isinstance(node.target, ast.Name):
                record(node.target.id, node.value)
            self.visit(node.value)

    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        arguments = scope.args
        for argument in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
                         arguments.vararg, arguments.kwarg]:
            if argument is not None:
                record(argument.arg, None)
    visitor = Bindings()
    for statement in getattr(scope, "body", []):
        visitor.visit(statement)
    return result


def _call_name(node: ast.AST, bindings: dict[str, ast.expr | str | None]) -> str:
    name, _, suffix = _dotted(node).partition(".")
    target = bindings.get(name)
    if isinstance(target, str):
        return f"{target}.{suffix}" if suffix else target
    return ""


def _resolve(node: ast.AST, bindings: dict[str, ast.expr | str | None],
             seen: frozenset[str]) -> ast.AST | None:
    if isinstance(node, ast.Name):
        value = bindings.get(node.id)
        if isinstance(value, ast.expr) and node.id not in seen:
            return _resolve(value, bindings, seen | {node.id})
        return None
    return node


def _strings(node: ast.AST, bindings: dict[str, ast.expr | str | None],
             seen: frozenset[str] = frozenset()) -> tuple[set[str], bool]:
    """Resolve finite literals/maps; the boolean records unresolved branches."""
    if isinstance(node, ast.Constant):
        return ({node.value}, False) if isinstance(node.value, str) else (set(), False)
    if isinstance(node, ast.Name):
        value = bindings.get(node.id)
        if isinstance(value, ast.expr) and node.id not in seen:
            return _strings(value, bindings, seen | {node.id})
        return set(), True
    if isinstance(node, ast.IfExp):
        parts = (node.body, node.orelse)
    elif isinstance(node, ast.Dict):
        parts = tuple(node.values)
    elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        parts = tuple(node.elts)
    elif (isinstance(node, ast.Subscript)
          or isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
          and node.func.attr == "get" and node.args):
        is_get = isinstance(node, ast.Call)
        container = node.func.value if is_get else node.value
        selector = node.args[0] if is_get else node.slice
        defaults = tuple(node.args[1:]) if is_get else ()
        resolved = _resolve(container, bindings, seen)
        query = _resolve(selector, bindings, seen)
        if (isinstance(resolved, ast.Dict) and isinstance(query, ast.Constant)
                and all(isinstance(key, ast.Constant) for key in resolved.keys)):
            for key, value in reversed(list(zip(resolved.keys, resolved.values, strict=True))):
                if key.value == query.value:
                    return _strings(value, bindings, seen)
            if is_get:
                return _strings(defaults[0], bindings, seen) if defaults else (set(), False)
            return set(), True
        if (isinstance(resolved, (ast.Tuple, ast.List)) and isinstance(query, ast.Constant)
                and isinstance(query.value, int)):
            try:
                return _strings(resolved.elts[query.value], bindings, seen)
            except IndexError:
                return set(), True
        parts = (container, *defaults)
    else:
        return set(), True
    values: set[str] = set()
    unresolved = False
    for part in parts:
        found, unknown = _strings(part, bindings, seen)
        values.update(found)
        unresolved |= unknown
    return values, unresolved


def _argument(node: ast.Call, keyword: str) -> ast.expr | None:
    if node.args:
        return node.args[0]
    return next((item.value for item in node.keywords if item.arg == keyword), None)


def _source_strings(tree: ast.Module) -> set[str]:
    """Documentation does not prove a message is still used by production code."""
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                first = node.body[0].value
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    docstrings.add(first)
    return {node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and node not in docstrings}


def _read_references(root: Path, report: Report) -> None:
    literal_ids, source_literals, resolved_ids = set(), set(), set()
    catalog_path = root / "acprof" / "tui" / "catalog"
    for path in sorted((root / "acprof").rglob("*.py")):
        if path.is_relative_to(catalog_path):
            continue
        tree = _parse(path, root, report)
        if tree is None:
            continue
        relative = path.relative_to(root)
        strings = _source_strings(tree)
        literal_ids.update(strings & report.entries.keys())
        source_literals.update(strings)
        module_bindings = _bindings(tree)
        strict_tui = path.is_relative_to(root / "acprof" / "tui") and path.name != "i18n.py"

        class References(ast.NodeVisitor):
            bindings = module_bindings

            def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
                previous = self.bindings
                for expression in [*node.decorator_list, *node.args.defaults,
                                   *node.args.kw_defaults]:
                    if expression is not None:
                        self.visit(expression)
                self.bindings = {**previous, **_bindings(node)}
                for statement in node.body:
                    self.visit(statement)
                self.bindings = previous

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                previous = self.bindings
                class_bindings = {**previous, **_bindings(node)}
                for statement in node.body:
                    # Method bodies do not close over class-local names.
                    self.bindings = (previous if isinstance(
                        statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    ) else class_bindings)
                    self.visit(statement)
                self.bindings = previous

            def visit_Call(self, node: ast.Call) -> None:
                function = _call_name(node.func, self.bindings)
                if function == f"{_I18N_MODULE}.message_id":
                    first = _argument(node, "msgid")
                    values, unknown = (_strings(first, self.bindings) if first is not None
                                       else (set(), True))
                    resolved_ids.update(values)
                    for value in sorted(values - report.entries.keys()):
                        report.errors.append(Issue(relative, node.lineno, "unknown-key", value))
                    if unknown:
                        report.dynamic.append(Issue(relative, node.lineno, "dynamic-reference",
                                                    "key cannot be fully resolved statically"))
                is_message = function == "acprof.messages.message"
                is_translate = isinstance(node.func, ast.Attribute) and node.func.attr == "tr"
                first = _argument(node, "template" if is_message else "source")
                if (strict_tui and (is_message or is_translate) and isinstance(first, ast.Constant)
                        and isinstance(first.value, str) and _CJK.search(first.value)):
                    report.errors.append(Issue(relative, node.lineno, "literal-ui-message",
                                               "register application copy and use message_id"))
                self.generic_visit(node)

        References().visit(tree)
    used = literal_ids | resolved_ids
    report.unused = [entry for key, entry in sorted(report.entries.items())
                     if key not in used and entry.zh not in source_literals]


def check_catalog(root: Path) -> Report:
    """Return hard errors, unresolved calls and conservative unused candidates."""
    report = Report()
    _read_catalog(root, report)
    if report.entries:
        _read_references(root, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--unused", action="store_true", help="also list candidates; never delete")
    args = parser.parse_args(argv)
    report = check_catalog(args.root.resolve())
    for issue in report.errors:
        print(issue)
    for issue in report.dynamic:
        print(issue)
    if args.unused:
        for entry in report.unused:
            print(f"{entry.path}:{entry.line}: unused-candidate: {entry.key}")
    print(f"Catalog: {len(report.entries)} entries; {len(report.errors)} errors; "
          f"{len(report.dynamic)} unresolved dynamic references; "
          f"{len(report.unused)} unused candidates.")
    return int(bool(report.errors))


if __name__ == "__main__":
    raise SystemExit(main())
