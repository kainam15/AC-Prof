"""Resolve ordinary Python imports against pinned repository paths, never sys.path."""
from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from acprof.models.inspection.source import top_level_bound_names


@dataclass
class ImportTargets:
    sources: list[str] = field(default_factory=list)
    possible_sources: list[str] = field(default_factory=list)
    external_roots: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def _constant_initializer(tree: ast.Module) -> bool:
    """Only trivial initializers cannot re-enter the package or mutate its exports."""
    return all(isinstance(stmt, ast.Pass)
               or isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
               or isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant)
               and all(isinstance(target, ast.Name) for target in stmt.targets) for stmt in tree.body)


class RepositoryImports:
    def __init__(self, files: set[str], load_tree: Callable[[str], ast.Module]):
        self.files = files
        self.load_tree = load_tree
        self._exports: dict[str, set[str]] = {}
        self._constant_initializers: set[str] = set()

    def package_initializers(self, filename: str) -> list[str]:
        """A dotted entry point also loads its parents, even without explicit imports."""
        return [init for parent in PurePosixPath(filename).parents if parent.parts
                and (init := str(parent / "__init__.py")) in self.files and init != filename]

    def _source(self, module: PurePosixPath) -> str | None:
        # A regular Python package takes precedence over a same-named .py file.
        return next((path for path in (str(module / "__init__.py"), str(module) + ".py")
                     if path in self.files), None)

    def _binds(self, initializer: str, name: str) -> bool:
        if initializer not in self._exports:
            tree = self.load_tree(initializer)
            self._exports[initializer] = top_level_bound_names(tree)
            if _constant_initializer(tree):
                self._constant_initializers.add(initializer)
        return name in self._exports[initializer]

    def _fromlist(self, targets: ImportTargets, parent: PurePosixPath, names: list[ast.alias],
                  filename: str, *, bare_relative: bool, local: bool = True) -> None:
        initializer = str(parent / "__init__.py")
        for alias in names:
            child = self._source(parent / alias.name) if alias.name != "*" else None
            if not child and not bare_relative:
                continue  # Ordinary module attributes are not imports to resolve.
            if (initializer in self.files and (not child or filename != initializer)
                    and self._binds(initializer, alias.name)):
                # A final binding does not prove the child was shadowed during
                # circular imports or after dynamic mutation of the package.
                if child and initializer not in self._constant_initializers:
                    targets.possible_sources.append(child)
                continue
            if child:
                (targets.sources if local else targets.possible_sources).append(child)
            elif bare_relative:
                targets.missing.append(alias.name)

    def _named(self, targets: ImportTargets, base: PurePosixPath, module: str, names: list[ast.alias],
               filename: str, *, relative: bool) -> None:
        parts = module.split(".")
        local = relative or self._source(base / parts[0]) is not None
        if not local:
            # A possible namespace directory does not prove that the target
            # supplies the top-level package. Keep the lock check as well.
            targets.external_roots.append(parts[0])
        sources = targets.sources if local else targets.possible_sources
        path = base
        for part in parts[:-1]:
            path /= part
            if source := self._source(path):
                sources.append(source)
                if source != str(path / "__init__.py"):
                    targets.missing.append(module)  # A plain module is not a package.
                    return
        path /= parts[-1]
        source = self._source(path)
        if source:
            sources.append(source)
            if source != str(path / "__init__.py"):
                return
        elif local:
            targets.missing.append(module)
            return
        self._fromlist(targets, path, names, filename, bare_relative=False, local=local)

    def resolve(self, filename: str, node: ast.Import | ast.ImportFrom) -> ImportTargets:
        targets = ImportTargets()
        if isinstance(node, ast.Import):
            for alias in node.names:
                self._named(targets, PurePosixPath(), alias.name, [], filename, relative=False)
            return targets
        base = PurePosixPath(filename).parent if node.level else PurePosixPath()
        for _ in range(node.level - 1):
            if not base.parts:
                raise ValueError(f"{filename}: relative import escapes pinned snapshot")
            base = base.parent
        if node.level and (initializer := str(base / "__init__.py")) in self.files:
            targets.sources.append(initializer)
        if node.module:
            self._named(targets, base, node.module, node.names, filename, relative=bool(node.level))
        else:
            self._fromlist(targets, base, node.names, filename, bare_relative=True)
        return targets
