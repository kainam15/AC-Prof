"""Keep tests on pytest while allowing the standard-library mocking tools."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def violations(source, filename="<source>"):
    errors = []
    nodes = list(ast.walk(ast.parse(source, filename=filename)))
    # A plain submodule import also binds its parent package. Do not let
    # importing mock reopen access to the retired execution framework.
    parent_bound = any(isinstance(node, ast.Import) and any(
        alias.name.startswith("unittest.") and alias.asname is None for alias in node.names
    ) for node in nodes)
    for node in nodes:
        if isinstance(node, ast.Import):
            imports = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imports = [f"{node.module}.{alias.name}" for alias in node.names]
        elif (parent_bound and isinstance(node, ast.Attribute)
              and isinstance(node.value, ast.Name) and node.value.id == "unittest"):
            imports = [f"unittest.{node.attr}"]
        else:
            continue
        if any((name == "unittest" or name.startswith("unittest."))
               and not (name == "unittest.mock" or name.startswith("unittest.mock."))
               for name in imports):
            errors.append(f"{filename}:{node.lineno}: use pytest; only unittest.mock is allowed")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args(argv)
    errors = []
    for path in args.paths or [ROOT / name for name in ("acprof", "scripts", "tests")]:
        for file in sorted(path.rglob("*.py")) if path.is_dir() else [path]:
            errors.extend(violations(file.read_text(encoding="utf-8"), str(file)))
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
