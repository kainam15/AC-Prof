"""Enforce the package contracts declared in pyproject.toml with Import Linter."""
from __future__ import annotations

import argparse
from contextlib import chdir
from pathlib import Path

from importlinter.cli import lint_imports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    with chdir(args.root):
        return lint_imports(config_filename="pyproject.toml", no_cache=True, no_logo=True)


if __name__ == "__main__":
    raise SystemExit(main())
