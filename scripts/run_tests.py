"""CI compatibility wrapper for the single pytest runner and evidence plugin."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "tests")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pattern", action="append")
    parser.add_argument("--require-no-skips", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args, extra = parser.parse_known_args(argv)
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("需要 0 <= --shard-index < --shard-count")
    import pytest

    directory = args.directory.resolve()
    root = ROOT if directory.is_relative_to(ROOT) else directory
    command = [str(directory), "--rootdir", str(root), "-p", "acprof.testing.plugin",
               "-p", "no:asyncio", "-p", "pytest_asyncio.plugin",
               "--import-mode=importlib", "-o", "asyncio_mode=auto",
               "-o", "asyncio_default_fixture_loop_scope=function",
               "--report", str(args.report), "--shard-index", str(args.shard_index),
               "--shard-count", str(args.shard_count)]
    if root == ROOT:
        # Existing report paths on another mount must not affect pytest's
        # common-ancestor inference and hide the checkout configuration.
        command += ["-c", str(ROOT / "pyproject.toml")]
    for pattern in args.pattern or []:
        command += ["--pattern", pattern]
    if args.require_no_skips:
        command.append("--require-no-skips")
    return int(pytest.main([*command, *extra]))


if __name__ == "__main__":
    raise SystemExit(main())
