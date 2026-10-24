"""Convert every hashed Docker runtime wheel lock into audit-only version pins.

No packages or wheels are downloaded, resolved, installed or modified. Hashes and
direct wheel identities remain enforced by the canonical dependency lock parser.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acprof.dependency_locks import read_python_lock  # noqa: E402
from acprof.runtime_profiles import ENVIRONMENTS, PLATFORMS  # noqa: E402


def export_runtime_locks(output_dir: Path, *, root: Path = ROOT,
                         python_version: str) -> dict[str, int]:
    """Fail if any registered platform or runtime lock escapes the audit."""
    platforms = {item.requirements_lock for item in PLATFORMS.values()}
    environments = {item.requirements_lock for item in ENVIRONMENTS.values()}
    expected = {Path(name) for name in platforms | environments}
    found = {path.relative_to(root) for path in (root / "dockerfiles/locks").glob("*.txt")}
    if not expected or found != expected:
        raise ValueError(f"runtime audit lock coverage differs: missing={sorted(expected - found)}, "
                         f"unregistered={sorted(found - expected)}")
    mismatches = {
        key: spec.python_version for key, spec in PLATFORMS.items()
        if not spec.python_version.startswith(python_version + ".")
    }
    if mismatches:
        raise ValueError(f"runtime audit must use the target Python version: {mismatches}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('runtime audit input directory must be empty to prevent stale reports')
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for relative in sorted(expected):
        records = read_python_lock(root / relative)
        if not records:
            raise ValueError(f"empty runtime dependency lock: {relative}")
        lines = [
            f"{record['name']}=={record['version']} --hash=sha256:{record['sha256']}"
            for record in records
        ]
        destination = output_dir / relative.name
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        counts[relative.name] = len(lines)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-version", required=True, help="Major.minor runtime lock target (e.g. 3.10)")
    args = parser.parse_args(argv)
    counts = export_runtime_locks(args.output_dir, python_version=args.python_version)
    print(f"Runtime audit inputs: {len(counts)} registered locks, "
          f"{sum(counts.values())} pinned package occurrences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
