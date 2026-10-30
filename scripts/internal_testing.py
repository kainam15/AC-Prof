"""Manage ignored AC-Prof developer evidence without touching profiling results.

Only manifest-validated runs beneath internal-testing/runs are prune candidates.
Legacy directories, fixed shared outputs, and scientific results are never pruned.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "internal-testing"
CATEGORIES = ("runs", "evidence", "shared", "cache")
MANIFEST = "manifest.json"
SCHEMA_VERSION = 1
KIND = "acprof-dev-run"
SUCCESS_DAYS = 14
RESOLVED_DAYS = 30
# Scientific evidence and resumable experiment state are not disposable test logs.
PROTECTED_SUFFIXES = {".csv", ".pcap", ".pcapng", ".jsonl", ".parquet", ".npy", ".npz"}
PROTECTED_NAMES = {".acprof", "raw", "checkpoints", "checkpoint", "recovery"}
TASK_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
RUN_PATTERN = re.compile(r"(\d{8}T\d{6}Z)--([a-z0-9]+(?:-[a-z0-9]+)*)--([0-9a-f]{8})\Z")
MONTH_PATTERN = re.compile(r"\d{4}-(0[1-9]|1[0-2])\Z")


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("Expected an ISO UTC timestamp")
    result = datetime.fromisoformat(value[:-1] + "+00:00")
    if result.utcoffset() != timedelta(0):
        raise ValueError("Expected UTC")
    return result


def _directory(path: Path, *, create: bool = False) -> Path:
    """Never follow a symlink at any managed directory boundary."""
    if path.is_symlink():
        raise ValueError(f"Refusing symlink directory: {path}")
    if create:
        path.mkdir(exist_ok=True)
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")
    return path


def initialize(root: Path = DATA_ROOT) -> None:
    _directory(root, create=True)
    for category in CATEGORIES:
        _directory(root / category, create=True)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    commit = result.stdout.strip()
    return commit if re.fullmatch(r"[0-9a-f]{40}", commit) else None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return bool(result.stdout.strip())


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    """Publish a complete file; never expose a half-written manifest."""
    descriptor, filename = tempfile.mkstemp(prefix=".manifest-", dir=path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(filename, path / MANIFEST)
    finally:
        Path(filename).unlink(missing_ok=True)


def create_run(
    task: str,
    *,
    root: Path = DATA_ROOT,
    keep: bool = False,
    now: datetime | None = None,
    command: list[str] | None = None,
) -> Path:
    """Atomically allocate a new unique directory (no collision overwrites)."""
    if len(task) > 48 or not TASK_PATTERN.fullmatch(task):
        raise ValueError("Task must be lowercase kebab-case, 1..48 characters")
    moment = now or utc_now()
    if moment.tzinfo is None:
        raise ValueError("Run time must be timezone-aware")
    moment = moment.astimezone(UTC)
    initialize(root)
    runs = _directory(root / "runs")
    month = _directory(runs / moment.strftime("%Y-%m"), create=True)
    for _ in range(10):
        run_id = f"{moment.strftime('%Y%m%dT%H%M%SZ')}--{task}--{secrets.token_hex(4)}"
        path = month / run_id
        try:
            path.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        document = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "run_id": run_id,
            "task": task,
            "created_at": iso_utc(moment),
            "finished_at": None,
            "status": "running",
            "retention": "keep" if keep else "temporary",
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
        }
        # Only trusted callers should provide a redacted/safe command; never
        # automatically store arbitrary shell invocations, environment or tokens.
        if command is not None:
            document["command"] = list(command)
        try:
            for folder in ("logs", "reports", "screenshots", "artifacts"):
                (path / folder).mkdir()
            _write_manifest(path, document)
        except BaseException:
            # Nothing was launched yet; retain a failed allocation if cleanup fails.
            shutil.rmtree(path)
            raise
        return path
    raise FileExistsError("Could not allocate a unique run directory")


def _run_components(path: Path, root: Path) -> tuple[str, str]:
    """Check lexical layout and every managed ancestor before reading a file."""
    root = root.absolute()
    path = path.absolute()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Run must be below the configured internal-testing root")
    relative = path.relative_to(root).parts
    if len(relative) != 3 or relative[0] != "runs" or not MONTH_PATTERN.fullmatch(relative[1]):
        raise ValueError("Expected runs/YYYY-MM/<UTC>--<task>--<id>")
    match = RUN_PATTERN.fullmatch(relative[2])
    if match is None:
        raise ValueError("Invalid managed run directory name")
    stamp = datetime.strptime(match[1], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    if stamp.strftime("%Y-%m") != relative[1]:
        raise ValueError("Run date and month directory disagree")
    for ancestor in (root, root / "runs", root / "runs" / relative[1], path):
        _directory(ancestor)
    return relative[2], match[2]


def read_run(path: Path, *, root: Path = DATA_ROOT) -> dict[str, Any]:
    run_id, task = _run_components(path, root)
    manifest = path / MANIFEST
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("Missing or symlinked manifest")
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid manifest: {error}") from error
    if not isinstance(data, dict) or any(
        (
            data.get("schema_version") != SCHEMA_VERSION,
            data.get("kind") != KIND,
            data.get("run_id") != run_id,
            data.get("task") != task,
            data.get("status") not in {"running", "success", "failed", "interrupted", "resolved"},
            data.get("retention") not in {"temporary", "keep"},
        )
    ):
        raise ValueError("Manifest identity or lifecycle is invalid")
    created = parse_utc(data.get("created_at"))
    stamp = datetime.strptime(run_id.split("--", 1)[0], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    if created != stamp:
        raise ValueError("Manifest timestamp does not match directory")
    finished = data.get("finished_at")
    if data["status"] == "running":
        if finished is not None:
            raise ValueError("Running task cannot have finished_at")
    else:
        finished_at = parse_utc(finished)
        if finished_at < created:
            raise ValueError("Finish time precedes creation")
    return data


def finish_run(
    path: Path,
    *,
    status: str,
    root: Path = DATA_ROOT,
    keep: bool = False,
    now: datetime | None = None,
) -> None:
    if status not in {"success", "failed", "interrupted", "resolved"}:
        raise ValueError("Status must be success, failed, interrupted or resolved")
    data = read_run(path, root=root)
    previous = data["status"]
    if not (
        (previous == "running" and status in {"success", "failed", "interrupted"})
        or (previous in {"failed", "interrupted"} and status == "resolved")
    ):
        raise ValueError(f"Invalid lifecycle transition: {previous} -> {status}")
    moment = now or utc_now()
    if moment.tzinfo is None or moment.astimezone(UTC) < parse_utc(data["created_at"]):
        raise ValueError("Finish time is invalid")
    data["status"] = status
    data["finished_at"] = iso_utc(moment)
    if keep:
        data["retention"] = "keep"
    _write_manifest(path, data)


def _run_paths(root: Path):
    runs = root / "runs"
    if not runs.exists():
        return
    _directory(root)
    _directory(runs)
    for month in sorted(runs.iterdir()):
        if month.is_symlink() or not month.is_dir() or not MONTH_PATTERN.fullmatch(month.name):
            continue
        for path in sorted(month.iterdir()):
            if path.is_dir() and not path.is_symlink():
                yield path


def inventory(root: Path = DATA_ROOT, *, limit: int = 20) -> dict[str, Any]:
    if not root.exists():
        return {
            "categories": {key: 0 for key in CATEGORIES},
            "managed": {},
            "unrecognized_runs": 0,
            "legacy_count": 0,
            "legacy_sample": [],
        }
    _directory(root)
    categories: dict[str, int] = {}
    for name in CATEGORIES:
        child = root / name
        if child.is_symlink():
            categories[name] = -1
        elif child.is_dir():
            categories[name] = sum(1 for _ in child.iterdir())
        else:
            categories[name] = 0
    legacy = sorted(
        entry.name for entry in root.iterdir() if entry.name not in (*CATEGORIES, "AGENTS.md")
    )
    managed: dict[str, int] = {}
    invalid = 0
    for path in _run_paths(root):
        try:
            data = read_run(path, root=root)
        except ValueError:
            invalid += 1
            continue
        state = data["status"]
        managed[state] = managed.get(state, 0) + 1
    return {
        "categories": categories,
        "managed": managed,
        "unrecognized_runs": invalid,
        "legacy_count": len(legacy),
        "legacy_sample": legacy[:limit],
    }


def _has_protected_evidence(path: Path) -> bool:
    def raise_walk_error(error: OSError) -> None:
        raise error

    try:
        for _, folders, filenames in os.walk(path, followlinks=False, onerror=raise_walk_error):
            for name in (*folders, *filenames):
                if name in PROTECTED_NAMES or Path(name).suffix.lower() in PROTECTED_SUFFIXES:
                    return True
    except OSError:
        # Unreadable evidence is not disposable evidence.
        return True
    return False


def _eligible(path: Path, *, root: Path, now: datetime) -> bool:
    try:
        data = read_run(path, root=root)
    except ValueError:
        return False
    if data["retention"] != "temporary" or (path / ".keep").exists():
        return False
    days = {"success": SUCCESS_DAYS, "resolved": RESOLVED_DAYS}.get(data["status"])
    if days is None:
        return False
    return now.astimezone(UTC) >= parse_utc(data["finished_at"]) + timedelta(
        days=days
    ) and not _has_protected_evidence(path)


def prune_candidates(root: Path = DATA_ROOT, *, now: datetime | None = None) -> list[Path]:
    moment = now or utc_now()
    if moment.tzinfo is None:
        raise ValueError("Current time must be timezone-aware")
    return [path for path in _run_paths(root) if _eligible(path, root=root, now=moment)]


def prune(
    root: Path = DATA_ROOT, *, apply: bool = False, confirm: str = "", now: datetime | None = None
) -> list[Path]:
    if apply and confirm != "DELETE-ELIGIBLE-RUNS":
        raise ValueError("Applying prune requires --confirm DELETE-ELIGIBLE-RUNS")
    moment = now or utc_now()
    paths = prune_candidates(root, now=moment)
    if not apply:
        return paths
    removed = []
    for path in paths:
        # Revalidate immediately before destruction. Ignore unknown/changed
        # status, unrecognized directories and symlinked managed ancestors.
        if not _eligible(path, root=root, now=moment):
            continue
        shutil.rmtree(path)
        removed.append(path)
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create the four managed top-level directories")
    new = sub.add_parser("new", help="Allocate one run with an isolated manifest")
    new.add_argument("--task", required=True, help="Lowercase kebab-case task name")
    new.add_argument("--keep", action="store_true")
    finish = sub.add_parser("finish", help="Finish a run after its process exits")
    finish.add_argument("path", type=Path)
    finish.add_argument(
        "--status", required=True, choices=("success", "failed", "interrupted", "resolved")
    )
    finish.add_argument("--keep", action="store_true")
    listing = sub.add_parser("inventory", help="Count managed and legacy entries without deletion")
    listing.add_argument("--limit", type=int, default=20)
    listing.add_argument("--json", action="store_true")
    cleaning = sub.add_parser("prune", help="Preview eligible successful/resolved runs (default)")
    cleaning.add_argument(
        "--dry-run", action="store_true", help="Explicit preview (also the default)"
    )
    cleaning.add_argument("--apply", action="store_true", help="Delete only eligible managed runs")
    cleaning.add_argument(
        "--confirm", default="", help="Required with --apply: DELETE-ELIGIBLE-RUNS"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            initialize()
            print(DATA_ROOT)
        elif args.command == "new":
            print(create_run(args.task, keep=args.keep))
        elif args.command == "finish":
            finish_run(args.path, status=args.status, keep=args.keep)
            print(args.path)
        elif args.command == "inventory":
            if args.limit < 0:
                parser.error("--limit must be non-negative")
            result = inventory(limit=args.limit)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print("Managed categories:", result["categories"])
                print("Run statuses:", result["managed"])
                print("Unrecognized run directories:", result["unrecognized_runs"])
                print("Legacy entries:", result["legacy_count"])
                for name in result["legacy_sample"]:
                    print("  ", name)
        elif args.command == "prune":
            if args.apply and args.dry_run:
                parser.error("--dry-run and --apply cannot be combined")
            paths = prune(apply=args.apply, confirm=args.confirm)
            action = "Removed" if args.apply else "Would remove"
            print(f"{action} {len(paths)} managed run(s):")
            for path in paths:
                print(path)
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
