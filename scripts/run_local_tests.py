"""Local pytest validation with bounded safe shards and CI-compatible evidence.

Only isolated non-TUI unit tests may run concurrently. CI/full and TUI suites
retain the serial default until shared-resource isolation is audited.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.aggregate_test_reports import aggregate_reports  # noqa: E402

THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)
PARALLEL_SAFE_ROOTS = (ROOT / "tests" / "analysis", ROOT / "tests" / "quality")


def build_command(
    directory: Path, *, scope: str, shard: int, jobs: int, output: Path, keyword: str = "",
) -> list[str]:
    command = [
        sys.executable, str(ROOT / "scripts" / "run_tests.py"),
        "--directory", str(directory), "--report", str(output / f"shard-{shard}.json"),
        "--shard-index", str(shard), "--shard-count", str(jobs),
        "-q", "-p", "no:cacheprovider",
        "--basetemp", str(output / f"tmp-{shard}"),
    ]
    if scope != "full":
        command += ["-m", {"fast": "unit", "tui": "tui"}[scope]]
    if keyword:
        command += ["-k", keyword]
    return command


def collect_shards(output: Path, *, jobs: int) -> dict:
    """Verify all locally executed shards with the same checks as CI."""
    reports, errors = [], []
    for index in range(jobs):
        path = output / f"shard-{index}.json"
        try:
            reports.append(json.loads(path.read_text()))
        except (OSError, UnicodeError, ValueError) as error:
            errors.append(f"{path.name}: {error}")
    version = ".".join(platform.python_version().split(".")[:2])
    result = aggregate_reports(reports, python_versions=[version], shard_count=jobs)
    result["errors"].extend(errors)
    result["successful"] = result["successful"] and not errors
    return result


def stop_processes(processes: list[subprocess.Popen]) -> None:
    """Stop each shard's process group, including its owned test children."""
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def run_local(*, directory: Path, scope: str, jobs: int, output: Path, keyword: str = "") -> bool:
    output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    if jobs > 1:
        for variable in THREAD_VARIABLES:
            environment[variable] = "1"

    processes: list[subprocess.Popen] = []
    logs = []
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for index in range(jobs):
            command = build_command(
                directory, scope=scope, shard=index, jobs=jobs, output=output, keyword=keyword,
            )
            log = (output / f"shard-{index}.log").open("w", encoding="utf-8")
            logs.append(log)
            processes.append(subprocess.Popen(
                command, cwd=ROOT, env=environment, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            ))
        # Preserve all failures: other shards finish so the coverage union can be audited.
        statuses = [process.wait() for process in processes]
    except (KeyboardInterrupt, Exception):
        stop_processes(processes)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        for log in logs:
            log.close()

    result = collect_shards(output, jobs=jobs)
    result["exit_codes"] = statuses
    if any(status != 0 for status in statuses):
        result["errors"].append(f"nonzero shard exit codes: {statuses}")
        result["successful"] = False
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if result["successful"]:
        for index in range(jobs):
            shutil.rmtree(output / f"tmp-{index}", ignore_errors=False)
    version = ".".join(platform.python_version().split(".")[:2])
    count = result["versions"].get(version, {}).get("tests", 0)
    print(f"Scope={scope}, jobs={jobs}, checked cases={count}, complete={result['successful']}")
    for error in result["errors"]:
        print(f"ERROR: {error}", file=sys.stderr)
    print(f"Evidence: {output}")
    return result["successful"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("fast", "tui", "full"), default="fast")
    parser.add_argument("--jobs", type=int, default=1, help="1 by default; at most 2 for fast tests")
    parser.add_argument("--path", default="tests", help="Repository-relative test file or directory")
    parser.add_argument("--keyword", default="", help="Optional pytest -k expression")
    parser.add_argument("--output", type=Path, help="New evidence directory (must not exist)")
    args = parser.parse_args(argv)
    if not 1 <= args.jobs <= 2:
        parser.error("--jobs must be 1 or 2; avoid oversubscribing the shared server")
    if args.jobs > 1 and args.scope != "fast":
        parser.error("only isolated fast tests allow --jobs 2; TUI/full remain serial")
    directory = (ROOT / args.path).resolve()
    if not directory.is_relative_to(ROOT / "tests") or not directory.exists():
        parser.error("--path must name an existing file or directory inside tests/")
    if args.jobs > 1 and not any(directory.is_relative_to(root) for root in PARALLEL_SAFE_ROOTS):
        parser.error(
            "--jobs 2 requires --path within tests/analysis or tests/quality; "
            "other suites await resource-isolation validation"
        )
    output = args.output or (
        ROOT / "internal-testing" / "local-tests" / f"{args.scope}-{uuid.uuid4().hex[:10]}"
    )
    output = (ROOT / output).resolve()
    evidence_root = (ROOT / "internal-testing").resolve()
    if output == evidence_root or not output.is_relative_to(evidence_root):
        parser.error("--output must be inside internal-testing/ (not results/)")
    if output.exists():
        parser.error("--output must be a new directory to avoid overwriting previous evidence")
    try:
        return 0 if run_local(
            directory=directory, scope=args.scope, jobs=args.jobs, output=output, keyword=args.keyword,
        ) else 1
    except KeyboardInterrupt:
        print("Interrupted: stopped owned test process groups", file=sys.stderr)
        return 130
    except OSError as error:
        print(f"Cannot run tests: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
