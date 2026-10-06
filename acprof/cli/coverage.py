"""Freeze a Hub sample or report resolution and isolated execution coverage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from acprof.artifacts import read_json_object


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot", help="Freeze revisions and download weights for a rolling sample")
    snapshot.add_argument("--stratum", action="append", required=True, help="TASK:LIBRARY; repeat for multiple strata")
    snapshot.add_argument("--limit", type=int, default=5, help="Head models per stratum")
    snapshot.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="Evaluate an existing frozen sample")
    run.add_argument("manifest", type=Path)
    run.add_argument("--output-dir", type=Path, required=True, help="Report directory; existing only with --resume")
    run.add_argument("--resume", action="store_true", help="Continue unfinished models with the recorded configuration")
    run.add_argument("--retry-failed", action="store_true", help="With --resume, retry failed models in new attempts")
    run.add_argument("--retry-stage", action="append", default=[], help="With --resume, retry matching failure stages")
    run.add_argument("--retry-reason", action="append", default=[], help="With --resume, retry matching reason codes")
    run.add_argument("--validate-runtime", action="store_true", help="Validate one end-to-end request per model")
    run.add_argument("--cpus", type=int, default=2)
    run.add_argument("--mems", type=int, default=4)
    run.add_argument("--gpus", choices=("off", "on"), default="off")
    run.add_argument("--timeout-seconds", type=float, default=300)
    run.add_argument("--max-parameters", type=int, help="Conservative parameter budget; excluded models remain unverified")
    run.add_argument("--max-download-bytes", type=int, help="Budget for selected pinned artifacts, including dependencies")
    report = commands.add_parser("report", help="Summarize recorded result directories without rerunning models")
    report.add_argument("sources", type=Path, nargs="+")
    report.add_argument("--output-dir", type=Path, required=True, help="New report directory")
    args = parser.parse_args(argv)
    from acprof.host.env_utils import bootstrap_project_env
    from acprof.host.model_coverage import CoverageCleanupError, run_sample, snapshot_sample
    from acprof.host.run_state import RunStateError
    bootstrap_project_env(Path.cwd())
    try:
        if args.command == "snapshot":
            if args.output.exists():
                raise ValueError("sample already exists; choose a new output path")
            sample = snapshot_sample(args.stratum, args.limit)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(sample, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
            print(f"Frozen sample: {args.output}")
        elif args.command == "report":
            from acprof.analysis.compatibility import report_results
            result = report_results(args.sources, args.output_dir)
            print(f"Reported {len(result['rows'])} recorded results: {args.output_dir}")
        else:
            report = run_sample(read_json_object(args.manifest, label="coverage sample"), args.output_dir,
                                 probe="full" if args.validate_runtime else "none",
                                 cpus=args.cpus, memory_gb=args.mems, gpu=args.gpus == "on", timeout_seconds=args.timeout_seconds,
                                 max_parameters=args.max_parameters, max_download_bytes=args.max_download_bytes,
                                 resume=args.resume, retry_failed=args.retry_failed,
                                 retry_stages=args.retry_stage, retry_reasons=args.retry_reason)
            print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError, TypeError, RunStateError, CoverageCleanupError) as exc:
        print(f"[coverage][ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
