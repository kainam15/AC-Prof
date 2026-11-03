"""Validate, publish and report a completed resource matrix."""
from __future__ import annotations

import sys
import time
from argparse import Namespace

from acprof.application.errors import CollectionFailed
from acprof.application.notifications import RunNotifications
from acprof.application.outcome import RunOutcome
from acprof.application.resources import format_elapsed
from acprof.capabilities import (
    CapabilityReport,
    apply_collection_result,
    missing_required_measurements,
)
from acprof.events.progress import emit_event
from acprof.host.collection_history import COLLECTION_HISTORY_NAME
from acprof.host.orchestrator import merge_all_csvs
from acprof.host.result_cleanup import cleanup_intermediate_results
from acprof.host.run_state import RunState
from acprof.results.layout import ArtifactLayout


def publish_matrix_results(
    csv_paths: list[str],
    *,
    args: Namespace,
    layout: ArtifactLayout,
    run_state: RunState,
    capability_report: CapabilityReport,
    start_time: float,
    notifications: RunNotifications,
) -> RunOutcome:
    static_meta_json = str(layout.path("static_meta.json"))
    collection_history_json = str(layout.path(COLLECTION_HISTORY_NAME))
    if not csv_paths:
        elapsed = format_elapsed(time.perf_counter() - start_time)
        print(
            f"\n[WARN] No results produced after {elapsed}. "
            f"Static meta is still available: {static_meta_json}"
        )
        notifications.record_completion(final_csv=None, completed_cases=0)
        return RunOutcome("no_results")

    final_csv = str(layout.result_csv)
    emit_event("results_merging")
    merge_all_csvs(csv_paths, final_csv, expected=run_state.expected())
    from acprof.results.csv import read_result_csv
    _, collected_rows = read_result_csv(final_csv)
    apply_collection_result(capability_report, collected_rows)
    from acprof.artifacts import atomic_write_json
    atomic_write_json(layout.path("capability_report.json"), capability_report.to_dict())
    missing = missing_required_measurements(capability_report, collected_rows)
    if missing:
        print(
            f"[capability][ERROR] {args.profiling_mode} 必需指标缺少有效测量：{', '.join(missing)}。"
            "能力报告和 CSV 已保留；本次画像失败。",
            file=sys.stderr,
        )
        raise CollectionFailed("required measurements missing: " + ", ".join(missing))
    run_state.finish(final_csv)
    emit_event("result_published", final_csv=final_csv)
    cleanup_intermediate_results(csv_paths, str(layout.root), final_csv)
    elapsed = format_elapsed(time.perf_counter() - start_time)
    print(f"\n{'=' * 60}")
    print("Profiling complete!")
    print(f"  Profiling mode:   {args.profiling_mode}")
    if not capability_report.to_dict()["requested_measurements_complete"]:
        print("  [WARN] 请求指标存在缺失；能力报告保留具体状态，结果不标记为完整画像。")
    print(f"  Static meta:      {static_meta_json}")
    print(f"  Collection log:   {collection_history_json}")
    print(f"  Frozen matrix:    {layout.path('matrix_plan.json')}")
    if args.prune_startup_oom:
        print(f"  Startup evidence: {layout.path('startup_oom_pruning.json')}")
    print(f"  Merged results:   {final_csv}")
    print(f"  Total elapsed:    {elapsed}")
    print("  Intermediate files from this run were cleaned up.")
    print(f"{'=' * 60}")
    notifications.record_completion(final_csv=final_csv, completed_cases=len(csv_paths))
    return RunOutcome("ok", final_csv, len(csv_paths))
