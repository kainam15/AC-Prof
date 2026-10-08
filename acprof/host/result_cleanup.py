"""Clean per-case temporary artifacts only after final result publication."""
from __future__ import annotations

import os
from pathlib import Path

from acprof.artifact_layout import ArtifactLayout


def cleanup_intermediate_results(csv_paths: list[str], output_dir: str, final_csv: str) -> None:
    """Delete per-run intermediate artifacts after the merged CSV is safely written."""
    if not csv_paths:
        return

    if not os.path.exists(final_csv):
        print(f"[cleanup][WARN] Skip cleanup because merged CSV is missing: {final_csv}")
        return

    if os.path.getsize(final_csv) <= 0:
        print(f"[cleanup][WARN] Skip cleanup because merged CSV is empty: {final_csv}")
        return

    layout = ArtifactLayout.discover(output_dir)
    targets: set[str] = set()
    for csv_path in csv_paths:
        if layout.layout_version == 2:
            case = layout.case_from_csv(csv_path)
            case.retain_requests()
            targets.update(str(layout.contained(path.relative_to(layout.root))) for path in case.temporary_files())
            continue
        targets.add(csv_path)
        targets.add(f"{csv_path}.sniff_groups.jsonl")

        base_name = os.path.basename(csv_path)
        if not (base_name.startswith("result_case_") and base_name.endswith(".csv")):
            print(f"[cleanup][WARN] Skip derived cleanup for unexpected CSV name: {csv_path}")
            continue

        case_name = base_name[len("result_"):-len(".csv")]
        targets.add(os.path.join(output_dir, f"lat_{case_name}.json"))
        targets.add(os.path.join(output_dir, f"sniff_{case_name}.pcap"))

    removed = 0
    missing = 0
    failed = 0

    for path in sorted(targets):
        if not os.path.exists(path):
            missing += 1
            continue
        try:
            os.remove(path)
            removed += 1
            print(f"[cleanup] Removed: {path}")
        except OSError as exc:
            failed += 1
            print(f"[cleanup][WARN] Failed to remove {path}: {exc}")

    if layout.layout_version == 2:
        for csv_path in csv_paths:
            try:
                Path(csv_path).parent.rmdir()
            except OSError:
                pass  # Keep any unexpected file for diagnosis.
    print(f"[cleanup] Done. removed={removed}, missing={missing}, failed={failed}")
