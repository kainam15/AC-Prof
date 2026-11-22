"""Publish one synthetic window for the TUI's on-disk result attribution checks."""

import csv
import hashlib
import json
from pathlib import Path

from acprof.results.layers import publish_result_rows
from acprof.results.layout import ArtifactLayout


def publish_synthetic_window(
    directory: Path, *, model: str, pid: int, metrics_complete: bool
) -> Path:
    layout = ArtifactLayout.for_new_run(directory)
    layout.initialize()
    run_id = json.loads(layout.result_csv.read_text())["run_id"]
    case = layout.case(model, 1, 4, "off").csv
    row = dict(
        cpu_cores="1",
        mem_cap_gb="4",
        gpu_mode="off",
        input_scale="64",
        warmup="0",
        repeat_idx="0",
        status="ok",
        latency_app_s="0.02",
        environment_class="unknown",
    )
    case.parent.mkdir(parents=True, exist_ok=True)
    with case.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    state = dict(
        schema_version=1,
        layout_version=2,
        run_id=run_id,
        status="complete",
        outcome="ok",
        host={},
        artifacts={},
        attempts=[
            dict(
                attempt_id="fixture-attempt",
                pid=pid,
                resume=False,
                started_at="2026-01-01T00:00:00Z",
                ended_at="2026-01-01T00:00:01Z",
            )
        ],
        options=dict(cpus="1", mems="4", gpus="off", warmup=0, repeat=1),
        runtime=dict(
            task=dict(
                model_id=model,
                pipeline_tag="text-classification",
                task_family="nlp",
                runtime_backend="transformers_pipeline",
                library_name="transformers",
                model_revision="a" * 40,
                detection_method="fixture",
            ),
            image=dict(tag="sha256:" + "a" * 64),
            planned=dict(scales=[64], source="manual"),
            compute_plan="",
            execution_plan="",
        ),
        cases={
            str(case.relative_to(directory)): dict(
                status="complete", sha256=hashlib.sha256(case.read_bytes()).hexdigest()
            )
        },
    )
    layout.path("run_state.json").write_text(json.dumps(state), encoding="utf-8")
    publish_result_rows(list(row), [row], directory)
    layout.path("capability_report.json").write_text(
        json.dumps({"requested_measurements_complete": metrics_complete}), encoding="utf-8"
    )
    return layout.result_csv
