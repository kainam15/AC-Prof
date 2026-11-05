"""Profiler scales are identities through commands, payloads, plans and resume."""
import json
import math
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from acprof.host.detect import TaskInfo
from acprof.host.profiling import compute_plan, execution_plan, support
from acprof.host.profiling.tools import advisor, massif, ncu, nsys

SCALES = [1.0000001, 1.0000002]
NCU_METRIC = "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"


@pytest.mark.parametrize("scale", [8.0, 1.0000001, 1.0000002, 0.13392857142857142])
def test_runner_argument_preserves_scale(scale):
    args = support.profile_runner_args({"input_scale": scale}, 1, "cpu")
    assert float(args[args.index("--input-scale") + 1]) == scale
    if scale == 8.0:
        assert args[args.index("--input-scale") + 1] == "8"


@pytest.mark.parametrize(("requested", "expected"), [(SCALES[1], "second"), (1.00000015, None)])
def test_runner_selects_only_exact_payload(tmp_path, monkeypatch: pytest.MonkeyPatch, requested, expected):
    from acprof.container import compute_profile_runner as runner

    payload_file = tmp_path / "input_scale_plan.json"
    payload_file.write_text(json.dumps({"schema_version": 2, "entries": [
        {"input_scale": SCALES[0], "payload": {"label": "first"}},
        {"input_scale": SCALES[1], "payload": {"label": "second"}},
    ]}))
    config = SimpleNamespace(
        model_id="fixture", model_revision="main", backend="fixture",
        resolve_source=lambda: "fixture",
    )
    monkeypatch.setattr(runner, "ModelLoadConfig", SimpleNamespace(from_environment=lambda **_kwargs: config))
    monkeypatch.setattr(runner, "prepare_execution", lambda _config: (None, "cpu"))
    handler = Mock()
    handler.preprocess.side_effect = RuntimeError("payload selected")
    monkeypatch.setattr(runner, "select_handler", lambda _config: handler)
    monkeypatch.setattr(runner, "load_model_session", lambda *_args, **_kwargs: SimpleNamespace(context={}))
    monkeypatch.setattr(sys, "argv", [
        "compute_profile_runner", "--payload-file", str(payload_file),
        "--input-scale", str(requested), "--repeat", "1", "--profile-mode", "cpu",
    ])

    message = "payload selected" if expected else "compute payload not found"
    with pytest.raises(RuntimeError, match=message):
        runner.main()
    if expected:
        handler.preprocess.assert_called_once_with({}, {"label": expected})
    else:
        handler.preprocess.assert_not_called()
    handler.predict.assert_not_called()


@pytest.mark.parametrize(("mode", "tool", "field", "error"), [
    ("off", "torch_profiler_eager", compute_plan.TORCH_LOGICAL_MFLOP_FIELD, compute_plan.TORCH_ERROR_FIELD),
    ("on", "ncu", compute_plan.NCU_TOTAL_MFLOP_FIELD, compute_plan.NCU_ERROR_FIELD),
])
def test_compute_plan_uses_exact_scale(mode: str, tool: str, field: str, error: str):
    plan = {"profiles": {"cpu" if mode == "off" else "gpu": {tool: {"entries": [
        {"input_scale": SCALES[0], field: 10.0, "error": "first only"},
        {"input_scale": SCALES[1], field: 20.0, "error": ""},
    ]}}}}
    selected = compute_plan.find_compute_profile_entry(plan, mode, SCALES[1])
    assert selected[field] == 20.0
    assert selected[error] == ""
    missing = compute_plan.find_compute_profile_entry(plan, mode, 1.00000015)
    assert math.isnan(missing[field])
    assert "missing_scale" in missing[error]


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize(("mode", "tool", "field", "error"), [
    ("off", "massif", execution_plan.MASSIF_HEAP_PEAK_FIELD, execution_plan.MASSIF_ERROR_FIELD),
    ("on", "nsys", execution_plan.NSYS_HOST_WALL_TIME_FIELD, execution_plan.NSYS_ERROR_FIELD),
])
def test_execution_plan_uses_exact_scale(indexed: bool, mode: str, tool: str, field: str, error: str):
    plan = {"profiles": [{
        "cpu_cores": 2, "mem_cap_gb": 4, "gpu_mode": mode,
        "tools": {tool: {"entries": [
            {"input_scale": SCALES[0], field: 100, "error": "first only"},
            {"input_scale": SCALES[1], field: 200, "error": ""},
        ]}},
    }]}
    source = execution_plan.ExecutionProfileResolver(plan) if indexed else plan
    selected = execution_plan.find_execution_profile_entry(source, 2, 4, mode, SCALES[1])
    assert selected[field] == 200
    assert selected[error] == ""
    missing = execution_plan.find_execution_profile_entry(source, 2, 4, mode, 1.00000015)
    assert math.isnan(missing[field])
    assert "missing_scale" in missing[error]


@pytest.fixture
def profile_fixture(tmp_path, monkeypatch: pytest.MonkeyPatch):
    task = TaskInfo(
        model_id="fixture/model", model_revision="fixed", pipeline_tag="fill-mask",
        task_family="nlp", runtime_backend="transformers_pipeline",
        library_name="transformers", detection_method="fixture",
    )
    collections = []

    def run(command, **_kwargs):
        if "--query-metrics" in command:
            return SimpleNamespace(returncode=0, stdout=NCU_METRIC, stderr="")
        if "--help" in command:
            return SimpleNamespace(returncode=0, stdout="--nvtx-include", stderr="")
        assert "--input-scale" in command
        collections.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="fixture collection boundary")

    for module in (ncu, massif, nsys, advisor):
        monkeypatch.setattr(module, "run_profiler_command", run)
        monkeypatch.setattr(module, "profiler_container_command", lambda **_kwargs: nullcontext(["fixture-container"]))
    monkeypatch.setattr(ncu, "tool_mount_roots", lambda *_args: ())
    monkeypatch.setattr(advisor, "tool_mount_roots", lambda *_args: ())

    def profile(tool, scales, *, resume=False):
        common = dict(
            entries=[{"input_scale": scale} for scale in scales], task_info=task,
            cpu=2, mem=4, payload_file=str(tmp_path / "plan.json"), profile_root=str(tmp_path),
            repeat=1,
        )
        if tool == "ncu":
            return ncu.profile_gpu_entries(
                **common, ncu_bin="/fixture/ncu", ncu_root=None, image_tag="fixture-image", resume_existing=resume,
            )
        if tool == "nsys":
            return nsys.profile(
                **common, global_error="", image_tag="fixture-image", nsys_bin="/fixture/nsys",
                nsys_mount_root="/fixture", output_dir=str(tmp_path),
            )
        if tool == "advisor":
            return advisor.profile_cpu_entries(
                **common, advisor_bin="/fixture/advisor", advisor_root=None, image_tag="fixture-image",
            )
        return massif.profile(
            **common, global_error="", derived_image="fixture-image", output_dir=str(tmp_path), resume_existing=resume,
        )

    return profile, collections, task


@pytest.mark.parametrize("tool", ["ncu", "massif", "nsys", "advisor"])
def test_distinct_scales_get_distinct_artifacts(profile_fixture, tool):
    profile, collections, _task = profile_fixture
    result = profile(tool, SCALES)
    assert len(collections) == 2
    assert all("fixture collection boundary" in entry["error"] for entry in result["entries"])
    if tool == "ncu":
        artifacts = [command[command.index("-o") + 1] for command in collections]
    elif tool == "advisor":
        artifacts = [command[command.index("--project-dir") + 1] for command in collections]
    else:
        prefix = "--output=" if tool == "nsys" else "--massif-out-file="
        artifacts = [
            next(part.split("=", 1)[1] for part in command if part.startswith(prefix))
            for command in collections
        ]
    assert len(set(artifacts)) == 2
    assert [float(command[command.index("--input-scale") + 1]) for command in collections] == SCALES


@pytest.mark.parametrize("tool", ["ncu", "massif"])
@pytest.mark.parametrize(("checkpoint_scale", "recollect"), [(1.0, False), (math.nextafter(1.0, math.inf), True)])
def test_resume_requires_exact_checkpoint_scale(tmp_path, profile_fixture, tool, checkpoint_scale, recollect):
    profile, collections, task = profile_fixture
    prefix = "ncu_scale_1" if tool == "ncu" else "massif_cpu_2_mem_4_scale_1"
    raw = tmp_path / (prefix + (".csv" if tool == "ncu" else ".out"))
    raw.write_text("cached fixture\n")
    fields = ncu.NCU_COMPLETE_ENTRY_FIELDS if tool == "ncu" else massif.MASSIF_FIELDS
    checkpoint: dict[str, Any] = {
        "schema_version": 1, "model_id": task.model_id, "model_revision": task.model_revision,
        "input_scale": checkpoint_scale, "repeat": 1,
        "entry": {"input_scale": checkpoint_scale, **dict.fromkeys(fields, 42.0), "error": ""},
    }
    if tool == "ncu":
        checkpoint.update(
            image_tag="fixture-image", metrics=[NCU_METRIC, "gpu__time_duration.sum"],
            csv_size_bytes=raw.stat().st_size,
        )
        checkpoint["entry"]["flop_counting_version"] = 2
    else:
        checkpoint.update(derived_image="fixture-image", cpu_cores=2, mem_cap_gb=4, report_size_bytes=raw.stat().st_size)
    (tmp_path / (prefix + ".checkpoint.json")).write_text(json.dumps(checkpoint))

    result = profile(tool, [1.0], resume=True)

    assert len(collections) == int(recollect)
    if recollect:
        assert "fixture collection boundary" in result["entries"][0]["error"]
    else:
        assert result["entries"][0]["input_scale"] == 1.0
        assert result["entries"][0][fields[0]] == 42.0
