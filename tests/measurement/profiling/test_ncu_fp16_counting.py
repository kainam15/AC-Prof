"""Packed FP16 counters count thread instructions, with two lanes each."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.host.detect import TaskInfo
from acprof.host.posthoc.plans import compute_plan_covers_tool
from acprof.host.profiling.tools import compute_parsers, ncu

PREFIX = "smsp__sass_thread_inst_executed_op_"


def write_counters(path: Path, metrics: dict[str, int], layout: str = "wide") -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if layout == "wide":
            writer.writerow(["ID", "Kernel Name", *metrics, "gpu__time_duration.sum"])
            writer.writerow(["", "", *["inst" for _ in metrics], "usec"])
            writer.writerow(["0", "packed_half", *metrics.values(), "10"])
        else:
            writer.writerow(["ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
            for name, value in metrics.items():
                writer.writerow(["0", "packed_half", name, "inst", value])
            writer.writerow(["0", "packed_half", "gpu__time_duration.sum", "usec", "10"])


@pytest.mark.parametrize("layout", ["long", "wide"])
@pytest.mark.parametrize(("operation", "expected"), [("hadd", 256), ("hmul", 256), ("hfma", 512)])
def test_half_instruction_counts_both_lanes(tmp_path, layout, operation, expected):
    path = tmp_path / "half.csv"
    write_counters(path, {f"{PREFIX}{operation}_pred_on.sum": 128}, layout)

    result = compute_parsers.parse_ncu_profile_csv(str(path))

    assert result["scalar_flops_per_request"] == expected
    assert result["total_flops_per_request"] == expected


def test_half_weights_preserve_other_precisions_and_repeat_normalization(tmp_path):
    path = tmp_path / "mixed.csv"
    metrics = {f"{PREFIX}{op}_pred_on.sum": 128 for op in ("hadd", "hmul", "hfma")}
    for precision in ("f", "d"):
        for operation, count in (("add", 10), ("mul", 20), ("fma", 30)):
            metrics[f"{PREFIX}{precision}{operation}_pred_on.sum"] = count
    metrics["sm__ops_path_tensor_src_fp16_dst_fp32.sum"] = 4096
    # Aggregate companions in a wide raw report must not be counted again.
    metrics[f"{PREFIX}hadd_pred_on.avg"] = 999
    write_counters(path, metrics)

    result = compute_parsers.parse_ncu_profile_csv(str(path), repeat=2)

    assert result["scalar_flops_per_request"] == 602
    assert result["tensor_flops_per_request"] == 2048
    assert result["total_flops_per_request"] == 2650
    assert result["tensor_share_pct"] == pytest.approx(2048 / 2650 * 100)
    assert result["kernel_launch_count_per_request"] == 0.5
    assert result["kernel_time_sum_ms_per_request"] == 0.005


@pytest.mark.parametrize("valid_csv", [True, False])
def test_legacy_checkpoint_reparses_raw_counts_and_preserves_gpu_identity(tmp_path, valid_csv):
    task = TaskInfo(
        model_id="example/model", model_revision="revision-1",
        task_family="nlp", pipeline_tag="fill-mask",
        runtime_backend="transformers_pipeline", library_name="transformers",
        detection_method="hub_api",
    )
    root = tmp_path / "compute_profiles"
    root.mkdir()
    raw = root / "ncu_scale_1.csv"
    counters = {f"{PREFIX}{op}_pred_on.sum": 128 for op in ("hadd", "hmul", "hfma")}
    write_counters(raw, counters)
    if not valid_csv:
        raw.write_text("incomplete raw report\n")
    original_raw = raw.read_bytes()
    # Independent legacy fixture: the old formula counted 128 + 128 + 2*128.
    old_entry = {
        "input_scale": 1.0, "tool": "ncu", "error": "",
        "gpu_executed_mflop_per_request_ncu": 0.000512,
        "gpu_executed_scalar_mflop_per_request_ncu": 0.000512,
        "gpu_executed_tensor_mflop_per_request_ncu": 0,
        "gpu_executed_tensor_share_pct_ncu": 0,
        "gpu_kernel_launch_count_per_request_ncu": 1,
        "gpu_kernel_time_sum_ms_per_request_ncu": 0.01,
        "gpu_compute_capability": "8.9", "gpu_sm_count": 24,
    }
    from acprof.gpu_probe import gpu_probe_report, set_gpu_probe_value

    metadata = gpu_probe_report(("gpu_compute_capability", "gpu_sm_count"), status="unknown")
    metadata["device"] = {"cuda_uuid": "captured-device", "cuda_index": 0}
    set_gpu_probe_value(metadata, "gpu_compute_capability", "8.9", "runner")
    set_gpu_probe_value(metadata, "gpu_sm_count", 24, "runner")
    old_entry["gpu_compute_metadata"] = metadata
    checkpoint = root / "ncu_scale_1.checkpoint.json"
    ncu._write_ncu_checkpoint(
        checkpoint_path=str(checkpoint), task_info=task, image_tag="sha256:original",
        input_scale=1, repeat=1, metrics=list(counters), host_csv=str(raw), entry=old_entry,
    )
    original_checkpoint = checkpoint.read_bytes()
    kwargs = dict(
        ncu_bin="/opt/ncu", ncu_metrics=list(counters), task_info=task,
        image_tag="sha256:original", cpu=1, mem=4, payload_file="unused.json",
        profile_root=str(root), tool_mount_roots=(), entry={"input_scale": 1.0}, repeat=1,
    )
    with patch.object(ncu, "profiler_container_command", side_effect=AssertionError("no recollection")):
        result = ncu._resume_ncu_for_entry(**kwargs)
    assert raw.read_bytes() == original_raw
    if not valid_csv:
        assert result is None
        assert checkpoint.read_bytes() == original_checkpoint
        return

    assert result["gpu_executed_mflop_per_request_ncu"] == 0.001024
    assert result["gpu_executed_scalar_mflop_per_request_ncu"] == 0.001024
    assert result["flop_counting_version"] == 2
    assert result["gpu_compute_metadata"] == metadata
    assert result["gpu_compute_capability"] == "8.9"
    assert result["gpu_sm_count"] == 24
    assert json.loads(checkpoint.read_text())["entry"] == result
    with patch.object(ncu, "_ncu_entry_from_csv", side_effect=AssertionError("already current")):
        assert ncu._resume_ncu_for_entry(**kwargs) == result


@pytest.mark.parametrize("version", [None, 1, 2])
def test_only_current_ncu_formula_plan_is_reusable(version):
    entry = {field: 1.0 for field in ncu.NCU_COMPLETE_ENTRY_FIELDS}
    entry.update(input_scale=1.0, error="")
    profile = {"tool": "ncu", "entries": [entry], "error": ""}
    if version is not None:
        profile["flop_counting_version"] = version
    plan = {"profiles": {"gpu": {"ncu": profile}}}
    context = SimpleNamespace(scales_by_mode={"on": [1.0]})

    assert compute_plan_covers_tool(plan, context, "ncu") is (version == 2)
