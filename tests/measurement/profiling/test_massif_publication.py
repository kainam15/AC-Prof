"""Massif recovery publishes the report only after its producer succeeds."""
import json
import math
from contextlib import nullcontext
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from acprof.host.detect import TaskInfo
from acprof.host.profiling import execution
from acprof.host.profiling.tools import massif


@pytest.fixture
def task(monkeypatch):
    monkeypatch.setattr(massif, "profiler_container_command", lambda **kwargs: nullcontext(["fixture"]))
    return TaskInfo(model_id="fixture/model", model_revision="fixed", pipeline_tag="fill-mask", task_family="nlp",
                    runtime_backend="transformers_pipeline", library_name="transformers", detection_method="fixture")


def output_path(root, argv):
    for argument in argv:
        if argument.startswith("--massif-out-file="):
            return root / Path(argument.split("=", 1)[1]).name
    raise AssertionError("Massif output path missing")


@pytest.mark.parametrize("first_failure", ["exit", "interrupt", "invalid", "checkpoint_interrupted"])
def test_resume_retries_unsuccessful_report(tmp_path, monkeypatch, task, first_failure):
    artifacts = tmp_path / "profiles"
    artifacts.mkdir()
    paths = []

    def command(argv, **kwargs):
        path = output_path(artifacts, argv)
        paths.append(path)
        heap = 100 * len(paths)
        text = f"time_unit: ms\nsnapshot=0\ntime=2\nmem_heap_B={heap}\nmem_heap_extra_B=4\nmem_stacks_B=8\n"
        path.write_text("time_unit: ms\n" if len(paths) == 1 and first_failure == "invalid" else text)
        if len(paths) == 1 and first_failure == "interrupt":
            raise KeyboardInterrupt()
        return CompletedProcess(argv, 1 if len(paths) == 1 and first_failure == "exit" else 0, "", "fixture error")

    monkeypatch.setattr(massif, "run_profiler_command", command)

    def collect():
        return massif.profile(entries=[{"input_scale": 8}], global_error="", task_info=task, derived_image="fixture",
                              cpu=2, mem=4, payload_file="unused", profile_root=str(artifacts),
                              output_dir=str(tmp_path), repeat=1, resume_existing=True)["entries"][0]

    if first_failure == "checkpoint_interrupted":
        with monkeypatch.context() as interrupted:
            def stop_publication(**kwargs):
                raise KeyboardInterrupt()
            interrupted.setattr(massif, "_write_massif_checkpoint", stop_publication)
            with pytest.raises(KeyboardInterrupt):
                collect()
        assert (artifacts / "massif_cpu_2_mem_4_scale_8.out").is_file()
    elif first_failure == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            collect()
    else:
        failed = collect()
        assert failed["error"]
        assert tmp_path / failed["report"] == paths[0]
    diagnostic = paths[0].read_bytes() if first_failure != "checkpoint_interrupted" else None
    new = collect()
    assert new["error"] == ""
    assert new["cpu_heap_peak_total_bytes_massif"] == 212
    if diagnostic is not None:
        assert paths[0].read_bytes() == diagnostic
    assert Path(new["report"]).name == "massif_cpu_2_mem_4_scale_8.out"
    assert collect() == new
    assert len(paths) == 2


@pytest.mark.parametrize(("recorded_error", "changed_report"), [
    ("massif_failed:fixture import failed", False), ("", True), ("", False),
])
def test_previous_plan_controls_recovery_of_uncheckpointed_output(tmp_path, monkeypatch, task, recorded_error, changed_report):
    payload = tmp_path / "input_scale_plan.json"
    payload.write_text(json.dumps({"schema_version": 2, "entries": [{"input_scale": 8, "payload": {"text": "hello"}}]}))
    raw = tmp_path / "execution_profiles" / "massif_cpu_2_mem_4_scale_8.out"
    raw.parent.mkdir()
    recorded = "time_unit: ms\nsnapshot=0\ntime=2\nmem_heap_B=100\nmem_heap_extra_B=4\nmem_stacks_B=8\n"
    raw.write_text(recorded)
    entry = {"input_scale": 8, "error": recorded_error, "report": str(raw.relative_to(tmp_path))}
    if not recorded_error:
        entry.update(massif.parse_massif_output(str(raw)))
    if changed_report:
        raw.write_text(recorded.replace("time=2", "time=9"))
    plan = tmp_path / "execution_profile_plan.json"
    plan.write_text(json.dumps({"model_id": task.model_id, "model_revision": task.model_revision,
        "execution_profile_tool_mode": "massif", "profiles": [{"cpu_cores": 2, "mem_cap_gb": 4, "gpu_mode": "off",
        "tools": {"massif": {"tool": "massif", "entries": [entry]}}}]}))
    attempts = []

    def command(argv, **kwargs):
        path = output_path(raw.parent, argv)
        attempts.append(path)
        path.write_text(recorded.replace("mem_heap_B=100", "mem_heap_B=200"))
        return CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(massif, "run_profiler_command", command)
    monkeypatch.setattr(execution, "_prepare_massif", lambda *args: execution._MassifRuntime("fixture", "", "fixture"))
    request = execution.ExecutionProfileRequest(task, "fixture", [2], [4], ["off"], str(tmp_path), str(payload))
    options = execution.ExecutionProfileOptions(tool_mode="massif", resume_existing_profiles=True)
    published = json.loads(Path(execution.collect_execution_profile_plan(request, options)).read_text())
    result = published["profiles"][0]["tools"]["massif"]["entries"][0]
    assert result["error"] == ""
    recollect = bool(recorded_error) or changed_report
    assert result["cpu_heap_peak_total_bytes_massif"] == (212 if recollect else 112)
    assert len(attempts) == int(recollect)


@pytest.mark.parametrize("change", ["none", "same_size_raw", "size", "entry_scale", "entry_error", "typed_error",
                                    *massif.MASSIF_FIELDS])
def test_checkpoint_and_report_must_describe_the_same_observation(tmp_path, monkeypatch, task, change):
    calls = []
    def producer(argv, **kwargs):
        path = output_path(tmp_path, argv)
        calls.append(path)
        path.write_text(f"time_unit: ms\nsnapshot=0\ntime=2\nmem_heap_B={100 * len(calls)}\n"
                        "mem_heap_extra_B=4\nmem_stacks_B=8\n")
        return CompletedProcess(argv, 0, "", "")

    def collect():
        return massif.profile(entries=[{"input_scale": 8}], global_error="", task_info=task, derived_image="fixture",
                              cpu=2, mem=4, payload_file="unused", profile_root=str(tmp_path),
                              output_dir=str(tmp_path), repeat=1, resume_existing=True)["entries"][0]

    monkeypatch.setattr(massif, "run_profiler_command", producer)
    first = collect()
    p, = tmp_path.glob("*.checkpoint.json")
    checkpoint = json.loads(p.read_bytes())
    raw = tmp_path / first["report"]
    if change == "same_size_raw":
        raw.write_bytes(raw.read_bytes().replace(b"time=2", b"time=9"))
    elif change == "size":
        checkpoint["report_size_bytes"] += 0.5
    elif change == "entry_scale":
        checkpoint["entry"]["input_scale"] = math.nextafter(8.0, math.inf)
    elif change in ("entry_error", "typed_error"):
        field = "error" if change == "entry_error" else "compute_profile_error_massif"
        checkpoint["entry"][field] = "massif_failed:incomplete"
    elif change in massif.MASSIF_FIELDS:
        checkpoint["entry"][change] += 1
    p.write_text(json.dumps(checkpoint))

    result = collect()
    assert not result["error"]
    assert result["input_scale"] == 8
    assert result["cpu_heap_peak_at_ms_massif"] == 2
    assert len(calls) == (1 if change == "none" else 2)
    assert result["cpu_heap_peak_total_bytes_massif"] == (112 if change == "none" else 212)
