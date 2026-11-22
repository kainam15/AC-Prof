"""Offline result consumers must use input evidence from the recorded experiment."""
import hashlib
import json

import pytest

from acprof.cli import audit, posthoc
from acprof.host import run_state
from acprof.host.detect import TaskInfo
from acprof.host.images.runtime import ImageInfo
from acprof.host.input_plan import PlannedInputScales
from acprof.platform import Environment
from acprof.results.layers import publish_result_rows


@pytest.fixture
def recorded_input(tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "host_identity", lambda _: {"fixture": "fixed"})
    options = {"cpus": "2", "mems": "4", "gpus": "off", "warmup": 0, "repeat": 1}
    state = run_state.RunState(tmp_path / "experiment", options, resume=False, project_dir=".")
    layout = state.layout
    plan = layout.path("input_scale_plan.json")
    meta = layout.path("static_meta.json")
    image = ImageInfo("sha256:" + "a" * 64)
    task = TaskInfo("fixture/model", "fill-mask", "nlp", "transformers_pipeline",
                    "transformers", "b" * 40, "manual")
    try:
        plan.write_text(json.dumps({"schema_version": 2, "model_id": task.model_id, "entries": [
            {"input_scale": 8, "payload": {"text": "first [MASK]"}}]}))
        digest = hashlib.sha256(plan.read_bytes()).hexdigest()
        meta.write_text(json.dumps({"schema_version": 7, "model_name": task.model_id, **Environment("native_linux").metadata(),
            "model_revision": task.model_revision, "pipeline_tag": task.pipeline_tag,
            "task_family": task.task_family, "runtime_backend": task.runtime_backend,
            "image_id": image.tag, "input_scale_plan_sha256": digest}))
        state.bind_runtime(task, image, PlannedInputScales([8], "manual", str(plan), plan_sha256=digest), "", "")
        row = dict(cpu_cores="2", mem_cap_gb="4", gpu_mode="off", input_scale="8",
                   warmup="0", repeat_idx="0", status="ok", run_id=state.run_id, environment_class="native_linux", latency_app_s="1")
        publish_result_rows(list(row), [row], state.directory)
        state.finish(str(layout.result_csv))
    finally:
        state.close()
    return layout


@pytest.mark.parametrize("case", ["matching", "changed-payload", "state-only", "artifact-only", "conflicting-registry", "legacy", "registry-only-match"])
@pytest.mark.parametrize("command", ["audit", "profile"])
def test_consumers_check_the_frozen_plan(recorded_input, capsys, case, command):
    layout = recorded_input
    plan = layout.path("input_scale_plan.json")
    meta = layout.path("static_meta.json")
    state_file = layout.path("run_state.json")
    state = json.loads(state_file.read_text())
    metadata = json.loads(meta.read_text())
    if case in {"changed-payload", "state-only", "artifact-only"}:
        plan.write_text(plan.read_text().replace("first [MASK]", "other [MASK]"))
        if case == "changed-payload":
            metadata["input_scale_plan_sha256"] = hashlib.sha256(plan.read_bytes()).hexdigest()
        else:
            del metadata["input_scale_plan_sha256"]
        if case == "artifact-only":
            del state["runtime"]["planned"]["plan_sha256"]
        if case == "state-only":
            del state["artifacts"][str(plan.relative_to(layout.root))]
    elif case in {"legacy", "registry-only-match"}:
        del metadata["input_scale_plan_sha256"]
        del state["runtime"]["planned"]["plan_sha256"]
        if case == "legacy":
            del state["artifacts"]
    elif case == "conflicting-registry":
        state["artifacts"][str(plan.relative_to(layout.root))] = "f" * 64
    meta.write_text(json.dumps(metadata))
    state_file.write_text(json.dumps(state))
    valid = case in {"matching", "legacy", "registry-only-match"}
    if command == "audit":
        assert audit.main([str(layout.root), "--require-complete", "--require-ok", "--json"]) == (0 if valid else 1)
        result = json.loads(capsys.readouterr().out)
        assert result["accepted"] == valid
        assert result["counts"]["formal_ok"] == 1
        assert ("input_plan_hash" in {issue["code"] for issue in result["issues"]}) != valid
        return
    args = [str(layout.root), "--tools", "torch", "--dry-run"]
    if valid:
        posthoc.main(args)
    else:
        with pytest.raises(SystemExit) as stopped:
            posthoc.main(args)
        assert stopped.value.code == 2
        assert "SHA256" in capsys.readouterr().err
