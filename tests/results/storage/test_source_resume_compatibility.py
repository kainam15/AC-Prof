"""Frozen Hugging Face experiments retain their recorded source and decision."""

import copy
import json
from argparse import Namespace
from unittest.mock import Mock

import pytest

from acprof.host.automation import AutomaticRun
from acprof.host.detect import TaskInfo
from acprof.host.doctor import DoctorCheck
from acprof.host.images.runtime import ImageInfo
from acprof.host.input_plan import PlannedInputScales
from acprof.host.matrix_plan import freeze_matrix_plan, matrix_identity
from acprof.host.run_state import RunState, RunStateError, load_run_state
from acprof.results.layout import ArtifactLayout

REVISION = "a" * 40
IMAGE_ID = "sha256:" + "b" * 64
EMPTY_OBJECT_SHA256 = "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"


@pytest.fixture
def historical_task():
    # Serialized before model_source/requested_revision existed. Keep the recorded
    # resolver identity literal; the current resolver must not generate it.
    provenance = {
        "schema_version": 1, "resolver_version": "model-selection-v3",
        "model_id": "example/model", "revision": REVISION,
        "sources": {
            "repository_snapshot": {
                "model_id": "example/model", "revision": REVISION,
                "files_sha256": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
                "derived_from": [],
            },
            "hub": {
                "sha256": "0de9cdf8c8e84f115cfd908cc7cf7269d51cb7ff70dedfe2de47226366a7959d",
                "revision": REVISION, "family": "repository_metadata",
                "derived_from": ["repository_snapshot"],
            },
            "repository_config": {
                "sha256": EMPTY_OBJECT_SHA256, "revision": REVISION,
                "family": "repository_metadata", "derived_from": ["repository_snapshot"],
            },
            "repository_spec": {
                "sha256": EMPTY_OBJECT_SHA256, "revision": REVISION,
                "family": "declaration", "derived_from": ["repository_snapshot"],
            },
            "local_spec": {
                "sha256": EMPTY_OBJECT_SHA256, "revision": REVISION,
                "family": "user", "derived_from": [],
            },
            "explicit": {
                "sha256": "593ca37e52be6c289ead6101c1e089bbf0f343bdb5232b679dd655dfc3b7ad64",
                "revision": REVISION, "family": "user", "derived_from": [],
            },
            "generated_contract": {
                "sha256": EMPTY_OBJECT_SHA256, "revision": REVISION,
                "family": "derived", "derived_from": ["repository_config"],
            },
        },
        "observations": [{
            "field": "manual", "source_id": "repository_config",
            "task": "fill-mask", "backend": "transformers_pipeline",
        }],
        "selected_task": "fill-mask", "status": "candidate", "overridden_conflicts": [],
        "identity_sha256": "0c20896e34f56ca77c72d75a082ca3a316e27b036b535ff67ff5d9f0d5ada9fb",
        "decision": "consistent_evidence",
    }
    return {
        "model_id": "example/model", "pipeline_tag": "fill-mask", "task_family": "nlp",
        "runtime_backend": "transformers_pipeline", "library_name": "transformers",
        "model_revision": REVISION, "detection_method": "manual",
        "model_resolution": {"schema_version": 1, "status": "candidate", "provenance": provenance},
    }


def frozen_historical_layout(root, task, monkeypatch):
    """Build valid state through the public writer, retaining literal old TaskInfo."""
    monkeypatch.setattr("acprof.host.run_state.host_identity", lambda _: {})
    options = {"profiling_mode": "basic", "cpus": "1", "mems": "4", "gpus": "off", "warmup": 0, "repeat": 1}
    state = RunState(root, options, resume=False, project_dir=".")
    try:
        state.bind_runtime(TaskInfo(**task), ImageInfo(IMAGE_ID),
                           PlannedInputScales([1.0], "fixture"), "", "")
    finally:
        state.close("failed")
    path = state.layout.path("run_state.json")
    snapshot = json.loads(path.read_text())
    # Do not let current dataclass defaults rewrite the historical source evidence.
    snapshot["runtime"]["task"] = copy.deepcopy(task)
    path.write_text(json.dumps(snapshot))
    assert load_run_state(root)["runtime"]["task"] == task
    return state.layout


@pytest.mark.parametrize("grouped", [False, True])
def test_automatic_resume_preserves_historical_decision_without_resolving(
        tmp_path, monkeypatch, historical_task, grouped):
    root = tmp_path / "example--model"
    root.mkdir()
    if grouped:
        state_path = root / "metadata" / "task_runs" / "frozen.json"
        state_path.parent.mkdir(parents=True)
        state_path.write_text(json.dumps({"schema_version": 1, "task_snapshot": historical_task}))
    else:
        layout = frozen_historical_layout(root, historical_task, monkeypatch)
        state_path = layout.path("run_state.json")
    layout = ArtifactLayout.discover(root)
    layout.path("auto_report.json").write_text(json.dumps({
        "schema_version": 1, "requested_profiling_mode": "basic", "status": "failed",
    }))
    original = state_path.read_bytes()
    expected = copy.deepcopy(historical_task["model_resolution"]["provenance"])
    for target in (
        "acprof.host.detect.detect_task",
        "acprof.host.automation.check_repository_access",
        "acprof.models.evidence.resolution_provenance",
    ):
        monkeypatch.setattr(target, Mock(side_effect=AssertionError("frozen evidence was resolved again")))
    monkeypatch.setattr("acprof.host.doctor.collect_checks",
                        lambda **_: [DoctorCheck("docker", "available", "fixture")])
    args = Namespace(model="example/model", output_dir=str(tmp_path), resume=True, revision=None,
                     profiling_mode="basic", batch_size=1, gpus="off", sniff_iface="docker0")

    restored = AutomaticRun(args).prepare()

    assert restored.model_source == "huggingface"
    assert restored.model_resolution["provenance"] == expected
    assert args.resolution_identity == expected["identity_sha256"]
    saved = json.loads(layout.path("model_resolution.json").read_text())
    assert saved["provenance"] == expected
    assert state_path.read_bytes() == original


def test_automatic_resume_rejects_missing_identity_before_writes(tmp_path, monkeypatch, historical_task):
    root = tmp_path / "example--model"
    layout = frozen_historical_layout(root, historical_task, monkeypatch)
    state_path = layout.path("run_state.json")
    snapshot = json.loads(state_path.read_text())
    del snapshot["run_id"]
    state_path.write_text(json.dumps(snapshot))
    layout.path("auto_report.json").write_text(json.dumps({
        "schema_version": 1, "requested_profiling_mode": "basic", "status": "failed",
    }))
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    access = Mock(side_effect=AssertionError("invalid resume must not query the repository"))
    monkeypatch.setattr("acprof.host.automation.check_repository_access", access)
    args = Namespace(model="example/model", output_dir=str(tmp_path), resume=True, revision=None,
                     profiling_mode="basic", batch_size=1, gpus="off", sniff_iface="docker0")
    run = AutomaticRun(args)
    with pytest.raises(RunStateError, match="run_id"):
        run.prepare()
    access.assert_not_called()
    assert not run.started
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("payload", [
    '{"schema_version": 1, "extra": NaN}',
    '{"schema_version": 1, "extra": 1e999}',
])
def test_matrix_plan_rejects_non_finite_json_before_resume(tmp_path, historical_task, payload):
    path = tmp_path / "matrix_plan.json"
    path.write_text(payload)
    task = TaskInfo(**historical_task)
    image = ImageInfo(IMAGE_ID)
    identity = matrix_identity(task, image, [1], [4], ["off"], [16.0],
                               order="seeded", seed=37, prune=False)
    with pytest.raises(ValueError, match="invalid matrix plan JSON"):
        freeze_matrix_plan(path, identity, {})


def test_matrix_plan_rejects_oversized_json_before_resume(tmp_path, historical_task):
    path = tmp_path / "matrix_plan.json"
    path.write_text('{"schema_version":1,"padding":"' + ("x" * (4 * 1024 * 1024)) + '"}')
    task = TaskInfo(**historical_task)
    image = ImageInfo(IMAGE_ID)
    identity = matrix_identity(task, image, [1], [4], ["off"], [16.0],
                               order="seeded", seed=37, prune=False)
    with pytest.raises(ValueError, match="matrix plan exceeds the 4 MiB read limit"):
        freeze_matrix_plan(path, identity, {})


def test_historical_matrix_is_reused_for_hf_and_rejects_modelscope(
        tmp_path, monkeypatch, historical_task):
    # Golden v1 plan generated before source support; hashes and order are fixed.
    plan = {
        "schema_version": 1, "algorithm_version": "sha256-sort-v1",
        "matrix_order": "seeded", "seed": 37,
        "identity": {
            "model_id": "example/model", "model_revision": REVISION, "image_id": IMAGE_ID,
            "cpus": [2, 1], "mems": [4], "gpus": ["off"], "input_scales": [16.0, 32.0],
            "order": "seeded", "seed": 37, "prune_startup_oom": False, "input_plan_sha256": "",
        },
        "startup_oom_prefixes": {},
        "cases": [
            {"cpu_cores": 1, "mem_cap_gb": 4, "gpu_mode": "off",
             "input_scale_seed": "42199da9c4ecaf5cd61b87e08ba0363326e17db12bc001643ea45b7fe9566513",
             "input_scales": [16.0, 32.0], "result_origin": "formal_measurement"},
            {"cpu_cores": 2, "mem_cap_gb": 4, "gpu_mode": "off",
             "input_scale_seed": "f1cc4f4cf33b44f4119fc7395a67bacb8a80217a6f44fa47d30bd7da6022076c",
             "input_scales": [16.0, 32.0], "result_origin": "formal_measurement"},
        ],
        "plan_sha256": "16064b47653dccd9c1218f3db757b3eb2ac1ebdc10300e09dbd3be128c1a7a67",
    }
    path = tmp_path / "matrix_plan.json"
    path.write_text(json.dumps(plan))
    original = path.read_bytes()
    monkeypatch.setattr("acprof.host.matrix_plan.build_matrix_plan",
                        Mock(side_effect=AssertionError("frozen matrix was rebuilt")))
    task = TaskInfo(**historical_task)
    image = ImageInfo(IMAGE_ID)
    identity = matrix_identity(task, image, [2, 1], [4], ["off"], [16.0, 32.0],
                               order="seeded", seed=37, prune=False)
    assert freeze_matrix_plan(path, identity, {}) == plan
    assert path.read_bytes() == original

    task.model_source = "modelscope"
    changed = matrix_identity(task, image, [2, 1], [4], ["off"], [16.0, 32.0],
                              order="seeded", seed=37, prune=False)
    with pytest.raises(ValueError, match="matrix plan identity"):
        freeze_matrix_plan(path, changed, {})
    assert path.read_bytes() == original
