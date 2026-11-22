"""Inspection exposes provenance without starting measurement or loading models."""
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import tests.models.contracts.test_model_contract as fixture
from acprof.cli.main import main


@pytest.mark.parametrize("mode", ["basic", "full", "none"])
def test_removed_probe_mode_is_rejected_before_model_lookup(mode):
    with patch("acprof.host.detect.detect_task") as detect, pytest.raises(SystemExit) as exc:
        main(["inspect", "example/model", "--probe", mode])
    assert exc.value.code == 2
    detect.assert_not_called()


def test_interface_timeout_is_reported_as_a_failed_cli_command(tmp_path):
    import subprocess
    task = fixture.TestModelContract().discover()
    with patch("acprof.host.detect.detect_task", return_value=task), patch(
        "acprof.host.env_utils.bootstrap_project_env"), patch("acprof.host.preflight.require_collection_host"), patch(
        "acprof.host.preflight.require_native_docker"), patch("acprof.host.images.runtime.configure_runtime_profile"), patch(
        "acprof.host.run_state.MeasurementLock"), patch(
        "acprof.host.preparation.interface_probe.probe_interface", side_effect=subprocess.TimeoutExpired("docker", 1)):
        assert main(["inspect", task.model_id, "--probe-interface", "--output-dir", str(tmp_path)]) == 1


@pytest.mark.parametrize(("family", "tag", "expected_scale", "field"), [('nlp', 'fill-mask', 64, 'text'), ('cv', 'image-classification', 0.1, 'image_base64')])
def test_native_probe_uses_family_defaults_when_generator_has_no_declared_scales(family, tag, expected_scale, field):
    from types import SimpleNamespace

    from acprof.host.detect import TaskInfo
    from acprof.host.preparation.model_inspection import validate_model_runtime
    task = TaskInfo("example/native", tag, family, "transformers_pipeline", "transformers", "a" * 40, "hub")
    with tempfile.TemporaryDirectory() as directory, patch(
            "acprof.host.images.runtime.prepare_image", return_value=SimpleNamespace(tag="sha256:" + "b" * 64)), patch(
            "acprof.host.preflight.require_collection_host"), patch(
            "acprof.host.preflight.require_native_docker"), patch(
            "acprof.host.run_state.MeasurementLock"), patch(
            "acprof.host.runtime_validation.validate_runtime", return_value={"status": "ok"}) as validate:
        validate_model_runtime(task, directory)
        entry = json.loads(Path(validate.call_args.kwargs["planned"].plan_file).read_text())["entries"][0]
        assert (entry["input_scale"]) == (expected_scale)
        assert (entry["payload"][field])
        if family == "nlp":
            assert ("[MASK]") in (entry["payload"][field])


@pytest.mark.parametrize("case", ["not_entered", "released", "competing", "other_thread"])
def test_runtime_validation_requires_an_active_measurement_scope(tmp_path, case):
    from concurrent.futures import ThreadPoolExecutor

    from acprof.host import run_state
    from acprof.host.preparation.model_inspection import validate_model_runtime

    lock = run_state.MeasurementLock()
    task = fixture.TestModelContract().discover()
    if case == "released":
        with lock:
            pass
    with (lock if case in {"competing", "other_thread"} else contextlib.nullcontext()), patch(
        "acprof.host.preflight.require_collection_host"
    ), patch("acprof.host.preflight.require_native_docker"), patch(
        "acprof.host.images.runtime.prepare_image", side_effect=AssertionError("rejected probe prepared an image")
    ) as prepare:
        if case == "other_thread":
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(validate_model_runtime, task, tmp_path, measurement_lock=lock)
                with pytest.raises(run_state.RunStateError):
                    future.result(timeout=5)
        elif case == "competing":
            with pytest.raises(run_state.RunStateError):
                validate_model_runtime(task, tmp_path)
        else:
            with pytest.raises(run_state.RunStateError):
                validate_model_runtime(task, tmp_path, measurement_lock=lock)
    prepare.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_full_probe_accepts_native_models_and_preserves_feature_width():
    from types import SimpleNamespace

    from acprof.host.detect import TaskInfo
    from acprof.host.preparation.model_inspection import validate_model_runtime
    task = TaskInfo("example/iris", "tabular-classification", "structured", "onnxruntime",
                    "onnx", "a" * 40, "manual")
    task.model_spec = {"schema_version": 1, "format": "onnxruntime", "task": "tabular-classification",
                       "model_file": "iris.onnx", "feature_dim": 4}
    with tempfile.TemporaryDirectory() as directory, patch("acprof.host.images.runtime.prepare_image",
            return_value=SimpleNamespace(tag="sha256:" + "b" * 64)), patch(
            "acprof.host.preflight.require_collection_host"), patch(
            "acprof.host.preflight.require_native_docker"), patch(
            "acprof.host.run_state.MeasurementLock"), patch(
            "acprof.host.runtime_validation.validate_runtime", return_value={"status": "ok"}) as validate:
        report = validate_model_runtime(task, directory)
        assert (report["status"]) == ("ok")
        plan = json.loads(Path(validate.call_args.kwargs["planned"].plan_file).read_text())
        assert (len(plan["entries"][0]["payload"]["features"][0])) == (4)
        assert ("params") not in (plan["entries"][0]["payload"])

def test_review_provenance_survives_export_and_subprocess_inspection(inspection_layout):
    from acprof.models.contract import write_model_resolution
    from acprof.models.spec import task_model_spec
    task = fixture.TestModelContract().discover()
    original = task.model_resolution["contract"]
    declaration = task_model_spec(task)
    author_task = fixture.TestModelContract().discover(spec=declaration)
    directory = inspection_layout.root
    with patch("acprof.host.detect.detect_task", return_value=author_task), patch(
        "acprof.host.env_utils.bootstrap_project_env",
    ), contextlib.redirect_stdout(io.StringIO()):
        write_model_resolution(task, directory)
        path = Path(directory, "acprof_model.json")
        path.write_text(json.dumps(declaration))
        assert (main(["inspect", task.model_id, "--model-spec", str(path),
                               "--expected-revision", task.model_revision, "--output-dir", str(directory)])) == (0)
        saved = json.loads(inspection_layout.path("model_resolution.json").read_text())["contract"]
        assert (saved["cache_key"]) == (original["cache_key"])
        assert (saved["fields"]["multimodal.inputs.prompt"]["sources"]) == (original["fields"]["multimodal.inputs.prompt"]["sources"])


@pytest.mark.parametrize("previous_contract", ["nonfinite-report", "v5-source-graph"])
def test_invalid_or_outdated_review_provenance_is_not_reused(previous_contract):
    from acprof.models.contract import write_model_resolution
    from acprof.models.spec import task_model_spec

    task = fixture.TestModelContract().discover()
    declaration = task_model_spec(task)
    author_task = fixture.TestModelContract().discover(spec=declaration)
    with tempfile.TemporaryDirectory() as directory, patch(
        "acprof.host.detect.detect_task", return_value=author_task
    ), patch("acprof.host.env_utils.bootstrap_project_env"), contextlib.redirect_stdout(io.StringIO()):
        write_model_resolution(task, directory)
        previous = Path(directory, "model_resolution.json")
        payload = json.loads(previous.read_text())
        payload["contract"]["legacy_marker"] = "must-not-survive"
        if previous_contract == "v5-source-graph":
            payload["contract"]["resolver_version"] = "pipeline-contract-v5"
        else:
            payload["corrupt_metric"] = float("nan")
        previous.write_text(json.dumps(payload), encoding="utf-8")
        declaration_path = Path(directory, "acprof_model.json")
        declaration_path.write_text(json.dumps(declaration), encoding="utf-8")

        assert main([
            "inspect", task.model_id, "--model-spec", str(declaration_path),
            "--expected-revision", task.model_revision, "--output-dir", directory,
        ]) == 0

        saved = json.loads(previous.read_text())["contract"]
        assert "legacy_marker" not in saved

def test_changed_revision_refuses_probe_and_preserves_previous_report():
    task = fixture.TestModelContract().discover()
    with tempfile.TemporaryDirectory() as directory, patch("acprof.host.detect.detect_task", return_value=task), patch(
        "acprof.host.env_utils.bootstrap_project_env",
    ), patch("acprof.host.images.runtime.prepare_image") as build:
        path = Path(directory, "model_resolution.json")
        path.write_text('{"preserve":true}')
        assert (main(["inspect", task.model_id, "--expected-revision", "b" * 40,
                               "--probe-interface", "--output-dir", directory])) == (2)
        assert (path.read_text()) == ('{"preserve":true}')
        build.assert_not_called()

def test_inspect_explains_and_exports_the_static_contract():
    task = fixture.TestModelContract().discover(transformers_info=fixture.GENERIC_LOADER)
    with tempfile.TemporaryDirectory() as directory, patch("acprof.host.detect.detect_task", return_value=task), patch(
        "acprof.host.env_utils.bootstrap_project_env",
    ), patch("acprof.host.images.runtime.prepare_image") as build, contextlib.redirect_stdout(io.StringIO()) as output:
        code = main(["inspect", task.model_id, "--explain", "--output-dir", directory])
        assert (code) == (0)
        report = json.loads(Path(directory, "model_resolution.json").read_text())
        assert (report["contract"]["runtime_validation"]) == ("not_run")
        assert ("multimodal.inputs.prompt") in (output.getvalue())
        assert ("pipeline.py") in (output.getvalue())
        assert ("feature-extraction [loader_hint]") in (output.getvalue())
        assert ("not a task declaration") in (output.getvalue())
        build.assert_not_called()

def test_unresolved_inspection_exports_draft_and_returns_two():
    task = fixture.TestModelContract().discover(fixture.SOURCE.replace('inputs.get("prompt", "Listen.")', 'inputs["turns"]'))
    with tempfile.TemporaryDirectory() as directory, patch("acprof.host.detect.detect_task", return_value=task), patch(
        "acprof.host.env_utils.bootstrap_project_env",
    ), contextlib.redirect_stdout(io.StringIO()) as output:
        assert (main(["inspect", task.model_id, "--output-dir", directory])) == (2)
        assert ("multimodal.inputs.turns") in (output.getvalue())
        assert (Path(directory, "model_resolution.json").is_file())


@pytest.fixture(params=["flat", "v2"])
def inspection_layout(request, tmp_path):
    from acprof.results.layout import MANIFEST_NAME, ArtifactLayout, _manifest

    if request.param == "v2":
        (tmp_path / MANIFEST_NAME).write_text(json.dumps(_manifest()))
        (tmp_path / "metadata").mkdir()
    return ArtifactLayout.discover(tmp_path)


@pytest.mark.parametrize("probe", [False, True])
def test_busy_inspection_directory_preserves_resolution(inspection_layout, probe):
    from acprof.host.run_state import ResultDirectoryLock

    task = fixture.TestModelContract().discover()
    resolution = inspection_layout.path("model_resolution.json")
    resolution.write_bytes(b"previous resolution")
    args = ["inspect", task.model_id, "--output-dir", str(inspection_layout.root)]
    if probe:
        args.append("--probe-interface")
    with ResultDirectoryLock(inspection_layout.root), patch("acprof.host.detect.detect_task", return_value=task), patch(
        "acprof.host.preflight.require_collection_host"), patch("acprof.host.preflight.require_native_docker"), patch(
        "acprof.host.run_state.MeasurementLock"), patch("acprof.host.preparation.interface_probe.probe_interface") as run:
        code = main(args)
    assert resolution.read_bytes() == b"previous resolution"
    assert code == 1
    run.assert_not_called()


@pytest.mark.parametrize(("marker", "probe"), [("interface_validation.json", True), ("interface_validation.json", False),
                                         ("run_state.json", True), ("runtime_validation.json", False),
                                         ("result_layers.json", False)])
def test_completed_evidence_survives_another_inspection(inspection_layout, marker, probe):
    task = fixture.TestModelContract().discover()
    resolution = inspection_layout.path("model_resolution.json")
    evidence = inspection_layout.path(marker)
    evidence.parent.mkdir(parents=True, exist_ok=True)
    resolution.write_bytes(b"previous resolution")
    evidence.write_bytes(b"previous evidence")
    args = ["inspect", task.model_id, "--output-dir", str(inspection_layout.root)]
    if probe:
        args.append("--probe-interface")
    with patch("acprof.host.detect.detect_task", return_value=task), patch(
        "acprof.host.preflight.require_collection_host"), patch("acprof.host.preflight.require_native_docker"), patch(
        "acprof.host.run_state.MeasurementLock"), patch("acprof.host.preparation.interface_probe.probe_interface") as run:
        code = main(args)
    assert resolution.read_bytes() == b"previous resolution"
    assert evidence.read_bytes() == b"previous evidence"
    assert code == 2
    run.assert_not_called()


def test_busy_machine_preserves_inspection_resolution(tmp_path):
    from acprof.host.run_state import RunStateError

    task = fixture.TestModelContract().discover()
    resolution = tmp_path / "model_resolution.json"
    resolution.write_bytes(b"previous resolution")
    with patch("acprof.host.detect.detect_task", return_value=task), patch(
        "acprof.host.preflight.require_collection_host"), patch("acprof.host.preflight.require_native_docker"), patch(
        "acprof.host.run_state.MeasurementLock", side_effect=RunStateError("busy")), patch(
        "acprof.host.preparation.interface_probe.probe_interface") as run:
        assert main(["inspect", task.model_id, "--probe-interface", "--output-dir", str(tmp_path)]) == 1
    assert resolution.read_bytes() == b"previous resolution"
    run.assert_not_called()


@pytest.mark.parametrize(("field", "value"), [("source", "modelscope"), ("model_id", "other/repository"),
                                             ("revision", "b" * 40)])
def test_existing_static_resolution_must_belong_to_same_snapshot(tmp_path, field, value):
    from acprof.models.contract import write_model_resolution

    task = fixture.TestModelContract().discover()
    previous = fixture.TestModelContract().discover()
    previous.model_resolution["provenance"]["sources"]["repository_snapshot"][field] = value
    path = write_model_resolution(previous, tmp_path)
    expected = path.read_bytes()
    with patch("acprof.host.detect.detect_task", return_value=task):
        assert main(["inspect", task.model_id, "--output-dir", str(tmp_path)]) == 2
    assert path.read_bytes() == expected


def test_windows_static_export_holds_a_nonblocking_lock(tmp_path, monkeypatch):
    import os
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock

    task = fixture.TestModelContract().discover()
    lock = Mock()
    lock.side_effect = [None, None, OSError("file is locked")]
    windows = SimpleNamespace(LK_NBLCK=2, LK_UNLCK=0, locking=lock)
    monkeypatch.setitem(sys.modules, "msvcrt", windows)
    monkeypatch.setattr(sys, "platform", "win32")
    command = ["inspect", task.model_id, "--output-dir", str(tmp_path)]
    with patch("acprof.host.detect.detect_task", return_value=task):
        assert main(command) == 0
        path = tmp_path / "model_resolution.json"
        expected = path.read_bytes()
        assert main(command) == 1
        assert path.read_bytes() == expected
    assert [call.args[1:] for call in lock.call_args_list] == [
        (windows.LK_NBLCK, 1), (windows.LK_UNLCK, 1), (windows.LK_NBLCK, 1),
    ]
    for call in lock.call_args_list:
        fd = call.args[0]
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(fd)


@pytest.mark.parametrize("previous", ["unreadable", "foreign"])
def test_pinned_review_preserves_unidentified_or_foreign_corrupt_output(tmp_path, previous):
    from acprof.models.spec import task_model_spec

    task = fixture.TestModelContract().discover()
    declaration = tmp_path / "acprof_model.json"
    declaration.write_text(json.dumps(task_model_spec(task)))
    path = tmp_path / "model_resolution.json"
    if previous == "unreadable":
        path.write_bytes(b"partial old report")
    else:
        saved = task.model_resolution
        saved["provenance"]["sources"]["repository_snapshot"]["source"] = "modelscope"
        path.write_text(json.dumps({**saved, "corrupt_metric": float("nan")}))
    original = path.read_bytes()
    with patch("acprof.host.detect.detect_task", return_value=task):
        assert main(["inspect", task.model_id, "--expected-revision", task.model_revision,
                     "--model-spec", str(declaration), "--output-dir", str(tmp_path)]) == 2
    assert path.read_bytes() == original


@pytest.mark.parametrize("selection", [[], ["--tasks", "text-generation"]])
def test_inspect_gguf_only_repository_reports_artifacts_before_requesting_a_profile(tmp_path, selection):
    from types import SimpleNamespace

    # Qwen/Qwen2.5-0.5B-Instruct-GGUF at this revision declares two workload profiles.
    repo = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    revision = "9217f5db79a29953eb74d5343926648285ec7e67"
    artifacts = [f"qwen2.5-0.5b-instruct-{variant}.gguf" for variant in (
        "fp16", "q2_k", "q3_k_m", "q4_0", "q4_k_m", "q5_0", "q5_k_m", "q6_k", "q8_0")]
    files = [".gitattributes", "LICENSE", "README.md", *artifacts]
    hub = SimpleNamespace(sha=revision, pipeline_tag="text-generation", config={},
                          tags=["gguf", "text-generation", "conversational"],
                          siblings=[SimpleNamespace(rfilename=name) for name in files])
    with patch("acprof.host.model_store.store.cached_model_info", return_value=hub), patch(
        "acprof.host.detect._download_metadata", side_effect=AssertionError("unexpected metadata read")
    ):
        assert main(["inspect", repo, "--revision", revision, "--output-dir", str(tmp_path), *selection]) == 2
    decision = json.loads((tmp_path / "model_resolution.json").read_text())
    assert decision["status"] == "blocked"
    failure = decision["failure"]
    assert failure["reason_code"] == "artifact_layout_unsupported"
    assert failure["evidence"]["artifact_format"] == "GGUF"
    assert failure["evidence"]["artifact_paths"] == sorted(artifacts)[:3]
    assert failure["evidence"]["file_count"] == len(files)
    assert failure["evidence"]["declared_task"] == "text-generation"
    assert decision["runtime_validation"]["status"] == "not_run"
