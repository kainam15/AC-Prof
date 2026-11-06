"""Automatic runs reject static input errors before preparing an experiment."""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.cli.main import main
from acprof.experiments.configuration import RunConfig, RunConfigError


@pytest.fixture
def preparation(monkeypatch):
    boundaries = SimpleNamespace(
        download=Mock(), environment=Mock(),
        measurement_lock=Mock(return_value=nullcontext()),
        directory_lock=Mock(return_value=nullcontext()),
        access=Mock(side_effect=RuntimeError("repository access reached")),
        resolution=Mock(side_effect=AssertionError("model resolution reached")),
        collection=Mock(side_effect=AssertionError("collection reached")),
    )
    for path, boundary in (
        ("acprof.application.download.apply_download_arguments", boundaries.download),
        ("acprof.host.env_utils.bootstrap_project_env", boundaries.environment),
        ("acprof.host.run_state.MeasurementLock", boundaries.measurement_lock),
        ("acprof.host.run_state.ResultDirectoryLock", boundaries.directory_lock),
        ("acprof.host.automation.check_repository_access", boundaries.access),
        ("acprof.host.detect.detect_task", boundaries.resolution),
        ("acprof.cli.run.main", boundaries.collection),
    ):
        monkeypatch.setattr(path, boundary)
    return boundaries


@pytest.mark.parametrize(("options", "field"), [
    (["--repeat", "0"], "repeat"),
    (["--warmup", "-1"], "warmup"),
    (["--cpus", "0"], "cpus"),
    (["--mems", "invalid"], "mems"),
    (["--input-scales", "nan"], "input_scales"),
    (["--sample-hz", "nan"], "sample_hz"),
    (["--raw-sample-limit", "1"], "raw_sample_limit"),
    (["--ncu-repeat", "0"], "ncu_repeat"),
    (["--latency-slo", "invalid"], "latency_slo"),
    (["--runtime", "process", "--profiling-mode", "full"], "runtime"),
    (["--resume", "--rebuild-image"], "rebuild_image"),
], ids=["repeat", "warmup", "cpu-list", "memory-list", "scale-list", "sample-frequency",
        "raw-sample-limit", "profiler-count", "latency-slo", "process-full", "resume-rebuild"])
def test_invalid_auto_options_stop_before_preparation(tmp_path, preparation, capsys, options, field):
    output = tmp_path / "output"
    try:
        code = main(["auto", "fixture/model", "--profiling-mode", "auto", "--notify", "none",
                     "--output-dir", str(output), *options])
    except SystemExit as error:
        code = error.code

    assert code == 2
    captured = capsys.readouterr()
    assert f"[{field}]" in captured.err
    assert "repository access reached" not in captured.err
    assert "[auto] fixture/model" not in captured.out
    assert not output.exists()
    for boundary in vars(preparation).values():
        boundary.assert_not_called()


@pytest.mark.parametrize("flag", ["--model-spec", "--workload-spec"])
def test_missing_auto_input_files_fail_before_preparation(tmp_path, preparation, capsys, flag):
    output = tmp_path / "output"
    try:
        code = main(["auto", "fixture/model", "--profiling-mode", "basic", "--notify", "none",
                     "--output-dir", str(output), flag, str(tmp_path / "missing.json")])
    except SystemExit as error:
        code = error.code

    assert code == 2
    assert f"[{flag[2:].replace('-', '_')}]" in capsys.readouterr().err
    assert not output.exists()
    for boundary in vars(preparation).values():
        boundary.assert_not_called()


@pytest.mark.parametrize("runtime", ["auto", "process"])
def test_valid_auto_preserves_pending_mode_relative_paths_and_omitted_transport(
    tmp_path, preparation, monkeypatch, runtime,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workload.json").write_text("{}", encoding="utf-8")
    task = SimpleNamespace(model_id="fixture/model", model_revision="a" * 40)
    automatic = Mock(root=tmp_path / "output", preparation_artifacts={}, collection_path=None)
    automatic.prepare.return_value = task
    automatic.finish.return_value = 0
    factory = Mock(return_value=automatic)
    monkeypatch.setattr("acprof.host.automation.AutomaticRun", factory)
    preparation.collection.side_effect = None
    preparation.collection.return_value = SimpleNamespace(task_collection_path=None)

    assert main(["auto", "fixture/model", "--profiling-mode", "auto", "--runtime", runtime,
                 "--notify", "none", "--output-dir", str(tmp_path / "output"),
                 "--workload-spec", "workload.json"]) == 0

    options = factory.call_args.args[0]
    assert options.profiling_mode == "auto"
    assert options.runtime == runtime
    assert options.workload_spec == "workload.json"
    assert all(getattr(options, field) is None for field in
               ("model_source", "download_mode", "max_download", "model_store", "model_store_max"))
    preparation.download.assert_called_once_with(options)
    preparation.environment.assert_called_once_with(tmp_path)
    preparation.collection.assert_called_once_with(args=options, prepared_task=task, preparation_artifacts={})
    automatic.finish.assert_called_once_with(None)


@pytest.mark.parametrize("runtime", ["auto", "docker", "process"])
def test_pending_profiling_is_only_accepted_with_explicit_automatic_policy(runtime):
    config = replace(RunConfig.smoke("fixture/model"), runtime=runtime, profiling_mode="auto")
    with pytest.raises(RunConfigError) as failure:
        config.validate()
    assert "profiling_mode" in {issue.field for issue in failure.value.issues}
    validated = config.validate(allow_auto_profiling=True)
    assert validated.profiling_mode == "auto"
    assert validated.runtime == runtime


@pytest.mark.parametrize("options", [
    {"compute_profile_tool": "torch"}, {"execution_profile_tool": "nsys"},
    {"extra_options": {"dram_energy": "required"}}, {"profiling_mode": "full"},
], ids=["compute-profiler", "execution-profiler", "required-dram", "explicit-full"])
def test_pending_profiling_does_not_relax_process_capabilities(options):
    config = replace(RunConfig.smoke("fixture/model"), runtime="process", profiling_mode="auto")
    with pytest.raises(RunConfigError) as failure:
        replace(config, **options).validate(allow_auto_profiling=True)
    assert "runtime" in {issue.field for issue in failure.value.issues}
