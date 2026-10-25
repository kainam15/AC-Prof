"""Behavioral coverage for the shared configuration/CLI schema."""
from dataclasses import fields, replace
from pathlib import Path

import pytest

from acprof.experiment import RunConfig, build_run_command
from acprof.run_args import build_parser
from acprof.run_options import RunOptions, configuration_arguments
from acprof.tui.commands import build_probe_command


def test_run_command_round_trip_retains_values_and_recorded_extras(tmp_path: Path):
    config = replace(
        RunConfig.smoke("org/model"),
        cpus="1,2",
        mems="4,8",
        input_scales="3,9",
        sample_hz=5.5,
        sampling_policy="skip",
        raw_sample_limit=2048,
        repeat_window_seconds=0.75,
        request_timeout_seconds=19.5,
        output_dir="results with spaces",
        model_store="/tmp/model cache",
        prune_startup_oom=False,
        idle_debug=True,
        extra_options={
            "matrix_seed": 37,
            "latency_slo": ["default=0.5", "task:text-generation=1.5"],
            "keep_execution_profiles": False,
        },
    ).validate(project_dir=tmp_path)

    command = build_run_command(config, project_dir=tmp_path)
    parsed = build_parser().parse_args(command[command.index("run") + 1:])
    restored = RunConfig.from_namespace(parsed).validate(project_dir=tmp_path)

    for item in fields(RunOptions):
        assert getattr(restored, item.name) == getattr(config, item.name), item.name
    assert restored.extra_options["matrix_seed"] == 37
    assert restored.extra_options["latency_slo"] == ["default=0.5", "task:text-generation=1.5"]
    assert restored.extra_options["keep_execution_profiles"] is False
    assert command[command.index("--model-store") + 1] == "/tmp/model cache"


def test_default_cli_and_form_configuration_agree():
    parsed = build_parser().parse_args(["--model", "org/model"])
    config = RunConfig.from_namespace(parsed)
    expected = RunConfig(model="org/model")
    for item in fields(RunOptions):
        assert getattr(config, item.name) == getattr(expected, item.name), item.name
    # Optional CLI absence remains distinct until normalized into a form config.
    assert parsed.revision is None
    assert parsed.model_source is None
    assert parsed.input_scales is None


def test_probe_receives_shared_values_without_formal_measurement_controls(tmp_path: Path):
    from acprof.cli.probe import _build_parser

    config = replace(
        RunConfig.smoke("org/model"),
        revision="frozen-revision",
        batch_size=3,
        model_source="modelscope",
        model_store="/tmp/model cache",
        output_dir="probe output",
        sampling_policy="skip",
        idle_debug=True,
        rebuild_image=True,
        prune_startup_oom=False,
    )
    command = build_probe_command(config, project_dir=tmp_path)
    parsed = _build_parser().parse_args(command[command.index("probe") + 1:])
    assert parsed.batch_size == 3
    assert parsed.model_source == "modelscope"
    assert parsed.model_store == "/tmp/model cache"
    assert parsed.timeout_seconds is None
    assert command[command.index("--model") + 1] == "org/model"
    assert command[command.index("--revision") + 1] == "frozen-revision"
    assert command[command.index("--batch-size") + 1] == "3"
    assert command[command.index("--model-source") + 1] == "modelscope"
    assert command[command.index("--output-dir") + 1] == "probe output"
    assert "--rebuild-image" in command
    for excluded in ("--warmup", "--repeat", "--sampling-policy", "--idle-debug", "--no-prune-startup-oom"):
        assert excluded not in command


@pytest.mark.parametrize("prune_startup_oom", [True, False])
def test_boolean_controls_round_trip_in_both_directions(prune_startup_oom: bool):
    config = RunOptions(model="org/model", prune_startup_oom=prune_startup_oom, resume=True)
    parsed = build_parser().parse_args(configuration_arguments(config, purpose="run"))
    assert parsed.prune_startup_oom is prune_startup_oom
    assert parsed.resume is True


def test_automatic_cli_retains_positional_model_and_auto_mode():
    parsed = build_parser(default_notify_provider="none", automatic=True).parse_args(
        ["org/model", "--profiling-mode", "auto", "--idle-seconds", "0.25"]
    )
    assert parsed.model == "org/model"
    assert parsed.profiling_mode == "auto"
    assert parsed.notify == "none"
    assert parsed.idle_seconds == 0.25


@pytest.mark.parametrize("purpose", ["run", "probe"])
def test_command_preserves_underlying_string_values(purpose: str, tmp_path: Path):
    from acprof.cli.probe import _build_parser

    class DisplayValue(str):
        def __str__(self) -> str:
            return "displayed_" + super().__str__()

    config = replace(
        RunConfig.smoke("org/model"),
        model_source=DisplayValue("huggingface"),
        revision=DisplayValue("frozen-revision"),
    )
    builder = build_run_command if purpose == "run" else build_probe_command
    parser = build_parser() if purpose == "run" else _build_parser()

    command = builder(config, project_dir=tmp_path)
    parsed = parser.parse_args(command[command.index(purpose) + 1:])

    assert parsed.model_source == "huggingface"
    assert parsed.revision == "frozen-revision"
    assert type(command[command.index("--model-source") + 1]) is str
