"""Saved CLI options must survive the same normalization used by a real run."""
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from acprof.experiment import RunConfig
from acprof.host.run_state import run_options
from acprof.run_args import build_parser
from acprof.tui.experiment_catalog import resume_command, scan_experiments


def normalized_args(arguments, root):
    args = build_parser().parse_args(arguments)
    config = RunConfig.from_namespace(args).validate(project_dir=root)
    for name, value in asdict(config).items():
        setattr(args, name, value)
    return args


@pytest.mark.parametrize("historical_internal_field", [False, True])
def test_real_cli_options_round_trip_without_internal_fields(tmp_path, historical_internal_field):
    args = normalized_args([
        "--model", "demo/model", "--gpus", "off", "--matrix-seed", "71",
        "--latency-slo", "default=3", "--discard-compute-profiles",
    ], tmp_path)
    options = run_options(args)
    if not historical_internal_field:
        assert "extra_options" not in options
    else:
        options["extra_options"] = {}
    directory = tmp_path / "demo--model"
    directory.mkdir()
    state_path = directory / "run_state.json"
    state_path.write_text(json.dumps({
        "schema_version": 1, "run_id": "failed-preparation", "status": "failed",
        "options": options, "cases": {}, "artifacts": {},
    }))
    original = state_path.read_bytes()
    record = scan_experiments([directory]).records[0]
    command = resume_command(record, python_executable=Path("/fixed/python"))
    restored = normalized_args(command[command.index("--model"):], tmp_path)
    assert restored.resume
    assert restored.matrix_seed == 71
    assert restored.latency_slo == ["default=3"]
    assert not restored.keep_compute_profiles
    assert restored.output_dir == str(tmp_path)
    assert run_options(restored) == {key: value for key, value in options.items() if key != "extra_options"}
    assert state_path.read_bytes() == original


def test_frozen_nested_options_cannot_silently_override_measurement_parameters(tmp_path):
    from acprof.run_args import flatten_run_options

    with pytest.raises(ValueError, match="matrix_seed"):
        flatten_run_options({"matrix_seed": 71, "extra_options": {"matrix_seed": 99}})
    with pytest.raises(ValueError, match="extra_options"):
        flatten_run_options({"extra_options": []})
