"""Public analysis outputs cannot become authoritative inputs on the next read."""
import pytest

from acprof.cli.main import main
from tests.support.independent_comparison_fixtures import IndependentComparisonFixture
from tests.support.load_sweep_fixtures import make_sweep


@pytest.fixture
def experiments(tmp_path):
    fixture = IndependentComparisonFixture()
    fixture.build(None, tmp_path)
    left = fixture.replicate("left", 0, [1, 2, 3])
    right = fixture.replicate("right", 0, [2, 3, 4])
    (left / "result_layers.json").unlink()
    return left, right


def reject(args, destination, capsys):
    with pytest.raises(SystemExit) as error:
        main([str(arg) for arg in args])
    assert error.value.code != 0
    assert destination.name in capsys.readouterr().err
    assert not destination.exists()


@pytest.mark.parametrize("option", ["--output", "--output-dir"])
def test_stats_preserves_absent_manifest(experiments, option, capsys):
    left, _ = experiments
    destination = left / "result_manifest.json"
    reject(["stats", left / "result_all.csv", option, destination], destination, capsys)


@pytest.mark.parametrize("reserved_option", ["--output", "--csv-output", "--html-output"])
def test_compare_checks_every_destination_before_writing(experiments, tmp_path, reserved_option, capsys):
    left, right = experiments
    alias = tmp_path / "other-experiment"
    alias.symlink_to(right, target_is_directory=True)
    destination = right / ("summary.csv" if reserved_option == "--csv-output" else "result_manifest.json")
    exports = {"--output": tmp_path / "comparison.json", "--csv-output": tmp_path / "comparison.csv",
               "--html-output": tmp_path / "comparison.html"}
    exports[reserved_option] = destination
    args = ["compare", "--left", left, "--right", alias, "--metric", "latency_app_s"]
    for option, path in exports.items():
        args.extend([option, path])
    reject(args, destination, capsys)
    assert not any(path.exists() for path in exports.values())


def test_report_protects_group_outside_positional_inputs(experiments, capsys):
    left, right = experiments
    destination = right / "result_manifest.json"
    reject(["report", left, "--left", left, "--right", right, "--output", destination], destination, capsys)


@pytest.mark.parametrize("kind", ["point", "sweep"])
def test_stream_report_preserves_reserved_paths(tmp_path, kind, capsys):
    sweep = make_sweep(tmp_path / "sweep")
    source = sweep / "sweep.json" if kind == "sweep" else sweep / "points" / "0001" / "load.json"
    destination = source.parent / "result_manifest.json"
    reject(["report", source, "--output", destination], destination, capsys)


@pytest.mark.parametrize("target_root", ["task", "collection"])
def test_stats_protects_an_expanded_task_and_its_collection(experiments, tmp_path, target_root, capsys):
    import json

    left, _ = experiments
    parent = tmp_path / "outputs"
    collection = parent / "org--model"
    task = collection / "tasks" / "generation" / "run"
    task.parent.mkdir(parents=True)
    left.rename(task)
    records = collection / "metadata" / "task_runs"
    records.mkdir(parents=True)
    (records / "run.json").write_text(json.dumps({
        "schema_version": 1, "run_id": "run", "tasks": [{"task_id": "generation"}],
    }))
    destination = (task if target_root == "task" else collection) / "result_manifest.json"
    reject(["stats", parent, "--output", destination], destination, capsys)
