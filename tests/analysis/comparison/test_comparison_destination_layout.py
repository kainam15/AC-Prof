"""Validate all comparison file names before creating output directories."""
from contextlib import nullcontext

import pytest

from acprof.results import outputs


def test_compare_refuses_file_as_another_outputs_parent_before_analysis(tmp_path, monkeypatch, capsys):
    from acprof.cli import compare

    calls = []

    def compare_runs(*args, **kwargs):
        calls.append(args)
        return {"status": "compatible", "groups": [{"difference": 1}]}

    monkeypatch.setattr("acprof.host.run_state.offline_analysis_lock", nullcontext)
    monkeypatch.setattr("acprof.analysis.independent_comparison.compare_experiments", compare_runs)
    target = tmp_path / "comparison"
    with pytest.raises(SystemExit) as error:
        compare.main(["--left", "left", "--right", "right", "--output", str(target),
                      "--csv-output", str(target / "table.csv")])
    assert error.value.code == 2
    assert str(target) in capsys.readouterr().err
    assert not calls
    assert not target.exists()


@pytest.mark.parametrize("relation", ["same", "child", "backtrack"])
@pytest.mark.parametrize("reverse", [False, True])
def test_generated_and_requested_destinations_must_be_distinct_files(tmp_path, relation, reverse):
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    sidecar = tmp_path / "report.windows-1.json"
    other = alias / sidecar.name
    if relation == "child":
        other /= "table.csv"
    elif relation == "backtrack":
        other = other / ".." / "table.csv"
    paths = [sidecar, other]
    with pytest.raises(ValueError, match="output"):
        outputs.validate_comparison_destinations(reversed(paths) if reverse else paths)
    assert set(tmp_path.iterdir()) == {alias}


@pytest.mark.parametrize("kind", ["file", "dangling", "loop"])
def test_existing_invalid_parent_after_missing_path_detour(tmp_path, kind):
    blocked = tmp_path / "blocked"
    if kind == "file":
        blocked.write_text("old data")
    elif kind == "dangling":
        blocked.symlink_to(tmp_path / "absent")
    else:
        blocked.symlink_to(blocked)
    missing = tmp_path / "missing"
    target = missing / ".." / blocked.name / "report.html"
    with pytest.raises(ValueError, match="blocked"):
        outputs.validate_comparison_destinations([target])
    assert not missing.exists()
    assert set(tmp_path.iterdir()) == {blocked}


def test_shared_new_parents_and_directory_links_are_allowed_without_creating_outputs(tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    shared = tmp_path / "new"
    outputs.validate_comparison_destinations([
        shared / "tables" / "comparison.csv", alias / "new" / "comparison", shared / "tables" / ".." / "view",
    ])
    assert not shared.exists()
