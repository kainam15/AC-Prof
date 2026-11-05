"""Profile preparation keeps experiment directories after collection expansion."""
import json
from pathlib import Path

import pytest

from acprof.results.layout import ArtifactLayout
from acprof.tui.commands import prepare_profile


@pytest.mark.parametrize("source_kind", ["collection", "parent", "task"])
@pytest.mark.parametrize("dry_run", [True, False])
def test_profile_targets_single_task_directory_without_writing(tmp_path, source_kind, dry_run):
    collection = tmp_path / "model collection"
    run_id = "fixture-run"
    task_id = "text-classification"
    task_dir = collection / "tasks" / task_id / run_id
    ArtifactLayout.for_new_run(task_dir).initialize()
    task_runs = collection / "metadata" / "task_runs"
    task_runs.mkdir(parents=True)
    (task_runs / f"{run_id}.json").write_text(json.dumps({
        "schema_version": 1,
        "run_id": run_id,
        "tasks": [{"task_id": task_id}],
    }), encoding="utf-8")
    before = {path.relative_to(tmp_path): path.read_bytes() if path.is_file() else None
              for path in tmp_path.rglob("*")}
    source = {"collection": collection, "parent": tmp_path, "task": task_dir}[source_kind]

    pending = prepare_profile(
        str(source.relative_to(tmp_path)), tools="torch,ncu", dry_run=dry_run,
        project_dir=tmp_path, python_executable=Path("/python"),
    )

    expected = ("/python", "-u", "-m", "acprof", "profile", str(task_dir), "--tools", "torch,ncu")
    assert pending.command == expected + (("--dry-run",) if dry_run else ())
    assert pending.result_dir == str(task_dir)
    assert pending.kind == ("profile-dry-run" if dry_run else "profile")
    assert {path.relative_to(tmp_path): path.read_bytes() if path.is_file() else None
            for path in tmp_path.rglob("*")} == before
