"""Only explicit, completed developer runs may be cleaned; legacy data is immutable."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts import internal_testing as layout, run_local_tests as local

START = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def root_at(tmp_path: Path) -> Path:
    return tmp_path / "internal-testing"


def make_finished(tmp_path: Path, *, status: str = "success", keep: bool = False) -> Path:
    root = root_at(tmp_path)
    path = layout.create_run("tui-layout", root=root, now=START, keep=keep)
    if status != "running":
        layout.finish_run(path, root=root, status=status, now=START + timedelta(hours=1))
    return path


def test_run_layout_is_unique_utc_and_has_a_traceable_manifest(tmp_path):
    root = root_at(tmp_path)
    first = layout.create_run("tui-layout", root=root, now=START, command=["pytest", "-q"])
    second = layout.create_run("tui-layout", root=root, now=START)
    assert first != second
    assert first.relative_to(root).parts[:2] == ("runs", "2026-10")
    assert first.name.startswith("20261001T120000Z--tui-layout--")
    data = layout.read_run(first, root=root)
    assert data["run_id"] == first.name
    assert data["status"] == "running"
    assert data["command"] == ["pytest", "-q"]
    assert data["git_commit"] is None or len(data["git_commit"]) == 40
    assert data["git_dirty"] is None or isinstance(data["git_dirty"], bool)
    assert all((root / category).is_dir() for category in layout.CATEGORIES)


@pytest.mark.parametrize("task", ["Upper", "has_underscore", "../escape", "a--b", "", "x" * 49])
def test_invalid_task_is_rejected_before_creating_anything(tmp_path, task):
    root = root_at(tmp_path)
    with pytest.raises(ValueError, match="Task must be"):
        layout.create_run(task, root=root, now=START)
    assert not root.exists()


def test_only_finished_success_and_resolved_runs_can_expire(tmp_path):
    root = root_at(tmp_path)
    ready = make_finished(tmp_path)
    active = layout.create_run("active-task", root=root, now=START)
    failed = make_finished(tmp_path, status="failed")
    interrupted = make_finished(tmp_path, status="interrupted")
    retained = make_finished(tmp_path, keep=True)
    resolved = make_finished(tmp_path, status="failed")
    layout.finish_run(resolved, status="resolved", root=root, now=START + timedelta(days=2))
    now = START + timedelta(days=20)
    assert layout.prune_candidates(root, now=now) == [ready]
    assert set((active, failed, interrupted, retained, resolved)).isdisjoint(
        layout.prune_candidates(root, now=now)
    )
    assert resolved in layout.prune_candidates(root, now=START + timedelta(days=33))


def test_prune_defaults_to_preview_and_requires_explicit_confirmation(tmp_path):
    root = root_at(tmp_path)
    ready = make_finished(tmp_path)
    legacy = root / "ci-20261001"
    legacy.mkdir()
    (legacy / "result.csv").write_text("important")
    evidence = root / "evidence" / "verified"
    evidence.mkdir()
    (evidence / "trace.log").write_text("important")
    now = START + timedelta(days=20)
    assert layout.prune(root, now=now) == [ready]
    assert ready.exists()
    with pytest.raises(ValueError, match="requires --confirm"):
        layout.prune(root, apply=True, now=now)
    assert ready.exists()
    assert layout.prune(root, apply=True, confirm="DELETE-ELIGIBLE-RUNS", now=now) == [ready]
    assert not ready.exists()
    assert (legacy / "result.csv").read_text() == "important"
    assert (evidence / "trace.log").read_text() == "important"


@pytest.mark.parametrize("protected", ["raw", ".acprof", "case.csv", "trace.pcap", "events.jsonl"])
def test_even_expired_runs_with_scientific_evidence_are_preserved(tmp_path, protected):
    path = make_finished(tmp_path)
    artifact = path / "artifacts" / protected
    artifact.parent.mkdir(exist_ok=True)
    if "." in protected and protected != ".acprof":
        artifact.write_text("original")
    else:
        artifact.mkdir()
    assert layout.prune_candidates(root_at(tmp_path), now=START + timedelta(days=20)) == []


def test_keep_marker_prevents_prune(tmp_path):
    path = make_finished(tmp_path)
    (path / ".keep").touch()
    assert layout.prune_candidates(root_at(tmp_path), now=START + timedelta(days=20)) == []


def test_unreadable_walk_does_not_authorize_deletion(tmp_path, monkeypatch):
    path = make_finished(tmp_path)

    def unreadable(*args, **kwargs):
        raise PermissionError("cannot inspect run")

    monkeypatch.setattr(layout.os, "walk", unreadable)
    assert layout.prune_candidates(root_at(tmp_path), now=START + timedelta(days=20)) == []
    assert path.exists()


def test_apply_rechecks_changes_and_reports_only_removed_runs(tmp_path, monkeypatch):
    path = make_finished(tmp_path)
    original = layout._eligible
    calls = 0

    def becomes_protected(run, *, root, now):
        nonlocal calls
        calls += 1
        if calls == 2:
            (run / ".keep").touch()
        return original(run, root=root, now=now)

    monkeypatch.setattr(layout, "_eligible", becomes_protected)
    removed = layout.prune(
        root_at(tmp_path),
        apply=True,
        confirm="DELETE-ELIGIBLE-RUNS",
        now=START + timedelta(days=20),
    )
    assert removed == []
    assert path.exists()


def test_corrupted_identity_never_becomes_a_deletion_candidate(tmp_path):
    root = root_at(tmp_path)
    path = make_finished(tmp_path)
    document = json.loads((path / layout.MANIFEST).read_text())
    document["run_id"] = "20261001T120000Z--wrong--deadbeef"
    (path / layout.MANIFEST).write_text(json.dumps(document))
    assert layout.prune_candidates(root, now=START + timedelta(days=20)) == []
    assert layout.inventory(root)["unrecognized_runs"] == 1
    with pytest.raises(ValueError, match="identity"):
        layout.finish_run(path, status="resolved", root=root)


def test_symlinks_and_incorrect_layout_are_not_managed(tmp_path):
    root = root_at(tmp_path)
    finished = make_finished(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep-me").write_text("value")
    symlink = finished.parent / "20261001T120000Z--tui-layout--abcdef12"
    symlink.symlink_to(outside, target_is_directory=True)
    assert layout.prune_candidates(root, now=START + timedelta(days=20)) == [finished]
    with pytest.raises(ValueError, match="below the configured"):
        layout.read_run(outside, root=root)
    layout.prune(root, apply=True, confirm="DELETE-ELIGIBLE-RUNS", now=START + timedelta(days=20))
    assert (outside / "keep-me").read_text() == "value"


def test_lifecycle_transitions_are_explicit(tmp_path):
    path = make_finished(tmp_path)
    with pytest.raises(ValueError, match="Invalid lifecycle"):
        layout.finish_run(path, status="failed", root=root_at(tmp_path))
    invalid = make_finished(tmp_path, status="failed")
    with pytest.raises(ValueError, match="Invalid lifecycle"):
        layout.finish_run(invalid, status="success", root=root_at(tmp_path))
    layout.finish_run(invalid, status="resolved", root=root_at(tmp_path))
    assert layout.read_run(invalid, root=root_at(tmp_path))["status"] == "resolved"


def test_inventory_lists_legacy_without_opening_or_moving_it(tmp_path):
    root = root_at(tmp_path)
    make_finished(tmp_path)
    (root / "historic-ci-12").mkdir()
    result = layout.inventory(root, limit=3)
    assert result["legacy_count"] == 1
    assert result["legacy_sample"] == ["historic-ci-12"]
    assert result["managed"] == {"success": 1}


def test_inventory_check_rejects_stray_root_and_accepts_canonical_layout(tmp_path, monkeypatch):
    root = root_at(tmp_path)
    layout.initialize(root)
    monkeypatch.setattr(layout, "DATA_ROOT", root)
    assert layout.main(["inventory", "--check"]) == 0
    (root / "old-ci").mkdir()
    assert layout.main(["inventory", "--check"]) == 1
    (root / "old-ci").rename(root / "evidence" / "old-ci")
    assert layout.main(["inventory", "--check"]) == 0


def test_inventory_check_rejects_symlinked_managed_category(tmp_path, monkeypatch):
    root = root_at(tmp_path)
    layout.initialize(root)
    monkeypatch.setattr(layout, "DATA_ROOT", root)
    (root / "shared").rmdir()
    (root / "shared").symlink_to(tmp_path, target_is_directory=True)
    assert layout.main(["inventory", "--check"]) == 1


def test_explicit_local_output_rejects_legacy_root(tmp_path):
    with pytest.raises(SystemExit) as error:
        local.main(["--output", "internal-testing/unmanaged-run"])
    assert error.value.code == 2


def test_local_test_runner_uses_managed_run_and_finishes_success(tmp_path, monkeypatch):
    root = root_at(tmp_path)
    monkeypatch.setattr(local, "EVIDENCE_ROOT", root)
    captured = []

    def fake_run_local(**kwargs):
        captured.append(kwargs)
        assert kwargs["precreated"] is True
        assert (kwargs["output"] / layout.MANIFEST).is_file()
        return True

    monkeypatch.setattr(local, "run_local", fake_run_local)
    assert local.main(["--scope", "fast"]) == 0
    path = captured[0]["output"]
    assert path.is_relative_to(root / "runs")
    assert layout.read_run(path, root=root)["status"] == "success"


def test_local_test_runner_records_failure_without_deleting_evidence(tmp_path, monkeypatch):
    root = root_at(tmp_path)
    monkeypatch.setattr(local, "EVIDENCE_ROOT", root)

    def interrupted(**kwargs):
        (kwargs["output"] / "failed.log").write_text("diagnostic evidence")
        raise KeyboardInterrupt

    monkeypatch.setattr(local, "run_local", interrupted)
    assert local.main(["--scope", "fast"]) == 130
    paths = list((root / "runs").glob("*/*"))
    assert len(paths) == 1
    assert layout.read_run(paths[0], root=root)["status"] == "interrupted"
    assert (paths[0] / "failed.log").read_text() == "diagnostic evidence"
