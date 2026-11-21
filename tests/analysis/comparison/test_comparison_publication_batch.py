"""Multi-file comparison publication keeps source inodes private until rollback ends."""

import errno
import os
from pathlib import Path

import pytest

from acprof.analysis.publication import ComparisonPublication


def test_publishes_sidecars_then_registered_outputs_across_directories(tmp_path, monkeypatch):
    csv = tmp_path / "tables" / "group.csv"
    html = tmp_path / "pages" / "index.html"
    sidecar = html.with_suffix(".windows-1.json")
    links = []
    link = os.link

    def record_link(source, destination):
        assert Path(source).stat().st_dev == destination.parent.stat().st_dev
        assert Path(source).name == destination.name
        links.append(destination)
        link(source, destination)

    with ComparisonPublication() as batch:
        staged_csv = batch.stage(csv)
        staged_html = batch.stage(html)
        staged_csv.write_text("table")
        staged_html.write_text("html")
        staged_html.with_name(sidecar.name).write_text("windows")
        assert not csv.parent.exists()
        assert not html.parent.exists()
        monkeypatch.setattr(os, "link", record_link)
        batch.publish()
    assert links == [sidecar, csv, html]
    assert [path.read_text() for path in links] == ["windows", "table", "html"]
    assert set(tmp_path.iterdir()) == {csv.parent, html.parent}


@pytest.mark.parametrize("nested", [False, True])
def test_discovered_sidecar_collision_creates_no_final_entry(tmp_path, nested):
    html = tmp_path / "new" / "report.html"
    sidecar = html.with_suffix(".windows-1.json")
    other = sidecar / "item.csv" if nested else sidecar
    with ComparisonPublication() as batch:
        batch.stage(other).write_text("other")
        staged = batch.stage(html)
        staged.write_text("html")
        staged.with_name(sidecar.name).write_text("windows")
        with pytest.raises(ValueError, match="output"):
            batch.publish()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("complete", [False, True])
def test_directory_sync_failure_preserves_only_a_complete_batch(tmp_path, monkeypatch, cancel, complete):
    first, last = (tmp_path / name for name in ("data.json", "report.html"))
    target = last if complete else first
    problem = KeyboardInterrupt("sync cancelled") if cancel else OSError("sync failed")

    def fail(directory):
        if directory == tmp_path and target.exists():
            raise problem

    with monkeypatch.context() as scoped, ComparisonPublication() as batch:
        batch.stage(first).write_text("first")
        batch.stage(last).write_text("last")
        scoped.setattr("acprof.analysis.publication.sync_directory", fail)
        with pytest.raises(type(problem)) as caught:
            batch.publish()
    assert caught.value is problem
    if complete:
        assert {path.name: path.read_text() for path in tmp_path.iterdir()} == {
            "data.json": "first", "report.html": "last",
        }
    else:
        assert not list(tmp_path.iterdir())
        with ComparisonPublication() as retry:
            retry.stage(first).write_text("retried")
            retry.publish()
        assert first.read_text() == "retried"


@pytest.mark.parametrize("timing", ["unsupported", "after-link"])
def test_link_failures_never_fall_back_to_overwrite(tmp_path, monkeypatch, timing):
    csv, html = (tmp_path / name for name in ("report.csv", "report.html"))
    problem = OSError(errno.EOPNOTSUPP, "hard links unsupported") if timing == "unsupported" else KeyboardInterrupt()
    link = os.link

    def fail(source, destination):
        if destination == csv:
            if timing == "after-link":
                link(source, destination)
            raise problem
        link(source, destination)

    with ComparisonPublication() as batch:
        batch.stage(csv).write_text("csv")
        batch.stage(html).write_text("html")
        monkeypatch.setattr(os, "link", fail)
        with pytest.raises(type(problem)) as caught:
            batch.publish()
    assert caught.value is problem
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("replacement", ["file", "directory", "symlink", "dangling", "gone"])
def test_rollback_preserves_a_replaced_output_and_late_claimant(tmp_path, monkeypatch, replacement):
    first, last = (tmp_path / name for name in ("data.json", "report.html"))
    link = os.link

    def compete(source, destination):
        if destination == last:
            first.unlink()
            if replacement == "file":
                first.write_text("unrelated")
            elif replacement == "directory":
                first.mkdir()
            elif replacement in {"symlink", "dangling"}:
                first.symlink_to(last if replacement == "symlink" else "absent")
            last.write_text("other writer")
        link(source, destination)

    with ComparisonPublication() as batch:
        batch.stage(first).write_text("data")
        batch.stage(last).write_text("html")
        monkeypatch.setattr(os, "link", compete)
        with pytest.raises(FileExistsError):
            batch.publish()
    assert last.read_text() == "other writer"
    assert set(tmp_path.iterdir()) == ({last} if replacement == "gone" else {first, last})
    if replacement == "file":
        assert first.read_text() == "unrelated"
    elif replacement == "directory":
        assert first.is_dir()
    elif replacement in {"symlink", "dangling"}:
        assert first.is_symlink()


def test_rollback_attempts_remaining_files_and_keeps_primary_failure(tmp_path, monkeypatch):
    first, second, last = (tmp_path / name for name in ("a", "b", "c"))
    problem = OSError("link denied")
    link = os.link
    unlink = os.unlink

    def fail_link(source, destination):
        if destination == last:
            raise problem
        link(source, destination)

    def fail_unlink(path, **options):
        if path == second:
            raise PermissionError("rollback denied")
        unlink(path, **options)

    with ComparisonPublication() as batch:
        for path in (first, second, last):
            batch.stage(path).write_text(path.name)
        monkeypatch.setattr(os, "link", fail_link)
        monkeypatch.setattr(os, "unlink", fail_unlink)
        with pytest.raises(OSError, match="link denied") as caught:
            batch.publish()
    assert caught.value is problem
    assert any("rollback denied" in note and str(second) in note for note in problem.__notes__)
    assert set(tmp_path.iterdir()) == {second}
