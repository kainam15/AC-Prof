"""Comparison JSON output is exclusive and publication errors are CLI failures."""

import errno
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from independent_comparison_fixtures import IndependentComparisonFixture

import acprof.cli.compare as compare_cli
from acprof.cli.main import main


def create_competitor(output: Path, kind: str) -> None:
    if kind == "file":
        output.write_bytes(b"other writer's JSON\n")
        output.chmod(0o600)
    elif kind == "directory":
        output.mkdir()
        (output / "keep.txt").write_bytes(b"existing contents\n")
    else:
        target = output.parent / "target.json"
        if kind == "symlink":
            target.write_bytes(b"linked contents\n")
        output.symlink_to(target.name)


def arguments(output):
    return ["--left", "left", "--right", "right", "--output", str(output)]


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "dangling_symlink"])
@pytest.mark.parametrize("timing", ["before", "during"])
def test_comparison_preserves_competing_directory_entries(tmp_path, monkeypatch, capsys, kind, timing):
    output = tmp_path / "comparison.json"
    entries = []

    def create():
        create_competitor(output, kind)
        entries.append(output.lstat())

    def compare(*_args, **_kwargs):
        assert timing == "during", "Existing outputs must be rejected before analysis"
        create()
        return {"status": "compatible", "groups": []}

    monkeypatch.setattr("acprof.analysis.independent_comparison.compare_experiments", compare)
    if timing == "before":
        create()
    with pytest.raises(SystemExit) as failure:
        compare_cli.main(arguments(output))
    assert failure.value.code == (2 if timing == "before" else 1)
    captured = capsys.readouterr()
    assert not captured.out
    assert ("output must be a new file" if timing == "before" else "比较失败") in captured.err
    assert "Traceback" not in captured.err
    assert len(entries) == 1
    assert output.lstat().st_ino == entries[0].st_ino
    assert output.lstat().st_mode == entries[0].st_mode
    if kind == "file":
        assert output.read_bytes() == b"other writer's JSON\n"
    elif kind == "directory":
        assert (output / "keep.txt").read_bytes() == b"existing contents\n"
    else:
        assert output.is_symlink()
        assert output.readlink() == Path("target.json")
        if kind == "symlink":
            assert output.read_bytes() == b"linked contents\n"
        else:
            assert not output.exists()
    assert not list(tmp_path.glob(".comparison.json.*.tmp"))


def test_competing_comparison_writers_publish_only_one_complete_json(tmp_path, monkeypatch):
    output = tmp_path / "comparison.json"
    rendezvous = Barrier(2)
    original_dump = json.dump

    def write_then_wait(payload, stream, **kwargs):
        original_dump(payload, stream, **kwargs)
        rendezvous.wait(timeout=10)

    def compare(left, _right, **_kwargs):
        return {"status": "compatible", "writer": str(left[0])}

    monkeypatch.setattr("acprof.analysis.independent_comparison.compare_experiments", compare)
    monkeypatch.setattr(json, "dump", write_then_wait)

    from acprof.analysis.independent_comparison import compare_experiments
    from acprof.artifacts import atomic_write_json

    def attempt(index):
        # Storage still excludes external/API writers that do not use the CLI lock.
        payload = compare_experiments([f"writer-{index}"], ["right"])
        try:
            atomic_write_json(output, payload, overwrite=False)
        except FileExistsError:
            return 1
        return 0

    with ThreadPoolExecutor(max_workers=2) as pool:
        codes = list(pool.map(attempt, range(2)))
    assert codes.count(0) == codes.count(1) == 1
    assert json.loads(output.read_text()) == {"status": "compatible", "writer": f"writer-{codes.index(0)}"}
    assert not list(tmp_path.glob(".comparison.json.*.tmp"))


@pytest.mark.parametrize("phase", ["serialize", "sync", "publish"])
def test_comparison_write_failures_are_reported_without_partial_output(tmp_path, monkeypatch, capsys, phase):
    output = tmp_path / "comparison.json"
    payload = {"status": "compatible", "value": float("nan") if phase == "serialize" else 1}
    monkeypatch.setattr("acprof.analysis.independent_comparison.compare_experiments", lambda *_args, **_kwargs: payload)

    def fail(*_args, **_kwargs):
        raise OSError(errno.ENOSPC if phase == "sync" else errno.EOPNOTSUPP, f"{phase} failed")

    if phase in {"sync", "publish"}:
        monkeypatch.setattr(os, "fsync" if phase == "sync" else "link", fail)
    with pytest.raises(SystemExit) as error:
        compare_cli.main(arguments(output))
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "比较失败" in captured.err
    assert "Traceback" not in captured.err
    assert not captured.out
    assert not output.exists()
    assert not list(tmp_path.glob(".comparison.json.*.tmp"))


@pytest.mark.parametrize("status", ["compatible", "incompatible", "unknown"])
@pytest.mark.parametrize("to_file", [False, True])
def test_comparison_status_and_json_format_remain_unchanged(tmp_path, monkeypatch, capsys, status, to_file):
    payload = {"status": status, "label": "中文", "value": None}
    monkeypatch.setattr("acprof.analysis.independent_comparison.compare_experiments", lambda *_args, **_kwargs: payload)
    output = tmp_path / "comparison.json"
    args = arguments(output) if to_file else ["--left", "left", "--right", "right"]
    assert compare_cli.main(args) == (0 if status == "compatible" else 1)
    captured = capsys.readouterr()
    assert not captured.err
    if to_file:
        assert not captured.out
        content = output.read_text(encoding="utf-8")
    else:
        assert not output.exists()
        content = captured.out
    assert content == '{\n  "status": "' + status + '",\n  "label": "中文",\n  "value": null\n}\n'


def test_compare_cli_publishes_real_comparison_without_changing_sources(tmp_path, request, capsys):
    fixture = IndependentComparisonFixture()
    fixture.build(request, tmp_path)
    left = fixture.replicate("left", 0, [2, 2])
    right = fixture.replicate("right", 0, [3, 3])
    originals = {path: path.read_bytes() for directory in (left, right)
                 for path in directory.rglob("*") if path.is_file()}
    output = tmp_path / "reports" / "comparison.json"
    assert main(["compare", "--left", str(left), "--right", str(right), "--metric", "latency_app_s",
                 "--resamples", "20", "--seed", "3", "--output", str(output)]) == 0
    payload = json.loads(output.read_text())
    assert payload["status"] == "compatible"
    assert payload["groups"][0]["difference"] == 1
    assert payload["groups"][0]["ratio"] == 1.5
    assert payload["groups"][0]["difference_ci"] is None
    assert not capsys.readouterr().err
    assert {path: path.read_bytes() for path in originals} == originals
