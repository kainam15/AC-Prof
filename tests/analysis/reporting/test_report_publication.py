"""Report publication must never replace an entry created during rendering."""

import errno
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier

import pytest

from acprof.analysis.model import AnalysisModel, load_analysis
from acprof.cli.main import main
from acprof.plotting.report import write_report


@pytest.fixture
def report_input(tmp_path) -> tuple[Path, AnalysisModel]:
    source = tmp_path / "result_all.csv"
    source.write_text(
        "cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status,latency_app_p95_s\n"
        "2,4,off,32,0,0,ok,0.04\n", encoding="utf-8",
    )
    return source, load_analysis([source])


def create_competitor(output: Path, kind: str) -> None:
    if kind == "file":
        output.write_bytes(b"other writer's report\n")
        output.chmod(0o600)
    elif kind == "directory":
        output.mkdir()
        (output / "keep.txt").write_bytes(b"existing contents\n")
    else:
        target = output.parent / "target.html"
        if kind == "symlink":
            target.write_bytes(b"linked contents\n")
        output.symlink_to(target.name)


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "dangling_symlink"])
@pytest.mark.parametrize("timing", ["before", "during"])
def test_report_preserves_competing_directory_entry(
    tmp_path, monkeypatch, report_input, kind, timing,
):
    source, model = report_input
    original_source = source.read_bytes()
    output = tmp_path / "report.html"
    entries = []

    def create():
        create_competitor(output, kind)
        entries.append(output.lstat())

    def render_bundle():
        assert timing == "during", "Existing entries must be rejected before rendering"
        create()
        return "/* test Plotly bundle */"

    monkeypatch.setattr("plotly.offline.get_plotlyjs", render_bundle)
    if timing == "before":
        create()
    error = ValueError if timing == "before" else FileExistsError
    with pytest.raises(error):
        write_report(model, output)

    assert len(entries) == 1
    assert output.lstat().st_ino == entries[0].st_ino
    assert output.lstat().st_mode == entries[0].st_mode
    if kind == "file":
        assert output.read_bytes() == b"other writer's report\n"
    elif kind == "directory":
        assert (output / "keep.txt").read_bytes() == b"existing contents\n"
    else:
        assert output.is_symlink()
        assert output.readlink() == Path("target.html")
        if kind == "symlink":
            assert output.read_bytes() == b"linked contents\n"
        else:
            assert not output.exists()
    assert not list(tmp_path.glob(".report.html.*.tmp"))
    assert source.read_bytes() == original_source


def test_two_report_writers_publish_exactly_one_complete_report(tmp_path, monkeypatch, report_input):
    source, model = report_input
    original_source = source.read_bytes()
    output = tmp_path / "report.html"
    rendezvous = Barrier(2)

    def render_bundle():
        rendezvous.wait(timeout=10)
        return "/* test Plotly bundle */"

    monkeypatch.setattr("plotly.offline.get_plotlyjs", render_bundle)

    def attempt(index):
        candidate = replace(model, configs=[{**model.configs[0], "model": f"writer-{index}"}])
        try:
            write_report(candidate, output)
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        published = list(pool.map(attempt, range(2)))
    assert published.count(True) == 1
    html = output.read_text(encoding="utf-8")
    match = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', html, re.S)
    assert match is not None
    payload = json.loads(match.group(1))
    assert payload["configs"][0]["model"] == f"writer-{published.index(True)}"
    assert html.rstrip().endswith("</html>")
    assert not list(tmp_path.glob(".report.html.*.tmp"))
    assert source.read_bytes() == original_source


def test_cli_publication_race_has_no_success_receipt(tmp_path, monkeypatch, report_input, capsys):
    source, _model = report_input
    original_source = source.read_bytes()
    output = tmp_path / "report.html"

    def render_bundle():
        create_competitor(output, "file")
        return "/* test Plotly bundle */"

    monkeypatch.setattr("plotly.offline.get_plotlyjs", render_bundle)
    with pytest.raises(SystemExit) as error:
        main(["report", str(source), "--output", str(output)])
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "报告生成失败" in captured.err
    assert str(output) in captured.err
    assert "Traceback" not in captured.err
    assert "ACPROF_REPORT" not in captured.out
    assert "报告已生成" not in captured.out
    assert output.read_bytes() == b"other writer's report\n"
    assert not list(tmp_path.glob(".report.html.*.tmp"))
    assert source.read_bytes() == original_source


def test_report_fails_when_exclusive_publication_is_unsupported(tmp_path, monkeypatch, report_input):
    _source, model = report_input
    output = tmp_path / "report.html"
    failure = OSError(errno.EOPNOTSUPP, "exclusive publication unsupported")

    def fail_link(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr("plotly.offline.get_plotlyjs", lambda: "/* test Plotly bundle */")
    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="exclusive publication unsupported") as error:
        write_report(model, output)
    assert error.value is failure
    assert not output.exists()
    assert not list(tmp_path.glob(".report.html.*.tmp"))
