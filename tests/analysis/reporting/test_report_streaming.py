"""Report assembly preserves literal substitutions without copying the full page."""

import errno
import gc
import tracemalloc
from typing import TextIO

import pytest

from acprof.analysis.model import AnalysisModel
from acprof.artifacts import atomic_write
from acprof.plotting.report import write_report

ESCAPED_DATA = '{"name": "__STYLE__ \\u003c/script\\u003e\\u0026\\u4e2d\\ud83d\\ude00"}'


@pytest.fixture
def resources(tmp_path, monkeypatch):
    directory = tmp_path / "assets"
    directory.mkdir()
    for name, content in {
        "report.css": "STYLE __APP__",
        "report.js": "APP __DATA__",
        "view_model.js": "VIEW __LICENSE__",
        "plotly-LICENSE.txt": '<MIT> & "author"',
    }.items():
        (directory / name).write_text(content, encoding="utf-8")
    monkeypatch.setattr("acprof.plotting.report.files", lambda _package: directory)
    monkeypatch.setattr("plotly.offline.get_plotlyjs", lambda: "Plotly </script> __STYLE__")
    monkeypatch.setattr("acprof.plotting.report.report_payload", lambda *_args, **_kwargs: {
        "name": "__STYLE__ </script>&中😀",
    })
    return directory


@pytest.mark.parametrize(("template", "expected"), [
    ("", ""),
    ("中文 __UNKNOWN__", "中文 __UNKNOWN__"),
    ("__DATA__", ESCAPED_DATA),
    ("头__STYLE____DATA__尾", "头STYLE __APP__" + ESCAPED_DATA + "尾"),
    ("__DATA__ / __DATA__", ESCAPED_DATA + " / " + ESCAPED_DATA),
    ("__PLOTLY__|__LICENSE__|__VIEW_MODEL__|__APP__",
     "Plotly <\\/script> __STYLE__|&lt;MIT&gt; &amp; &quot;author&quot;|VIEW __LICENSE__|APP __DATA__"),
])
def test_template_substitution_preserves_escaping_and_literal_asset_tokens(tmp_path, resources, template, expected):
    (resources / "report.html").write_text(template, encoding="utf-8")
    output = tmp_path / "report.html"
    result = write_report(AnalysisModel([], [], []), output)
    assert result == output
    assert output.read_bytes() == expected.encode("utf-8")
    assert not list(tmp_path.glob(".report.html.*.tmp"))


def test_large_embedded_asset_does_not_require_a_second_full_html_buffer(tmp_path, monkeypatch):
    # Allocate the supplied bundle before tracing, so the budget covers rendering
    # and encoding rather than the fixture's retained input string.
    bundle = "a" * (2 * 1024**2)
    monkeypatch.setattr("plotly.offline.get_plotlyjs", lambda: bundle)
    model = AnalysisModel([], [], [])
    output = tmp_path / "report.html"
    gc.collect()
    tracemalloc.start()
    try:
        write_report(model, output)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert output.stat().st_size > len(bundle)
    assert peak < 2 * len(bundle) + 1024**2


@pytest.mark.parametrize("cancel", [False, True])
def test_failure_after_a_written_fragment_leaves_no_partial_report(tmp_path, monkeypatch, resources, cancel):
    (resources / "report.html").write_text("prefix__STYLE__suffix", encoding="utf-8")
    output = tmp_path / "report.html"
    failure = KeyboardInterrupt("cancelled report") if cancel else OSError(errno.ENOSPC, "report disk full")
    opened = []
    written = []

    def publish(path, write, **kwargs):
        def fail_late(stream: TextIO):
            opened.append(stream)

            class FailingWriter:
                def write(self, fragment: str) -> int:
                    if written:
                        raise failure
                    written.append(fragment)
                    return stream.write(fragment)

            write(FailingWriter())

        atomic_write(path, fail_late, **kwargs)

    monkeypatch.setattr("acprof.plotting.report.atomic_write", publish)
    with pytest.raises(type(failure), match="cancelled report" if cancel else "report disk full") as error:
        write_report(AnalysisModel([], [], []), output)
    assert error.value is failure
    assert written == ["prefix"]
    assert opened[0].closed
    assert not output.exists()
    assert not list(tmp_path.glob(".report.html.*.tmp"))
