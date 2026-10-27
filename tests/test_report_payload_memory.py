"""Large report lists serialize without retaining a complete JSON string."""

import gc
import json
import tracemalloc
from pathlib import Path

import pytest

from acprof.analysis.model import AnalysisModel
from acprof.cli.main import main
from acprof.plotting.report import write_report


@pytest.fixture
def json_resources(tmp_path, monkeypatch) -> Path:
    directory = tmp_path / "assets"
    directory.mkdir()
    for name in ("report.css", "report.js", "view_model.js", "plotly-LICENSE.txt"):
        (directory / name).write_text("", encoding="utf-8")
    (directory / "report.html").write_text("__DATA__", encoding="utf-8")
    monkeypatch.setattr("acprof.plotting.report.files", lambda _package: directory)
    monkeypatch.setattr("plotly.offline.get_plotlyjs", lambda: "")
    return directory


@pytest.mark.parametrize("payload", [
    {},
    {"null": None, "bool": True, "integer": 2**80, "float": -0.0, "small": 1e-20},
    {"configs": [], "sources": [], "tail": "end"},
    {"configs": [None, True, 42, 0.25, "text", [], {}, [1, [2, 3]]]},
    {"sources": [{"<key>&": "__DATA__ </script>&中文😀\\\"\n\ud800"}],
     "configs": [{"mapping": {1: "integer", None: "null", False: "false"}}]},
    {"before": {"nested": [[], {"items": ["__STYLE__", "<>&"]}]},
     "configs": [{"same": [1, 2]}] * 2, "after": ("tuple", False)},
], ids=["empty", "scalars", "empty-lists", "mixed-list", "escaping", "nested-and-shared"])
@pytest.mark.parametrize("template", ["__DATA__", "前__DATA__后|__DATA__"])
def test_payload_bytes_match_standard_json_and_keep_literal_tokens(
    tmp_path, monkeypatch, json_resources, payload, template,
):
    (json_resources / "report.html").write_text(template, encoding="utf-8")
    monkeypatch.setattr("acprof.plotting.report.report_payload", lambda *_args, **_kwargs: payload)
    expected = json.dumps(payload, ensure_ascii=True, allow_nan=False)
    expected = expected.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    output = tmp_path / "report.html"
    write_report(AnalysisModel([], [], []), output)
    assert output.read_bytes() == template.replace("__DATA__", expected).encode("utf-8")
    assert not list(tmp_path.glob(".report.html.*.tmp"))


def test_many_configurations_do_not_require_a_full_serialized_payload(tmp_path, json_resources):
    # Model allocation precedes tracing: only the extra serialization/write
    # allocation is bounded, not the caller's complete analysis model.
    name = "__DATA__ </script>&中文😀" * 8
    model = AnalysisModel([], [], [{"config_id": str(index), "model": name} for index in range(4096)])
    output = tmp_path / "report.html"
    gc.collect()
    tracemalloc.start()
    try:
        write_report(model, output)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert output.stat().st_size > 1024**2
    assert peak < 1024**2
    decoded = json.loads(output.read_bytes())
    assert len(decoded["configs"]) == 4096
    assert decoded["configs"][0]["model"] == decoded["configs"][-1]["model"] == name


@pytest.mark.parametrize(("invalid", "error"), [
    (float("nan"), ValueError), (float("inf"), ValueError), (float("-inf"), ValueError),
    (object(), TypeError), ({(1, 2): "invalid key"}, TypeError),
], ids=["nan", "positive-inf", "negative-inf", "unsupported-value", "unsupported-key"])
def test_late_invalid_record_never_publishes_partial_json(
    tmp_path, monkeypatch, json_resources, invalid, error,
):
    payload = {"configs": [{"model": "valid"}, {"model": invalid}], "tail": "unreached"}
    monkeypatch.setattr("acprof.plotting.report.report_payload", lambda *_args, **_kwargs: payload)
    output = tmp_path / "report.html"
    with pytest.raises(error):
        write_report(AnalysisModel([], [], []), output)
    assert not output.exists()
    assert not list(tmp_path.glob(".report.html.*.tmp"))


@pytest.mark.parametrize("cycle", ["list", "payload"])
def test_circular_payload_is_rejected_without_a_partial_report(tmp_path, monkeypatch, json_resources, cycle):
    entries: list[object] = [{"model": "valid"}]
    payload = {"configs": entries}
    entries.append(entries if cycle == "list" else payload)
    monkeypatch.setattr("acprof.plotting.report.report_payload", lambda *_args, **_kwargs: payload)
    output = tmp_path / "report.html"
    with pytest.raises(ValueError, match="Circular reference"):
        write_report(AnalysisModel([], [], []), output)
    assert not output.exists()
    assert not list(tmp_path.glob(".report.html.*.tmp"))


def test_cli_encoding_failure_has_no_success_receipt(tmp_path, monkeypatch, json_resources, capsys):
    source = tmp_path / "result_all.csv"
    original = b"cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status,latency_app_p95_s\n2,4,off,32,0,0,ok,0.04\n"
    source.write_bytes(original)
    payload = {"configs": [{"model": "valid"}, {"model": float("nan")}]}
    monkeypatch.setattr("acprof.plotting.report.report_payload", lambda *_args, **_kwargs: payload)
    output = tmp_path / "report.html"
    with pytest.raises(SystemExit) as error:
        main(["report", str(source), "--output", str(output)])
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "报告生成失败" in captured.err
    assert "Traceback" not in captured.err
    assert "ACPROF_REPORT" not in captured.out
    assert "报告已生成" not in captured.out
    assert source.read_bytes() == original
    assert not output.exists()
    assert not list(tmp_path.glob(".report.html.*.tmp"))
