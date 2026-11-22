"""Check the public load/report paths for server-aborted text generation."""
from __future__ import annotations

import json
from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from acprof.analysis.load_report import read_load_report
from acprof.cli.main import main
from tests.execution.operations.test_streaming import frame, usage


def run_stream(tmp_path, monkeypatch, protocol, frames, *, split=False, warmup=0):
    raw = b"".join(frame(item) for item in frames)
    chunks = [raw[offset:offset + 7] for offset in range(0, len(raw), 7)] if split else [raw]
    response = Mock(status=200, will_close=False, length=0)
    response.getheader.return_value = "text/event-stream"
    response.read1.side_effect = [*chunks, b""]
    client = Mock(sock=None)
    client.getresponse.return_value = response
    monkeypatch.setattr("acprof.host.measurement.load_transport.http.client.HTTPConnection", Mock(return_value=client))
    monkeypatch.setattr("acprof.cli.streaming_load.MeasurementLock", nullcontext)
    monkeypatch.setattr("acprof.cli.streaming_load.host_identity", lambda _root: {"hostname": "fixture"})
    return main(["load", "--base-url", "http://127.0.0.1:1", "--protocol", protocol, "--model", "fixture",
                 "--prompt", "hello", "--requests", "1", "--warmup", str(warmup), "--slo-e2e-s", "10",
                 "--output-dir", str(tmp_path / "load")])


def choice(protocol, content="", finish=None):
    value = {"delta": {"content": content}} if protocol == "openai-chat" else {"text": content}
    return {"choices": [{"index": 0, **value, "finish_reason": finish}]}


@pytest.mark.parametrize("protocol", ["openai-chat", "openai-completions"])
@pytest.mark.parametrize("split", [False, True], ids=["whole", "fragmented"])
def test_server_abort_cannot_satisfy_load_slo(tmp_path, monkeypatch, protocol, split):
    frames = [choice(protocol, "partial"), choice(protocol, " end", "abort"), usage(2), "[DONE]"]
    code = run_stream(tmp_path, monkeypatch, protocol, frames, split=split)
    report = read_load_report(tmp_path / "load")
    result = report["result"]
    row, = result["requests"]
    assert code == 1
    assert not result["successful"]
    assert row["http_status"] == 200
    assert row["status"] == "error"
    assert row["error_code"] == row["stream_error_code"] == "server_aborted"
    assert row["finish_reason"] == "abort"
    assert [event["text"] for event in row["output_events"]] == ["partial", " end"]
    assert row["output_tokens"] == 2
    assert row["server_usage"]["completion_tokens"] == 2
    assert row["observed_ttft_s"] is not None
    assert row["stream_completed_s"] is not None
    assert row["ttft_s"] is row["e2e_s"] is None
    assert result["counts"]["failed"] == 1
    assert result["counts"]["succeeded"] == 0
    assert result["streaming"]["e2e_s"]["sample_count"] == 0
    assert result["capacity"]["slo"]["counts"]["satisfied"] == 0
    assert result["capacity"]["goodput_requests_per_s"] == 0
    assert result["capacity"]["token_throughput"]["output"]["tokens"] is None


@pytest.mark.parametrize("finish", ["stop", "length", "content_filter"])
def test_normal_finish_remains_successful(tmp_path, monkeypatch, finish):
    protocol = "openai-chat"
    frames = [choice(protocol, "partial"), choice(protocol, finish=finish), usage(1), "[DONE]"]
    assert run_stream(tmp_path, monkeypatch, protocol, frames) == 0
    result = read_load_report(tmp_path / "load")["result"]
    assert result["capacity"]["slo"]["counts"]["satisfied"] == 1
    assert result["requests"][0]["finish_reason"] == finish


def test_server_error_before_abort_keeps_original_failure(tmp_path, monkeypatch):
    protocol = "openai-completions"
    frames = [choice(protocol, "partial"), usage(1), {"error": {"message": "server failure"}},
              choice(protocol, finish="abort"), "[DONE]"]
    assert run_stream(tmp_path, monkeypatch, protocol, frames, split=True) == 1
    row, = read_load_report(tmp_path / "load")["result"]["requests"]
    assert row["error_code"] == "server_error"
    assert row["finish_reason"] is None
    assert row["output_tokens"] == 1
    assert row["output_events"][0]["text"] == "partial"


def test_abort_during_warmup_prevents_measured_load(tmp_path, monkeypatch):
    protocol = "openai-chat"
    frames = [choice(protocol, finish="abort"), "[DONE]"]
    with pytest.raises(RuntimeError, match="warmup failed"):
        run_stream(tmp_path, monkeypatch, protocol, frames, warmup=1)
    experiment = json.loads((tmp_path / "load" / "load.json").read_text())
    assert "result" not in experiment
    assert experiment["warmup"]["requests"][0]["error_code"] == "server_aborted"


@pytest.mark.parametrize(("finish", "with_done", "expected_error"), [
    ("stop", True, "server_aborted"), (None, False, "incomplete_stream"),
])
def test_abort_cannot_become_successful_after_an_invalid_suffix(tmp_path, monkeypatch, finish, with_done, expected_error):
    protocol = "openai-chat"
    frames = [choice(protocol, "partial", "abort"), usage(1)]
    if finish is not None:
        frames.append(choice(protocol, finish=finish))
    if with_done:
        frames.append("[DONE]")
    assert run_stream(tmp_path, monkeypatch, protocol, frames) == 1
    row, = read_load_report(tmp_path / "load")["result"]["requests"]
    assert row["error_code"] == expected_error
    assert row["finish_reason"] == "abort"
    assert row["output_tokens"] == 1


@pytest.mark.parametrize("phase", ["result", "warmup"])
def test_reader_identifies_historical_false_success(tmp_path, monkeypatch, phase):
    protocol = "openai-completions"
    frames = [choice(protocol, "partial", "stop"), usage(1), "[DONE]"]
    assert run_stream(tmp_path, monkeypatch, protocol, frames) == 0
    path = tmp_path / "load" / "load.json"
    experiment = json.loads(path.read_text())
    if phase == "warmup":
        experiment["warmup"] = json.loads(json.dumps(experiment["result"]))
    experiment[phase]["requests"][0]["finish_reason"] = "abort"
    path.write_text(json.dumps(experiment))
    old_bytes = path.read_bytes()
    with pytest.raises(ValueError, match=rf"{phase}.*abort"):
        read_load_report(path)
    assert path.read_bytes() == old_bytes
