"""Post-load failures retain cancellation and preceding execution evidence."""
import json
from contextlib import nullcontext

import pytest

from acprof.cli import load as load_cli
from acprof.host.measurement.load import LoadConfig, run_load
from acprof.host.measurement.streaming import MatchedTokenCounter, OpenAIStreamingAdapter
from tests.execution.operations.test_load_connection_finalization import connection_faults
from tests.execution.operations.test_streaming_load import payload, streaming_server


@pytest.mark.parametrize(("primary_type", "secondary_type", "retained"), [
    (None, RuntimeError, "secondary"),
    (None, KeyboardInterrupt, "secondary"),
    (None, SystemExit, "secondary"),
    (OSError, RuntimeError, "primary"),
    (OSError, KeyboardInterrupt, "secondary"),
    (OSError, SystemExit, "secondary"),
    (KeyboardInterrupt, RuntimeError, "primary"),
    (KeyboardInterrupt, KeyboardInterrupt, "primary"),
    (KeyboardInterrupt, SystemExit, "primary"),
])
def test_finalization_preserves_failure_identity_and_completed_requests(
    monkeypatch, primary_type, secondary_type, retained,
):
    primary = primary_type("fixture execution failed") if primary_type is not None else None
    secondary = secondary_type("fixture finalization interrupted")
    expected = primary if retained == "primary" else secondary
    adapter = OpenAIStreamingAdapter()
    finalize = adapter.finalize
    finalized = []

    def interrupt(record):
        finalized.append(record["request_id"])
        if len(finalized) == 2:
            raise secondary
        finalize(record)

    monkeypatch.setattr(adapter, "finalize", interrupt)
    with streaming_server() as (url, *_rest):
        with connection_faults(monkeypatch, [primary]) if primary is not None else nullcontext():
            with pytest.raises(type(expected), match=str(expected)) as caught:
                run_load(url, payload("/v1/chat/completions"), LoadConfig(requests=3),
                         token="finalize", adapter=adapter)
    assert caught.value is expected
    if primary is not None:
        detail = "\n".join((str(expected), *expected.__notes__))
        assert str(primary) in detail
        assert str(secondary) in detail
        if retained == "secondary":
            assert expected.__cause__ is primary
    report = getattr(caught.value, "load_report")
    assert report["successful"] is False
    assert report["counts"] == {
        "offered": 3, "sent": 3, "succeeded": 3, "failed": 0, "dropped": 0, "cancelled": 0,
    }
    assert finalized == ["finalize:0", "finalize:1"]
    assert [row["request_id"] for row in report["requests"]] == [f"finalize:{index}" for index in range(3)]
    assert all(row["stream_completed_s"] is not None for row in report["requests"])
    assert report["requests"][0]["ttft_s"] > 0
    assert report["requests"][0]["tpot_s"] > 0


@pytest.mark.parametrize("primary_type", [None, OSError, KeyboardInterrupt])
@pytest.mark.parametrize("cancellation_type", [KeyboardInterrupt, SystemExit])
def test_tokenizer_cancellation_stops_derivation_without_losing_http_evidence(
    monkeypatch, primary_type, cancellation_type,
):
    primary = primary_type("fixture connection finalization failed") if primary_type is not None else None
    cancellation = cancellation_type("fixture tokenizer cancelled")
    expected = primary if primary_type is KeyboardInterrupt else cancellation
    calls = []
    with streaming_server(fault="no_usage") as (url, _ports, _paths, bodies, _event):
        def count(text):
            calls.append(text)
            assert len(bodies) == 3
            if len(calls) == 2:
                raise cancellation
            return 3

        adapter = OpenAIStreamingAdapter(token_counter=MatchedTokenCounter(
            count, "fixture", "rev", "fixture", "rev",
        ))
        with connection_faults(monkeypatch, [primary]) if primary is not None else nullcontext():
            with pytest.raises(type(expected), match=str(expected)) as caught:
                run_load(url, payload("/v1/chat/completions"), LoadConfig(requests=3),
                         token="tokenizer", adapter=adapter)
    assert caught.value is expected
    if primary is not None:
        detail = "\n".join((str(expected), *expected.__notes__))
        assert str(primary) in detail
        assert str(cancellation) in detail
    assert calls == ["Hello world!", "Hello world!"]
    report = getattr(caught.value, "load_report")
    assert report["successful"] is False
    assert report["counts"]["succeeded"] == 3
    assert all(row["stream_completed_s"] is not None for row in report["requests"])
    assert [row["output_tokens"] for row in report["requests"]] == [3, None, None]
    assert report["requests"][0]["output_token_count_source"] == "matched_tokenizer"
    assert report["requests"][1]["ttft_s"] > 0
    assert report["requests"][1]["tpot_s"] is None


@pytest.mark.parametrize("warmup", [0, 3])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_endpoint_report_keeps_both_failures_and_cancellation_status(tmp_path, monkeypatch, warmup, error_type):
    primary = OSError("fixture connection finalization failed")
    secondary = error_type("fixture stream finalization interrupted")
    expected = secondary if error_type is KeyboardInterrupt else primary
    finalize = OpenAIStreamingAdapter.finalize
    calls = []

    def interrupt(adapter, record):
        calls.append(record["request_id"])
        if len(calls) == 2:
            raise secondary
        finalize(adapter, record)

    monkeypatch.setattr(OpenAIStreamingAdapter, "finalize", interrupt)
    monkeypatch.setattr("acprof.cli.streaming_load.MeasurementLock", nullcontext)
    monkeypatch.setattr("acprof.cli.streaming_load.host_identity", lambda _root: {"hostname": "fixture"})
    output = tmp_path / "load"
    with streaming_server() as (url, *_rest):
        with connection_faults(monkeypatch, [primary]):
            with pytest.raises(type(expected), match=str(expected)) as caught:
                load_cli.main(["--base-url", url, "--protocol", "openai-chat", "--model", "fixture",
                               "--prompt", "hello", "--requests", "3", "--warmup", str(warmup),
                               "--output-dir", str(output)])
    assert caught.value is expected
    report = json.loads((output / "load.json").read_text())
    phase = "warmup" if warmup else "result"
    assert report["status"] == ("cancelled" if error_type is KeyboardInterrupt else "failed")
    assert report["successful"] is False
    assert str(primary) in report["phase_errors"][phase]["detail"]
    assert str(secondary) in report["phase_errors"][phase]["detail"]
    assert report[phase]["counts"]["succeeded"] == 3
    assert len(calls) == 2
    if warmup:
        assert "result" not in report
