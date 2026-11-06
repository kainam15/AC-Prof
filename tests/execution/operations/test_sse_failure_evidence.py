"""Complete SSE observations survive a malformed suffix in the same read."""
from __future__ import annotations

import threading
from typing import Any

import pytest

from acprof.host.measurement.streaming import OpenAIStreamingAdapter, StreamProtocolError
from tests.execution.operations.test_streaming import TimedResponse, chat, frame, usage


@pytest.mark.parametrize(("suffix", "code"), [
    (b"data: \xff\n\n", "invalid_utf8"),
    (b"data: \xe4\n\n", "invalid_utf8"),
    (b"data: " + b"x" * 200 + b"\n\n", "event_limit"),
], ids=["invalid-byte", "incomplete-character", "event-limit"])
@pytest.mark.parametrize("line_ending", [b"\n", b"\r", b"\r\n"], ids=["lf", "cr", "crlf"])
@pytest.mark.parametrize("packetization", ["coalesced", "events", "bytes"])
@pytest.mark.parametrize("done", [False, True])
def test_stream_failure_keeps_complete_prefix(suffix, code, line_ending, packetization, done):
    parts = [frame(chat("中")), frame(chat("文")), frame(usage(2))]
    if done:
        parts.append(frame("[DONE]"))
    parts.append(suffix)
    parts = [part.replace(b"\n", line_ending) for part in parts]
    raw = b"\xef\xbb\xbf" + b"".join(parts)
    chunks = {"coalesced": [raw], "events": [b"\xef\xbb\xbf", *parts],
              "bytes": [raw[index:index + 1] for index in range(len(raw))]}[packetization]
    response = TimedResponse([(0.125, chunk) for chunk in chunks])
    record: dict[str, Any] = {"sent_s": 0.0, "completed_s": None}
    adapter = OpenAIStreamingAdapter(max_event_bytes=160)

    with pytest.raises(StreamProtocolError) as caught:
        adapter.consume(response, record, clock=lambda: response.now, cancelled=threading.Event())

    assert caught.value.code == code
    assert [event["text"] for event in record["output_events"]] == ["中", "文"]
    assert [event["received_s"] for event in record["output_events"]] == [0.125, 0.125]
    assert record["observed_ttft_s"] == 0.125
    assert record["first_output_s"] == record["last_output_s"] == 0.125
    assert record["output_event_intervals_s"] == [0.0]
    assert record["input_tokens"] == 7
    assert record["output_tokens"] == 2
    assert record["stream_completed_s"] == (0.125 if done else None)
    assert record["transport_completed_s"] is None
    assert record["stream_error_code"] == code
    assert not record["stream_successful"]
    assert all(record[name] is None for name in ("ttft_s", "tpot_s", "itl_s", "e2e_s", "output_e2e_s"))


@pytest.mark.parametrize("suffix", [b"data: \xff\n\n", b"data: " + b"x" * 200 + b"\n\n"],
                         ids=["invalid-byte", "event-limit"])
@pytest.mark.parametrize(("bad_event", "code"), [
    (frame({"error": {"message": "generation failed"}}), "server_error"),
    (b"data: not-json\n\n", "invalid_json"),
], ids=["server-error", "invalid-json"])
def test_first_invalid_event_is_not_masked_by_later_framing_error(suffix, bad_event, code):
    response = TimedResponse([(0.125, frame(chat("kept")) + bad_event + suffix)])
    record: dict[str, Any] = {"sent_s": 0.0}
    with pytest.raises(StreamProtocolError) as caught:
        OpenAIStreamingAdapter(max_event_bytes=160).consume(
            response, record, clock=lambda: response.now, cancelled=threading.Event(),
        )
    assert caught.value.code == code
    assert record["stream_error_code"] == code
    assert [event["text"] for event in record["output_events"]] == ["kept"]
    assert record["ttft_s"] is None


def test_failure_keeps_the_receive_time_of_each_completed_event():
    first = frame(chat("first"))
    response = TimedResponse([
        (0.1, first[:-1]),
        (0.2, first[-1:] + frame(chat("second")) + b"data: \xff\n\n"),
    ])
    record: dict[str, Any] = {"sent_s": 0.0}
    with pytest.raises(StreamProtocolError, match="UTF-8"):
        OpenAIStreamingAdapter().consume(
            response, record, clock=lambda: response.now, cancelled=threading.Event(),
        )
    assert [event["received_s"] for event in record["output_events"]] == [0.2, 0.2]
    assert record["observed_ttft_s"] == 0.2
    assert record["output_event_intervals_s"] == [0.0]
