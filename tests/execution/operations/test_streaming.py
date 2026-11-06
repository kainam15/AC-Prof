"""Deterministic SSE framing, token provenance and independently derived timings."""
from __future__ import annotations

import json
import threading

import pytest

from acprof.host.measurement.streaming import (
    MatchedTokenCounter,
    OpenAIStreamingAdapter,
    SSEDecoder,
    SSEEvent,
    StreamCancelledError,
    StreamProtocolError,
)


def frame(payload):
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {text}\n\n".encode()


def chat(text=None, **delta):
    if text is not None:
        delta["content"] = text
    return {"choices": [{"index": 0, "delta": delta}]}


def usage(count, prompt=7):
    return {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": count}}


class TimedResponse:
    status = 200

    def __init__(self, chunks, *, content_type="text/event-stream", eof=0.3):
        self.chunks = iter(chunks)
        self.content_type = content_type
        self.eof = eof
        self.now = 0.0
        self.reads = 0

    def getheader(self, name: str, default: str = "") -> str:
        return self.content_type

    def read1(self, size: int) -> bytes:
        self.reads += 1
        timestamp, chunk = next(self.chunks, (self.eof, b""))
        self.now = timestamp
        if isinstance(chunk, BaseException):
            raise chunk
        return chunk


def consume(chunks, adapter=None, *, response=None, cancelled=None, **options):
    response = response or TimedResponse(chunks)
    record = {"sent_s": 0.0, "completed_s": None}
    (adapter or OpenAIStreamingAdapter()).consume(
        response, record, clock=lambda: response.now,
        cancelled=cancelled or threading.Event(), **options,
    )
    return record


@pytest.mark.parametrize("width", [1, 2, 3, 7, 4096])
def test_sse_incremental_utf8_bom_multiline_and_all_line_endings(width):
    raw = '\ufeff: heartbeat\r\nid: 2\r\nretry: 1\r\ndata:{"text":\r\ndata: "中文"}\r\n\r\nevent:custom\rdata:other\r\ndata:\r\r'.encode()
    decoder = SSEDecoder()
    events = []
    for offset in range(0, len(raw), width):
        events.extend(decoder.feed(raw[offset:offset + width]))
    decoder.finish()
    assert events == [
        SSEEvent("message", '{"text":\n"中文"}'),
        SSEEvent("custom", "other\n"),
    ]


def test_sse_does_not_dispatch_complete_json_before_blank_line():
    decoder = SSEDecoder()
    assert decoder.feed(b'data: {"choices":[]}\n') == []
    with pytest.raises(StreamProtocolError, match="blank-line") as error:
        decoder.finish()
    assert error.value.code == "truncated_sse"


@pytest.mark.parametrize("raw", [b"data: \xff\n\n", b"data: \xe4\xb8"])
def test_sse_rejects_invalid_or_truncated_utf8(raw):
    decoder = SSEDecoder()
    def decode_to_eof():
        decoder.feed(raw)
        decoder.finish()

    with pytest.raises(StreamProtocolError) as error:
        decode_to_eof()
    assert error.value.code == "invalid_utf8"


@pytest.mark.parametrize("evidence", ["none", "token_ids", "tokenizer"])
def test_final_usage_must_cover_last_output_event(evidence):
    first = chat("AB", **({"token_ids": [1, 2]} if evidence == "token_ids" else {}))
    first["usage"] = {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
    second = chat("C", **({"token_ids": [3]} if evidence == "token_ids" else {}))
    second["usage"] = {"completion_tokens": None}
    counter = MatchedTokenCounter(lambda text: 3, "model", "rev", "model", "rev") if evidence == "tokenizer" else None
    record = consume([
        (0.1, frame(first)), (0.2, frame(second)), (0.3, frame("[DONE]")),
    ], OpenAIStreamingAdapter(token_counter=counter))
    assert record["output_tokens"] == (None if evidence == "none" else 3)
    assert record["output_usage_status"] == "stale_before_last_output"
    if evidence == "none":
        assert record["tpot_s"] is None
        assert record["output_token_count_source"] == "unavailable"
    else:
        assert record["tpot_s"] == pytest.approx(0.05)
        assert record["output_token_count_source"] == (
            "server_event_tokens" if evidence == "token_ids" else "matched_tokenizer"
        )


def test_partial_usage_update_does_not_reuse_old_total_and_covers_same_event():
    first = chat("AB")
    first["usage"] = {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
    second = chat("C")
    second["usage"] = {"completion_tokens": 3}
    record = consume([(0.1, frame(first)), (0.2, frame(second)), (0.3, frame("[DONE]"))])
    assert record["output_tokens"] == 3
    assert record["output_token_count_source"] == "server_usage"
    assert record["output_usage_status"] == "covers_last_output"
    assert record["tpot_s"] == pytest.approx(0.05)


def test_timing_uses_outputs_and_separates_done_from_last_output():
    record = consume([
        (0.05, frame(chat(role="assistant"))),
        (0.10, frame(chat("A"))),
        (0.14, frame(chat("BCD"))),
        (0.18, frame(chat("E"))),
        (0.21, frame(usage(5))),
        (0.23, frame("[DONE]")),
    ])
    assert record["ttft_s"] == pytest.approx(0.10)
    assert record["output_e2e_s"] == pytest.approx(0.18)
    assert record["e2e_s"] == pytest.approx(0.23)
    assert record["tpot_s"] == pytest.approx(0.02)
    assert record["output_event_intervals_s"] == pytest.approx([0.04, 0.04])
    assert record["itl_s"] is None
    assert record["tpot_precision"] == "client_output_event_estimate"
    assert record["completed_s"] == record["stream_completed_s"] == 0.23
    assert record["transport_completed_s"] == 0.3
    assert record["output_tokens"] == 5
    assert record["input_tokens"] == 7
    assert record["output_token_count_source"] == "server_usage"
    assert record["input_token_count_scope"] == "server_prompt_tokens"
    assert record["token_count_scope"] == "server_generated_tokens"
    assert len(record["output_events"]) == 3


def test_all_events_from_one_network_read_share_the_receive_time():
    body = b"".join(frame(item) for item in [chat("A"), chat("B"), usage(2), "[DONE]"])
    record = consume([(0.1, body)])
    assert [event["received_s"] for event in record["output_events"]] == [0.1, 0.1]
    assert record["output_event_intervals_s"] == [0.0]
    assert record["itl_s"] is None
    assert record["output_tokens"] == 2


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_reasoning_is_first_output_and_empty_role_heartbeat_usage_are_not(field):
    record = consume([
        (0.01, b": ping\n\n"),
        (0.03, frame(chat(role="assistant"))),
        (0.05, frame(chat(""))),
        (0.10, frame(chat(**{field: "think"}))),
        (0.15, frame(chat("answer"))),
        (0.16, frame(usage(6))),
        (0.2, frame("[DONE]")),
    ])
    assert record["ttft_s"] == 0.10
    assert [event["text"] for event in record["output_events"]] == ["think", "answer"]
    assert record["tpot_s"] == pytest.approx(0.01)


@pytest.mark.parametrize("count", [None, 0, 1])
def test_zero_single_and_unknown_counts_do_not_create_tpot_or_itl(count):
    chunks = [] if count == 0 else [(0.1, frame(chat("x")))]
    if count is not None:
        chunks.append((0.15, frame(usage(count))))
    chunks.append((0.2, frame("[DONE]")))
    record = consume(chunks)
    assert record["output_tokens"] == count
    assert record["tpot_s"] is None
    assert record["itl_s"] is None
    assert record["e2e_s"] == 0.2
    assert record["ttft_s"] == (None if count == 0 else 0.1)


def test_single_event_with_many_tokens_has_no_invented_zero_tpot():
    record = consume([(0.1, frame(chat("many tokens"))), (0.15, frame(usage(5))), (0.2, frame("[DONE]"))])
    assert record["tpot_s"] is None
    assert record["missing_reasons"]["tpot_s"] == "fewer_than_two_output_events"
    assert record["output_event_intervals_s"] == []


@pytest.mark.parametrize("evidence", ["token_ids", "logprobs"])
def test_explicit_one_token_events_allow_client_receive_itl(evidence):
    events = []
    for text, token_id in [("A", 12), ("B", 14), ("C", 17)]:
        choice = {"index": 0, "delta": {"content": text}}
        if evidence == "token_ids":
            choice["token_ids"] = [token_id]
        else:
            choice["logprobs"] = {"content": [{"token": text, "logprob": -0.1}]}
        events.append({"choices": [choice]})
    record = consume([
        (0.1, frame(events[0])), (0.14, frame(events[1])), (0.20, frame(events[2])),
        (0.22, frame(usage(3))), (0.25, frame("[DONE]")),
    ])
    assert record["itl_s"] == pytest.approx([0.04, 0.06])
    assert record["tpot_s"] == pytest.approx(0.05)
    assert record["itl_precision"] == "client_receive_explicit_single_token_events"


def test_explicit_token_evidence_can_count_without_usage_but_merged_event_cannot_yield_itl():
    adapter = OpenAIStreamingAdapter("/v1/completions")
    record = consume([
        (0.1, frame({"choices": [{"index": 0, "text": "AB", "token_ids": [1, 2]}]})),
        (0.2, frame({"choices": [{"index": 0, "text": "C", "token_ids": [3]}]})),
        (0.25, frame("[DONE]")),
    ], adapter)
    assert record["output_tokens"] == 3
    assert record["output_token_count_source"] == "server_event_tokens"
    assert record["tpot_s"] == pytest.approx(0.05)
    assert record["itl_s"] is None


def test_matching_event_and_usage_totals_do_not_prove_each_event_is_one_token():
    record = consume([(0.1, frame(chat("A"))), (0.2, frame(chat("BCD"))), (0.22, frame(usage(2))), (0.25, frame("[DONE]"))])
    assert record["itl_s"] is None


def test_usage_mismatch_disables_token_itl_but_does_not_replace_server_usage():
    record = consume([
        (0.1, frame({"choices": [{"delta": {"content": "A"}, "token_ids": [1]}]})),
        (0.2, frame({"choices": [{"delta": {"content": "B"}, "token_ids": [2]}]})),
        (0.22, frame(usage(3))), (0.25, frame("[DONE]")),
    ])
    assert record["output_tokens"] == 3
    assert record["output_token_count_source"] == "server_usage"
    assert record["itl_s"] is None


def test_explicit_empty_special_token_is_an_output_event():
    record = consume([
        (0.1, frame({"choices": [{"delta": {"content": ""}, "token_ids": [10]}]})),
        (0.2, frame("[DONE]")),
    ])
    assert record["ttft_s"] == 0.1
    assert record["output_tokens"] == 1


def test_matched_tokenizer_counts_whole_output_once_and_can_be_deferred():
    calls = []

    def counter(text):
        calls.append(text)
        return 2

    tokenizer = MatchedTokenCounter(counter, "model", "abc", "model", "abc")
    adapter = OpenAIStreamingAdapter(token_counter=tokenizer)
    record = consume([
        (0.1, frame(chat("hel"))), (0.2, frame(chat("lo"))), (0.25, frame("[DONE]")),
    ], adapter, defer_tokenization=True)
    assert calls == []
    assert record["output_tokens"] is None
    assert record["ttft_s"] is None
    assert record["e2e_s"] is None
    adapter.finalize(record)
    adapter.finalize(record)
    assert calls == ["hello"]
    assert record["output_tokens"] == 2
    assert record["ttft_s"] == 0.1
    assert record["e2e_s"] == 0.25
    assert record["tpot_s"] == pytest.approx(0.1)
    assert record["output_token_count_source"] == "matched_tokenizer"
    assert record["token_count_scope"] == "decoded_output_text_without_added_special_tokens"


def test_zero_server_usage_wins_over_fallback_and_invalidated_record_is_not_tokenized():
    calls = []
    counter = MatchedTokenCounter(lambda text: calls.append(text) or 5, "model", "abc", "model", "abc")
    adapter = OpenAIStreamingAdapter(token_counter=counter)
    record = consume([(0.1, frame(usage(0))), (0.2, frame("[DONE]"))], adapter)
    assert record["output_tokens"] == 0
    assert calls == []
    partial = {"sent_s": 0.0}
    response = TimedResponse([(0.1, frame(chat("partial")))])
    with pytest.raises(StreamProtocolError, match="without"):
        adapter.consume(response, partial, clock=lambda: response.now, cancelled=threading.Event())
    adapter.finalize(partial)
    assert calls == []


@pytest.mark.parametrize(("suffix", "code"), [
    (b"", "incomplete_stream"),
    (b'data: {"choices":[]}\n', "truncated_sse"),
    (b"data: nope\n\n", "invalid_json"),
    (b'data: {"value":NaN}\n\n', "invalid_json"),
    (b'data: {"error":{"message":"failed"}}\n\n', "server_error"),
    (b"event: error\ndata: failed\n\n", "server_error"),
    (frame("[DONE]") + frame(chat("after")), "data_after_done"),
    (frame({"choices": [{"delta": {"tool_calls": [{"id": "tool"}]}}]}), "unsupported_output"),
    (frame({"choices": [{"delta": {}}, {"delta": {}}]}), "invalid_choices"),
    (frame(usage(True)) + frame("[DONE]"), "invalid_token_count"),
])
def test_failed_stream_preserves_observations_and_invalidates_success_metrics(suffix, code):
    record = {"sent_s": 0.0, "completed_s": None}
    response = TimedResponse([(0.1, frame(chat("partial"))), (0.2, suffix)] if suffix else [(0.1, frame(chat("partial")))])
    with pytest.raises(StreamProtocolError) as error:
        OpenAIStreamingAdapter().consume(response, record, clock=lambda: response.now, cancelled=threading.Event())
    assert error.value.code == code
    assert record["observed_ttft_s"] == 0.1
    assert len(record["output_events"]) == 1
    assert not record["stream_successful"]
    assert all(record[name] is None for name in ("ttft_s", "tpot_s", "itl_s", "e2e_s", "output_e2e_s"))


@pytest.mark.parametrize("failure", [TimeoutError("read inactivity"), OSError("connection reset")])
def test_transport_error_retains_partial_output(failure):
    record = {"sent_s": 0.0}
    response = TimedResponse([(0.1, frame(chat("partial"))), (0.2, failure)])
    with pytest.raises(type(failure)):
        OpenAIStreamingAdapter().consume(response, record, clock=lambda: response.now, cancelled=threading.Event())
    assert record["observed_ttft_s"] == 0.1
    assert record["ttft_s"] is None
    assert record["stream_error_code"] == "transport_error"


def test_cancellation_does_not_read_body_or_produce_timings():
    stop = threading.Event()
    stop.set()
    record = {"sent_s": 0.0}
    response = TimedResponse([(0.1, frame(chat("ignored")))])
    with pytest.raises(StreamCancelledError):
        OpenAIStreamingAdapter().consume(response, record, clock=lambda: response.now, cancelled=stop)
    assert response.reads == 0
    assert record["stream_error_code"] == "cancelled"
    assert record["e2e_s"] is None


@pytest.mark.parametrize("payload", [
    b'data: {"choices":[],"nested":{"value":"\\ud800"}}\n\n',
    b'data: {"choices":[],"\\ud800":"nested-key"}\n\n',
    b'data: {"choices":[{"delta":{"content":"\\ud800"}}]}\n\n',
])
def test_sse_json_rejects_isolated_surrogates_in_values_and_keys(payload):
    with pytest.raises(StreamProtocolError) as error:
        consume([(0.1, payload), (0.2, frame("[DONE]"))])
    assert error.value.code == "invalid_unicode"
    assert error.value.__cause__ is not None


def test_sse_json_accepts_valid_surrogate_pairs():
    payload = b'data: {"choices":[],"nested":{"value":"\\ud83d\\ude00"}}\n\n'
    record = consume([(0.1, payload), (0.2, frame("[DONE]"))])
    assert record["stream_successful"]


@pytest.mark.parametrize(("options", "code"), [
    ({"max_event_bytes": 5}, "event_limit"),
    ({"max_response_bytes": 5}, "response_limit"),
    ({"max_output_events": 1}, "output_event_limit"),
])
def test_stream_limits_fail_explicitly(options, code):
    with pytest.raises(StreamProtocolError) as error:
        consume([(0.1, frame(chat("A"))), (0.2, frame(chat("B"))), (0.25, frame("[DONE]"))], OpenAIStreamingAdapter(**options))
    assert error.value.code == code


@pytest.mark.parametrize(("status", "content_type", "code"), [(503, "application/json", "http_error"), (200, "application/json", "content_type")])
def test_http_status_and_mime_are_validated_before_body(status, content_type, code):
    response = TimedResponse([], content_type=content_type)
    response.status = status
    with pytest.raises(StreamProtocolError) as error:
        consume([], response=response)
    assert error.value.code == code
    assert response.reads == 0


@pytest.mark.parametrize("mutation", [{"n": 2}, {"n": True}, {"stream": False}, {"stream_options": {"include_usage": False}}, {"echo": True}, {"tools": [{"name": "tool"}]}])
def test_payload_validation_refuses_ambiguous_measurement_modes(mutation):
    with pytest.raises(ValueError, match="n=1|stream|text output|echo"):
        OpenAIStreamingAdapter().prepare_payload({"model": "model", "messages": [{"role": "user", "content": "hi"}], **mutation})


def test_payload_preparation_is_nonmutating_and_tokenizer_identity_is_checked():
    payload = {"model": "model", "prompt": "hi", "stream_options": {"continuous_usage_stats": True}}
    prepared = OpenAIStreamingAdapter("/v1/completions").prepare_payload(payload)
    assert prepared["stream"] is True
    assert prepared["stream_options"] == {"continuous_usage_stats": True, "include_usage": True}
    assert "stream" not in payload
    assert "include_usage" not in payload["stream_options"]
    with pytest.raises(ValueError, match="match"):
        MatchedTokenCounter(len, "model", "abc", "other", "abc")
    with pytest.raises(ValueError, match="identities"):
        MatchedTokenCounter(len, "model", "", "model", "")


def test_done_does_not_hide_premature_content_length_eof():
    response = TimedResponse([(0.1, frame(chat("A"))), (0.2, frame("[DONE]"))])
    response.length = 20
    record = {"sent_s": 0.0}
    with pytest.raises(StreamProtocolError) as error:
        OpenAIStreamingAdapter().consume(response, record, clock=lambda: response.now, cancelled=threading.Event())
    assert error.value.code == "incomplete_http_body"
    assert record["stream_completed_s"] == 0.2
    assert record["e2e_s"] is None


def test_usage_total_is_checked_and_null_does_not_erase_valid_counts():
    record = consume([
        (0.1, frame(chat("A"))), (0.15, frame(usage(1))),
        (0.2, frame({"choices": [], "usage": {"completion_tokens": None}})),
        (0.25, frame("[DONE]")),
    ])
    assert record["output_tokens"] == 1
    assert record["server_usage"]["completion_tokens"] == 1
    with pytest.raises(StreamProtocolError) as error:
        consume([
            (0.1, frame(chat("A"))),
            (0.2, frame({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 100}})),
            (0.25, frame("[DONE]")),
        ])
    assert error.value.code == "invalid_usage"


def test_known_event_tokens_cannot_exceed_usage_count():
    with pytest.raises(StreamProtocolError) as error:
        consume([
            (0.1, frame({"choices": [{"delta": {"content": "AB"}, "token_ids": [1, 2]}]})),
            (0.2, frame(usage(1))), (0.25, frame("[DONE]")),
        ])
    assert error.value.code == "invalid_token_evidence"


def test_deferred_bad_counter_is_explicit_and_preserves_observed_request_timing():
    adapter = OpenAIStreamingAdapter(token_counter=MatchedTokenCounter(lambda _: True, "model", "abc", "model", "abc"))
    record = consume([(0.1, frame(chat("A"))), (0.2, frame("[DONE]"))], adapter, defer_tokenization=True)
    with pytest.raises(StreamProtocolError) as error:
        adapter.finalize(record)
    assert error.value.code == "invalid_token_count"
    assert record["ttft_s"] == 0.1
    assert record["e2e_s"] == 0.2
    assert record["output_tokens"] is None
    assert record["tpot_s"] is None

@pytest.mark.parametrize("partial", [{"prompt_tokens": 7}, {"completion_tokens": 3}])
def test_invalid_full_usage_is_not_masked_by_later_partial_snapshot(partial):
    with pytest.raises(StreamProtocolError) as error:
        consume([
            (0.1, frame(chat("ABC"))),
            (0.2, frame({"choices": [], "usage": {
                "prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 999,
            }})),
            (0.25, frame({"choices": [], "usage": partial})),
            (0.3, frame("[DONE]")),
        ])
    assert error.value.code == "invalid_usage"


def test_total_only_usage_does_not_validate_stale_counters_from_other_snapshot():
    record = consume([
        (0.1, frame({"choices": [{"delta": {"content": "A"}, "token_ids": [1]}],
                     "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8}})),
        (0.2, frame({"choices": [{"delta": {"content": "B"}, "token_ids": [2]}]})),
        (0.25, frame({"choices": [], "usage": {"total_tokens": 9}})),
        (0.3, frame("[DONE]")),
    ])
    assert record["output_tokens"] == 2
    assert record["output_token_count_source"] == "server_event_tokens"
    assert record["output_usage_status"] == "stale_before_last_output"
    assert record["tpot_s"] == pytest.approx(0.1)
    assert record["itl_s"] == pytest.approx([0.1])
