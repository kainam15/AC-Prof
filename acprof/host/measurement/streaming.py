"""OpenAI-compatible SSE observations; network chunks are never token counts."""
from __future__ import annotations

import codecs
import math
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from acprof.artifacts import loads_finite_json, reject_isolated_surrogates
from acprof.experiments.streaming import OPENAI_STREAMING_ENDPOINTS, prepare_streaming_payload

_METRICS = ("ttft_s", "tpot_s", "itl_s", "output_e2e_s", "e2e_s")


class StreamProtocolError(ValueError):
    """A classified failed response, including a server-side streaming error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class StreamTokenCountError(StreamProtocolError):
    def __init__(self, original: Exception):
        super().__init__(getattr(original, "code", "tokenizer_error"), str(original))
        self.original = original


class StreamCancelledError(RuntimeError):
    code = "cancelled"


class StreamRunBudget:
    def __init__(self, max_response_bytes: int, max_output_events: int):
        self.max_response_bytes = max_response_bytes
        self.max_output_events = max_output_events
        self._lock = threading.Lock()
        self._response_bytes = 0
        self._output_events = 0
        self._failure: tuple[str, str] | None = None

    def _raise_failure(self) -> None:
        if self._failure is not None:
            raise StreamProtocolError(*self._failure)

    def check(self) -> None:
        with self._lock:
            self._raise_failure()

    def reserve_response_bytes(self, count: int) -> None:
        with self._lock:
            self._raise_failure()
            if self._response_bytes + count > self.max_response_bytes:
                self._failure = ("run_response_limit", "run exceeds max_run_response_bytes")
                self._raise_failure()
            self._response_bytes += count

    def reserve_output_event(self) -> None:
        with self._lock:
            self._raise_failure()
            if self._output_events >= self.max_output_events:
                self._failure = ("run_output_event_limit", "run exceeds max_run_output_events")
                self._raise_failure()
            self._output_events += 1


class _Cancellation(Protocol):
    def is_set(self) -> bool: ...


class _Response(Protocol):
    status: int

    def getheader(self, name: str) -> str | None: ...
    def read1(self, size: int) -> bytes: ...


@dataclass(frozen=True)
class MatchedTokenCounter:
    """Explicitly matched tokenizer, loaded before the measured request window."""

    count: Callable[[str], int]
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value.strip() for value in (
            self.model_id, self.model_revision, self.tokenizer_id, self.tokenizer_revision,
        )):
            raise ValueError("tokenizer fallback requires explicit model/tokenizer identities and revisions")
        if self.model_id != self.tokenizer_id or self.model_revision != self.tokenizer_revision:
            raise ValueError("tokenizer fallback must match the model identity and revision")

    def identity(self) -> dict[str, str]:
        return {
            "model_id": self.model_id, "model_revision": self.model_revision,
            "tokenizer_id": self.tokenizer_id, "tokenizer_revision": self.tokenizer_revision,
            "scope": "decoded_output_text_without_added_special_tokens",
        }


@dataclass(frozen=True)
class SSEEvent:
    event: str
    data: str


class SSEDecoder:
    """Strict UTF-8 event framing with bounded partial events and no EOF dispatch."""

    def __init__(self, max_event_bytes: int = 1024 * 1024):
        if type(max_event_bytes) is not int or max_event_bytes < 1:
            raise ValueError("max_event_bytes must be a positive integer")
        self.max_event_bytes = max_event_bytes
        self._decoder = codecs.getincrementaldecoder("utf-8-sig")("strict")
        self._line_parts: list[str] = []
        self._line_bytes = 0
        self._event_bytes = 0
        self._data: list[str] = []
        self._event = ""
        self._skip_lf = False

    def _append(self, text: str) -> None:
        if text:
            self._line_parts.append(text)
            self._line_bytes += len(text.encode("utf-8"))
        if self._event_bytes + self._line_bytes > self.max_event_bytes:
            raise StreamProtocolError("event_limit", "SSE event exceeds max_event_bytes")

    def _line(self) -> SSEEvent | None:
        line = "".join(self._line_parts)
        self._line_parts.clear()
        self._event_bytes += self._line_bytes + 1
        self._line_bytes = 0
        if not line:
            result = SSEEvent(self._event or "message", "\n".join(self._data)) if (
                self._data or self._event == "error"
            ) else None
            self._data.clear()
            self._event = ""
            self._event_bytes = 0
            return result
        if self._event_bytes > self.max_event_bytes:
            raise StreamProtocolError("event_limit", "SSE event exceeds max_event_bytes")
        if line.startswith(":"):
            return None
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "data":
            self._data.append(value)
        elif field == "event":
            self._event = value
        return None

    def _decode(self, chunk: bytes) -> str:
        try:
            return self._decoder.decode(chunk)
        except UnicodeDecodeError as error:
            raise StreamProtocolError("invalid_utf8", "SSE is not valid UTF-8") from error

    def iter_events(self, chunk: bytes) -> Iterator[SSEEvent]:
        """Deliver complete events before decoding a possibly malformed suffix."""
        if self._skip_lf and chunk:
            chunk = chunk[1:] if chunk.startswith(b"\n") else chunk
            self._skip_lf = False
        start = 0
        for match in re.finditer(rb"\r\n|\r|\n", chunk):
            end = match.end()
            # Include the delimiter in strict decoding so an incomplete UTF-8
            # character cannot cross a line boundary and create a false event.
            text = self._decode(chunk[start:end])
            self._append(text[:-len(match.group())])
            event = self._line()
            self._skip_lf = match.group() == b"\r" and end == len(chunk)
            start = end
            if event is not None:
                yield event
        self._append(self._decode(chunk[start:]))

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        """Collect a chunk; consumers retaining failure evidence use iter_events."""
        return list(self.iter_events(chunk))

    def finish(self) -> None:
        try:
            self._decoder.decode(b"", final=True)
        except UnicodeDecodeError as error:
            raise StreamProtocolError("invalid_utf8", "SSE ended within a UTF-8 character") from error
        if self._line_bytes or self._data or self._event:
            raise StreamProtocolError("truncated_sse", "SSE ended before the event's blank-line delimiter")


def _count(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise StreamProtocolError("invalid_token_count", f"{label} must be a nonnegative integer")
    return value


def _event_token_evidence(choice: dict[str, Any], delta: dict[str, Any]) -> tuple[int | None, str | None]:
    evidence: list[tuple[int, str]] = []
    for container in (choice, delta):
        if "token_ids" in container and container["token_ids"] is not None:
            ids = container["token_ids"]
            if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
                raise StreamProtocolError("invalid_token_evidence", "token_ids must be nonnegative integers")
            evidence.append((len(ids), "token_ids"))
    logprobs = choice.get("logprobs")
    if logprobs is not None:
        if not isinstance(logprobs, dict):
            raise StreamProtocolError("invalid_token_evidence", "logprobs must be an object")
        if logprobs.get("content") is not None:
            tokens = logprobs["content"]
            if not isinstance(tokens, list) or any(
                not isinstance(item, dict) or not isinstance(item.get("token"), str) for item in tokens
            ):
                raise StreamProtocolError("invalid_token_evidence", "logprobs.content must contain tokens")
            evidence.append((len(tokens), "logprobs"))
        elif logprobs.get("tokens") is not None:
            tokens = logprobs["tokens"]
            if not isinstance(tokens, list) or any(not isinstance(token, str) for token in tokens):
                raise StreamProtocolError("invalid_token_evidence", "logprobs.tokens must contain strings")
            evidence.append((len(tokens), "logprobs"))
    if not evidence:
        return None, None
    if len({count for count, _ in evidence}) != 1:
        raise StreamProtocolError("invalid_token_evidence", "conflicting per-event token counts")
    return evidence[0]


class OpenAIStreamingAdapter:
    """Optional text-generation protocol for the existing load scheduler."""

    def __init__(
        self, endpoint: str = "/v1/chat/completions",
        token_counter: MatchedTokenCounter | None = None, *,
        max_event_bytes: int = 1024 * 1024,
        max_response_bytes: int = 64 * 1024 * 1024,
        max_output_events: int = 100000,
        max_run_response_bytes: int = 256 * 1024 * 1024,
        max_run_output_events: int = 1000000,
    ):
        if endpoint not in OPENAI_STREAMING_ENDPOINTS:
            raise ValueError("unsupported OpenAI streaming endpoint")
        for name, value in (
            ("max_event_bytes", max_event_bytes), ("max_response_bytes", max_response_bytes),
            ("max_output_events", max_output_events),
            ("max_run_response_bytes", max_run_response_bytes),
            ("max_run_output_events", max_run_output_events),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.endpoint = endpoint
        self.token_counter = token_counter
        self.max_event_bytes = max_event_bytes
        self.max_response_bytes = max_response_bytes
        self.max_output_events = max_output_events
        self.max_run_response_bytes = max_run_response_bytes
        self.max_run_output_events = max_run_output_events

    def new_run_budget(self) -> StreamRunBudget:
        return StreamRunBudget(self.max_run_response_bytes, self.max_run_output_events)

    def prepare_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return prepare_streaming_payload(
            payload, endpoint=self.endpoint,
            tokenizer_model=self.token_counter.model_id if self.token_counter is not None else None,
        )

    def protocol(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "kind": "openai_streaming", "endpoint": self.endpoint,
            "clock": "perf_counter_relative_seconds", "latency_unit": "seconds",
            "tpot_unit": "seconds_per_output_token", "completion": "sse_done_then_http_eof",
            "ttft_boundary": "first_nonempty_text_or_explicit_token_output",
            "tpot_boundary": "last_output_minus_first_output_over_output_tokens_minus_one",
            "itl_requirement": "explicit_single_token_events_with_matching_generation_count",
            "tokenizer": self.token_counter.identity() if self.token_counter else None,
            "limits": {
                "max_event_bytes": self.max_event_bytes,
                "max_response_bytes": self.max_response_bytes,
                "max_output_events": self.max_output_events,
                "max_run_response_bytes": self.max_run_response_bytes,
                "max_run_output_events": self.max_run_output_events,
            },
        }

    @staticmethod
    def invalidate(record: dict[str, Any], reason: str) -> None:
        for name in _METRICS:
            record[name] = None
        record["missing_reasons"] = dict.fromkeys(_METRICS, reason)
        record["tpot_precision"] = "unavailable"
        record["itl_precision"] = "unavailable"
        record["stream_successful"] = False

    def _initialize(self, record: dict[str, Any]) -> None:
        record.update(
            output_events=[], output_event_intervals_s=[], observed_ttft_s=None,
            first_output_s=None, last_output_s=None, stream_completed_s=None,
            transport_completed_s=None, finish_reason=None, input_tokens=None,
            output_tokens=None, input_token_count_source="unavailable",
            output_token_count_source="unavailable", input_token_count_scope="unavailable",
            token_count_scope="unavailable",
            tokenizer=self.token_counter.identity() if self.token_counter else None,
            server_usage=None, stream_error_code=None, stream_finalized=False,
            output_usage_event_count=None, output_usage_status="unavailable",
        )
        self.invalidate(record, "incomplete_request")

    def _usage(self, record: dict[str, Any], usage: Any) -> None:
        if not isinstance(usage, dict):
            raise StreamProtocolError("invalid_usage", "usage must be an object")
        prior = dict(record["server_usage"] or {})
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if value is not None:
                _count(value, f"usage.{key}")
                if prior.get(key) is not None and value < prior[key]:
                    raise StreamProtocolError("invalid_usage", "cumulative usage decreased")
        # Validate only fields belonging to this incoming snapshot.
        if all(usage.get(key) is not None for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
            if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
                raise StreamProtocolError("invalid_usage", "total_tokens does not equal prompt plus completion tokens")
        if usage.get("total_tokens") is None and any(
            usage.get(key) is not None for key in ("prompt_tokens", "completion_tokens")
        ):
            # A total from an earlier cumulative snapshot cannot validate a new
            # partial snapshot whose individual counters have advanced.
            prior.pop("total_tokens", None)
        # A later null counter must not erase a previously observed valid count.
        record["server_usage"] = {
            **prior, **{key: value for key, value in usage.items() if value is not None},
        }
        for key, field in (("prompt_tokens", "input"), ("completion_tokens", "output")):
            if usage.get(key) is not None:
                record[f"{field}_tokens"] = usage[key]
                record[f"{field}_token_count_source"] = "server_usage"
        if record["input_tokens"] is not None:
            record["input_token_count_scope"] = "server_prompt_tokens"
        if record["output_tokens"] is not None:
            record["token_count_scope"] = "server_generated_tokens"

    def _choice(
        self, record: dict[str, Any], choice: Any, received_s: float, run_budget: StreamRunBudget,
    ) -> None:
        if not isinstance(choice, dict) or ("index" in choice and (
            type(choice["index"]) is not int or choice["index"] != 0
        )):
            raise StreamProtocolError("invalid_choice", "only choice index 0 is supported")
        delta = choice.get("delta", {}) if self.endpoint.endswith("/chat/completions") else {}
        if not isinstance(delta, dict):
            raise StreamProtocolError("invalid_delta", "chat delta must be an object")
        if delta.get("tool_calls") or delta.get("function_call"):
            raise StreamProtocolError("unsupported_output", "tool/function output is not supported")
        values = (
            [delta.get("reasoning"), delta.get("reasoning_content"), delta.get("content")]
            if self.endpoint.endswith("/chat/completions") else [choice.get("text")]
        )
        if any(value is not None and not isinstance(value, str) for value in values):
            raise StreamProtocolError("unsupported_output", "output deltas must be text")
        if self.endpoint.endswith("/chat/completions") and values[0] and values[1]:
            raise StreamProtocolError("unsupported_output", "ambiguous simultaneous reasoning aliases")
        text = "".join(value for value in values if value is not None)
        count, source = _event_token_evidence(choice, delta)
        if text and count == 0:
            raise StreamProtocolError("invalid_token_evidence", "nonempty output has zero explicit tokens")
        if text or (count is not None and count > 0):
            if record["finish_reason"] is not None:
                raise StreamProtocolError("output_after_finish", "output followed a finish reason")
            if len(record["output_events"]) >= self.max_output_events:
                raise StreamProtocolError("output_event_limit", "stream exceeds max_output_events")
            run_budget.reserve_output_event()
            previous = record["last_output_s"]
            if previous is None:
                record["first_output_s"] = received_s
                record["observed_ttft_s"] = received_s - record["sent_s"]
            else:
                record["output_event_intervals_s"].append(received_s - previous)
            record["last_output_s"] = received_s
            record["output_events"].append({
                "received_s": received_s, "text": text, "token_count": count,
                "token_count_source": source,
            })
        finish = choice.get("finish_reason")
        if finish is not None:
            if not isinstance(finish, str) or not finish:
                raise StreamProtocolError("invalid_finish_reason", "finish_reason must be nonempty text")
            record["finish_reason"] = finish

    def _event(
        self, record: dict[str, Any], event: SSEEvent, received_s: float, run_budget: StreamRunBudget,
    ) -> None:
        if event.event == "error":
            raise StreamProtocolError("server_error", "server emitted an SSE error event")
        if record["stream_completed_s"] is not None:
            raise StreamProtocolError("data_after_done", "SSE data followed [DONE]")
        if event.data.strip() == "[DONE]":
            record["stream_completed_s"] = received_s
            record["completed_s"] = received_s
            return
        try:
            payload = loads_finite_json(event.data)
        except (TypeError, ValueError) as error:
            raise StreamProtocolError("invalid_json", "SSE data is not finite JSON") from error
        try:
            reject_isolated_surrogates(payload)
        except ValueError as error:
            raise StreamProtocolError("invalid_unicode", "SSE JSON contains an isolated surrogate") from error
        if not isinstance(payload, dict):
            raise StreamProtocolError("invalid_json", "SSE data must be a JSON object")
        if payload.get("error") is not None:
            raise StreamProtocolError("server_error", "server returned an error inside HTTP 200")
        if payload.get("usage") is not None:
            self._usage(record, payload["usage"])
        choices = payload.get("choices")
        if choices is None and payload.get("usage") is not None:
            choices = []
        if not isinstance(choices, list) or len(choices) > 1:
            raise StreamProtocolError("invalid_choices", "expected zero or one output choice")
        if choices:
            self._choice(record, choices[0], received_s, run_budget)
        usage = payload.get("usage")
        if usage is not None and usage.get("completion_tokens") is not None:
            # Usage covers the output in the same event as well.
            record["output_usage_event_count"] = len(record["output_events"])
            record["output_usage_status"] = "covers_last_output"

    @staticmethod
    def _derive_token_timings(
        record: dict[str, Any], events: list[dict[str, Any]], explicit: bool,
    ) -> None:
        count = record["output_tokens"]
        missing = record["missing_reasons"]
        if count is None:
            missing["tpot_s"] = "unknown_output_token_count"
        elif count < 2:
            missing["tpot_s"] = "fewer_than_two_output_tokens"
        elif len(events) < 2:
            missing["tpot_s"] = "fewer_than_two_output_events"
        else:
            record["tpot_s"] = (record["last_output_s"] - record["first_output_s"]) / (count - 1)
            record["tpot_precision"] = "client_output_event_estimate"
        if (
            count is not None and count > 1 and explicit and len(events) == count
            and record["token_count_scope"] == "server_generated_tokens"
        ):
            record["itl_s"] = list(record["output_event_intervals_s"])
            record["itl_precision"] = "client_receive_explicit_single_token_events"
            if record["tpot_s"] is not None:
                record["tpot_precision"] = "client_receive_explicit_single_token_events"
        else:
            missing["itl_s"] = "per_token_timing_unavailable"

    def _finish(self, record: dict[str, Any]) -> None:
        if record["stream_finalized"]:
            return
        events = record["output_events"]
        if (
            record["output_token_count_source"] == "server_usage"
            and record["output_usage_event_count"] != len(events)
        ):
            record.update(output_tokens=None, output_token_count_source="unavailable",
                          token_count_scope="unavailable", output_usage_status="stale_before_last_output")
        known_event_tokens = 0
        all_event_tokens_known = True
        for event in events:
            event_count = event["token_count"]
            if event_count is None:
                all_event_tokens_known = False
            else:
                known_event_tokens += event_count
        if record["output_tokens"] is None and events and all_event_tokens_known:
            record["output_tokens"] = known_event_tokens
            record["output_token_count_source"] = "server_event_tokens"
            record["token_count_scope"] = "server_generated_tokens"
        count = record["output_tokens"]
        if count == 0 and events:
            raise StreamProtocolError("invalid_usage", "zero output tokens contradict observed output")
        if (
            count is not None and record["token_count_scope"] == "server_generated_tokens"
            and known_event_tokens > count
        ):
            raise StreamProtocolError(
                "invalid_token_evidence", "observed event tokens exceed server output count",
            )
        record["e2e_s"] = record["stream_completed_s"] - record["sent_s"]
        record["stream_successful"] = True
        missing: dict[str, str] = {}
        if events:
            record["ttft_s"] = record["first_output_s"] - record["sent_s"]
            record["output_e2e_s"] = record["last_output_s"] - record["sent_s"]
        else:
            missing.update(ttft_s="no_output_event", output_e2e_s="no_output_event")
        record["missing_reasons"] = missing
        record["stream_finalized"] = True
        if record["output_tokens"] is None and self.token_counter is not None:
            try:
                text = "".join(event["text"] for event in events)
                count = _count(self.token_counter.count(text), "tokenizer output count")
                if count == 0 and events:
                    raise StreamProtocolError(
                        "invalid_usage", "zero output tokens contradict observed output",
                    )
            except Exception as error:
                raise StreamTokenCountError(error) from error
            record["output_tokens"] = count
            record["output_token_count_source"] = "matched_tokenizer"
            record["token_count_scope"] = "decoded_output_text_without_added_special_tokens"
        self._derive_token_timings(
            record, events, bool(events) and all_event_tokens_known and known_event_tokens == len(events),
        )

    def finalize(self, record: dict[str, Any]) -> None:
        if record.get("stream_successful"):
            self._finish(record)

    def consume(
        self, response: _Response, record: dict[str, Any], *,
        clock: Callable[[], float], cancelled: _Cancellation, defer_tokenization: bool = False,
        run_budget: StreamRunBudget | None = None,
    ) -> None:
        """Drain HTTP framing; all SSE events from one read share its timestamp."""
        self._initialize(record)
        run_budget = run_budget if run_budget is not None else self.new_run_budget()
        received_bytes = 0
        parser = SSEDecoder(self.max_event_bytes)
        previous = record["sent_s"]
        try:
            run_budget.check()
            if response.status != 200:
                raise StreamProtocolError("http_error", f"streaming HTTP status {response.status}")
            content_type = (response.getheader("Content-Type") or "").split(";", 1)[0].strip().lower()
            if content_type != "text/event-stream":
                raise StreamProtocolError("content_type", "streaming response is not text/event-stream")
            while True:
                if cancelled.is_set():
                    raise StreamCancelledError("streaming request cancelled")
                chunk = response.read1(64 * 1024)
                received_s = clock()
                if not isinstance(received_s, (int, float)) or not math.isfinite(received_s) or received_s < previous:
                    raise StreamProtocolError("invalid_clock", "request clock must be finite and monotonic")
                previous = received_s
                if cancelled.is_set():
                    raise StreamCancelledError("streaming request cancelled")
                if not chunk:
                    if getattr(response, "length", None) not in (None, 0):
                        raise StreamProtocolError("incomplete_http_body", "HTTP body ended before its Content-Length")
                    record["transport_completed_s"] = received_s
                    break
                received_bytes += len(chunk)
                run_budget.reserve_response_bytes(len(chunk))
                if received_bytes > self.max_response_bytes:
                    raise StreamProtocolError("response_limit", "stream exceeds max_response_bytes")
                for event in parser.iter_events(chunk):
                    self._event(record, event, received_s, run_budget)
            parser.finish()
            if record["stream_completed_s"] is None:
                raise StreamProtocolError("incomplete_stream", "stream ended without [DONE]")
            if defer_tokenization:
                record["stream_successful"] = True
            else:
                self._finish(record)
        except BaseException as error:
            code = getattr(error, "code", "transport_error")
            record["stream_error_code"] = code
            self.invalidate(record, code)
            raise
