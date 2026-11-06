"""Request preparation validates the whole dataset without retaining unused copies."""
import hashlib
import json
import weakref
from copy import deepcopy
from typing import Any
from unittest.mock import Mock

import pytest

from acprof.host.measurement.load import LoadConfig, run_load
from acprof.host.measurement.streaming import OpenAIStreamingAdapter
from tests.execution.operations.test_streaming_load import streaming_server


class ObservedPayload(dict):
    """Weak-referenceable request body for observing normalized-copy lifetime."""


class ObservedAdapter(OpenAIStreamingAdapter):
    def __init__(self, endpoint):
        super().__init__(endpoint=endpoint)
        self.prepared = []
        self.peak_live = 0

    def prepare_payload(self, payload):
        prepared = ObservedPayload(super().prepare_payload(payload))
        self.prepared.append(weakref.ref(prepared))
        self.peak_live = max(self.peak_live, sum(reference() is not None for reference in self.prepared))
        return prepared


def request_rows(endpoint: str, count: int = 40) -> tuple[dict[str, Any], ...]:
    return tuple(
        {"model": "fixture", "max_tokens": index + 1,
         **({"messages": [{"role": "user", "content": f"请求 {index}"}]} if "chat" in endpoint
            else {"prompt": f"请求 {index}"})}
        for index in range(count)
    )


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/completions"])
@pytest.mark.parametrize("count", [1, 3])
def test_only_selected_normalized_inputs_remain_live_during_dataset_validation(endpoint, count):
    payload = request_rows(endpoint)
    original = deepcopy(payload)
    adapter = ObservedAdapter(endpoint)
    expected = [{**body, "stream": True, "n": 1, "stream_options": {"include_usage": True}}
                for body in original[:count]]
    digest = hashlib.sha256(json.dumps(expected, sort_keys=True, allow_nan=False).encode()).hexdigest()
    with streaming_server(endpoint=endpoint) as (origin, _ports, _paths, received, _ready):
        report = run_load(origin, payload, LoadConfig(requests=count), token="prefix", adapter=adapter)
    assert len(adapter.prepared) == len(payload)
    # The selected prefix plus a previous/current row may coexist while normalizing.
    assert adapter.peak_live <= count + 2
    assert all(reference() is None for reference in adapter.prepared)
    assert payload == original
    assert received == expected
    assert report["schema_version"] == 3
    assert report["successful"]
    assert report["counts"]["succeeded"] == count
    assert report["protocol"]["request_inputs_sha256"] == digest
    assert [row["input_index"] for row in report["requests"]] == list(range(count))


@pytest.mark.parametrize(("change", "message"), [
    ({"n": 2}, "n=1"),
    ({"model": "other-model"}, "one served model"),
    ({"api_key": "fixture-credential"}, "credentials"),
    ({"stream": False}, "stream=true"),
    ({"stream_options": {"include_usage": False}}, "include_usage=true"),
])
def test_invalid_unselected_row_is_rejected_before_workers_start(monkeypatch, change, message):
    payload = request_rows("/v1/chat/completions")
    payload[-1].update(change)
    original = deepcopy(payload)
    workers = Mock(side_effect=AssertionError("invalid dataset reached workers"))
    monkeypatch.setattr("acprof.host.measurement.load.ThreadPoolExecutor", workers)
    with pytest.raises(ValueError, match=message):
        run_load("http://127.0.0.1:9", payload, LoadConfig(requests=1), token="invalid", adapter=OpenAIStreamingAdapter())
    assert payload == original
    workers.assert_not_called()


def test_cancellation_while_validating_unused_rows_never_starts_workers(monkeypatch):
    rows = request_rows("/v1/chat/completions")
    original = deepcopy(rows)
    failure = KeyboardInterrupt("fixture unused row interrupted")

    class InterruptedAdapter(OpenAIStreamingAdapter):
        def prepare_payload(self, payload):
            if payload is rows[-1]:
                raise failure
            return super().prepare_payload(payload)

    workers = Mock(side_effect=AssertionError("cancelled dataset reached workers"))
    monkeypatch.setattr("acprof.host.measurement.load.ThreadPoolExecutor", workers)
    with pytest.raises(KeyboardInterrupt, match="fixture unused row interrupted") as caught:
        run_load("http://127.0.0.1:9", rows, LoadConfig(requests=1), token="cancelled", adapter=InterruptedAdapter())
    assert caught.value is failure
    assert not hasattr(failure, "load_report")
    assert rows == original
    workers.assert_not_called()
