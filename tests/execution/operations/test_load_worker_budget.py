"""Finite loads start only useful workers while preserving the requested protocol."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.host.measurement.load import LoadConfig, run_load
from tests.execution.operations.test_load_connection_finalization import (
    endpoint_fixture as endpoint_fixture,
)
from tests.execution.operations.test_load_protocol import server


@pytest.fixture
def observed_workers(monkeypatch):
    observed = SimpleNamespace(threads=[], on_ready=lambda: None)
    guard = threading.Lock()

    def initialized():
        with guard:
            observed.threads.append(threading.current_thread())
            observed.on_ready()

    def executor(*args, **kwargs):
        return ThreadPoolExecutor(*args, **kwargs, initializer=initialized)

    monkeypatch.setattr("acprof.host.measurement.load.ThreadPoolExecutor", executor)
    return observed


@pytest.mark.parametrize(("requests", "concurrency"), [(1, 16), (2, 16), (8, 2), (3, 3)])
@pytest.mark.parametrize("scenario", ["concurrent", "arrival-rate"])
def test_worker_count_respects_request_budget_and_concurrency(endpoint, observed_workers, requests, concurrency, scenario):
    config = LoadConfig(
        scenario=scenario, requests=requests, concurrency=concurrency,
        rate=200 if scenario == "arrival-rate" else None, timeout_seconds=2,
    )
    identity = config.identity()
    report = run_load(endpoint.url, endpoint.payload, config, token="workers", adapter=endpoint.adapter)
    assert all(not thread.is_alive() for thread in observed_workers.threads)
    assert len(observed_workers.threads) == min(requests, concurrency)
    assert len(endpoint.ports) == requests
    assert report["successful"]
    assert report["counts"] == {
        "offered": requests, "sent": requests, "succeeded": requests,
        "failed": 0, "dropped": 0, "cancelled": 0,
    }
    assert [row["request_id"] for row in report["requests"]] == [f"workers:{index}" for index in range(requests)]
    assert report["protocol"]["concurrency"] == concurrency
    assert report["protocol"]["requests"] == requests
    assert config.identity() == identity
    if endpoint.adapter is None:
        assert report["protocol_sha256"] == identity
    if scenario == "arrival-rate":
        assert [row["planned_s"] for row in report["requests"]] == pytest.approx([index / 200 for index in range(requests)])


def test_worker_cancellation_keeps_partial_report_and_joins_limited_pool(endpoint, observed_workers, monkeypatch):
    failure = KeyboardInterrupt("fixture worker interrupted")
    monkeypatch.setattr("http.client.HTTPConnection.request", Mock(side_effect=failure))
    config = LoadConfig(scenario="concurrent", requests=2, concurrency=16, timeout_seconds=2)
    with pytest.raises(KeyboardInterrupt, match="fixture worker interrupted") as caught:
        run_load(endpoint.url, endpoint.payload, config, token="cancelled", adapter=endpoint.adapter)
    assert caught.value is failure
    assert all(not thread.is_alive() for thread in observed_workers.threads)
    assert len(observed_workers.threads) == 2
    report = getattr(failure, "load_report")
    assert report["successful"] is False
    assert report["counts"]["offered"] == report["counts"]["cancelled"] == 2
    assert report["protocol"]["concurrency"] == 16
    assert [row["request_id"] for row in report["requests"]] == ["cancelled:0", "cancelled:1"]
    assert endpoint.ports == []


def test_limited_worker_startup_stays_outside_load_duration(observed_workers, monkeypatch):
    clock = time.perf_counter
    state = SimpleNamespace(offset=0.0)

    def startup_cost():
        state.offset += 1000

    observed_workers.on_ready = startup_cost
    monkeypatch.setattr(time, "perf_counter", lambda: clock() + state.offset)
    with server() as (url, _ports, _peak):
        report = run_load(url, {}, LoadConfig(scenario="concurrent", requests=1, concurrency=16), token="startup")
    assert all(not thread.is_alive() for thread in observed_workers.threads)
    assert len(observed_workers.threads) == 1
    assert state.offset == 1000
    assert report["successful"]
    assert 0 < report["duration_s"] < 1000
    assert report["successful_requests_per_s"] == 1 / report["duration_s"]
