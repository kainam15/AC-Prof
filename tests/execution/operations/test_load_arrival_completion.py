"""Arrival admission reclaims completions without polling the entire backlog."""
import gc
import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from acprof.host.measurement.load import LoadConfig, run_load


@contextmanager
def controlled_load(monkeypatch):
    """Use real workers and Futures, holding responses until admission ends."""
    state = SimpleNamespace(
        release=threading.Event(), done_checks=0, immediate=False, response_failure=None,
        futures=weakref.WeakSet(), references=[], threads=[], clients=[], sent=[],
    )
    pools = []
    original_done, original_result = Future.done, Future.result

    def observed_done(future):
        if future in state.futures:
            state.done_checks += 1
        return original_done(future)

    def observed_result(future, timeout=None):
        if future in state.futures:
            state.release.set()
        return original_result(future, timeout)

    def initialized():
        state.threads.append(threading.current_thread())

    def executor(*args, **kwargs):
        pool = ThreadPoolExecutor(*args, **kwargs, initializer=initialized)
        pools.append(pool)

        def submit(function, *arguments, **keywords):
            future = pool.submit(function, *arguments, **keywords)
            if not isinstance(getattr(function, "__self__", None), threading.Barrier):
                state.futures.add(future)
                state.references.append(weakref.ref(future))
                if state.immediate:
                    _finished, pending = wait([future], timeout=2)
                    assert not pending, "fixture request did not finish before admission"
            return future

        return SimpleNamespace(submit=submit, shutdown=pool.shutdown)

    class Response:
        status = 200
        will_close = True

        @staticmethod
        def read():
            return b"{}"

        @staticmethod
        def close():
            if state.response_failure is not None:
                raise state.response_failure

    class Connection:
        sock = None

        def __init__(self, *_args, **_kwargs):
            self.closed = False
            state.clients.append(self)

        def request(self, _method, _path, *, body, headers):
            assert body == b"{}"
            self.closed = False
            state.sent.append(headers["X-Req-Id"])

        @staticmethod
        def getresponse():
            assert state.release.wait(5), "fixture response was never released"
            return Response()

        def close(self):
            self.closed = True

    monkeypatch.setattr(Future, "done", observed_done)
    monkeypatch.setattr(Future, "result", observed_result)
    monkeypatch.setattr("acprof.host.measurement.load.ThreadPoolExecutor", executor)
    monkeypatch.setattr("acprof.host.measurement.load_transport.http.client.HTTPConnection", Connection)
    try:
        yield state
    finally:
        state.release.set()
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)


def assert_resources_closed(state):
    # Check before fixture fallback cleanup can mask missing product cleanup.
    assert state.threads
    assert all(not thread.is_alive() for thread in state.threads)
    assert all(client.closed for client in state.clients)


@pytest.mark.parametrize("max_pending", [1, 16, 64])
def test_full_arrival_queue_keeps_accounting_with_linear_completion_checks(monkeypatch, max_pending):
    config = LoadConfig(scenario="arrival-rate", requests=512, rate=1e9, max_pending=max_pending)
    identity = config.identity()
    with controlled_load(monkeypatch) as state:
        report = run_load("http://fixture.invalid", {}, config, token="burst")
        assert_resources_closed(state)
        assert report["counts"] == {
            "offered": 512, "sent": max_pending, "succeeded": max_pending,
            "failed": 0, "dropped": 512 - max_pending, "cancelled": 0,
        }
        assert report["successful"] is False
        assert state.sent == [f"burst:{index}" for index in range(max_pending)]
        rows = report["requests"]
        assert [row["request_id"] for row in rows] == [f"burst:{index}" for index in range(512)]
        assert [row["planned_s"] for row in rows] == [index / 1e9 for index in range(512)]
        assert all(row["status"] == "ok" for row in rows[:max_pending])
        assert all(row["status"] == "dropped" and row["sent_s"] is None for row in rows[max_pending:])
        assert report["protocol_sha256"] == identity == config.identity()
        assert state.done_checks <= config.requests + max_pending


def test_completed_before_callback_registration_releases_arrival_capacity(monkeypatch):
    config = LoadConfig(scenario="arrival-rate", requests=8, rate=1e9, max_pending=1)
    with controlled_load(monkeypatch) as state:
        state.immediate = True
        state.release.set()
        report = run_load("http://fixture.invalid", {}, config, token="immediate")
        assert_resources_closed(state)
        assert report["successful"]
        assert report["counts"] == {
            "offered": 8, "sent": 8, "succeeded": 8, "failed": 0, "dropped": 0, "cancelled": 0,
        }
        assert state.sent == [f"immediate:{index}" for index in range(8)]


def test_finished_arrival_futures_do_not_wait_for_cyclic_gc(monkeypatch):
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        with controlled_load(monkeypatch) as state:
            report = run_load("http://fixture.invalid", {}, LoadConfig(
                scenario="arrival-rate", requests=128, rate=1e9, max_pending=16,
            ), token="lifetime")
            assert_resources_closed(state)
            assert report["counts"]["succeeded"] == len(state.references) == 16
            assert all(reference() is None for reference in state.references)
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_arrival_completion_preserves_future_failure_and_partial_report(monkeypatch, failure_type):
    failure = failure_type("fixture response finalization failed")
    with controlled_load(monkeypatch) as state:
        state.response_failure = failure
        state.immediate = True
        state.release.set()
        with pytest.raises(failure_type, match="fixture response finalization failed") as caught:
            run_load("http://fixture.invalid", {}, LoadConfig(
                scenario="arrival-rate", requests=8, rate=1e9, max_pending=1,
            ), token="failure")
        assert caught.value is failure
        assert_resources_closed(state)
        report = getattr(failure, "load_report")
        assert report["successful"] is False
        assert report["counts"] == {
            "offered": 8, "sent": 1, "succeeded": 1, "failed": 0, "dropped": 0, "cancelled": 7,
        }
        assert state.sent == ["failure:0"]
        assert [row["request_id"] for row in report["requests"]] == [f"failure:{index}" for index in range(8)]


def test_arrival_wait_cancellation_joins_workers_and_accounts_for_unsubmitted_requests(monkeypatch, caplog):
    failure = KeyboardInterrupt("fixture arrival wait interrupted")
    with controlled_load(monkeypatch) as state:
        def interrupt_sleep(_seconds):
            state.release.set()
            raise failure

        monkeypatch.setattr("acprof.host.measurement.load.time.sleep", interrupt_sleep)
        with pytest.raises(KeyboardInterrupt, match="fixture arrival wait interrupted") as caught:
            run_load("http://fixture.invalid", {}, LoadConfig(
                scenario="arrival-rate", requests=8, rate=1, max_pending=1,
            ), token="cancelled")
        assert caught.value is failure
        assert_resources_closed(state)
        report = getattr(failure, "load_report")
        assert report["successful"] is False
        assert report["counts"]["offered"] == 8
        assert report["counts"]["sent"] <= 1
        assert report["counts"]["failed"] == report["counts"]["dropped"] == 0
        assert report["counts"]["cancelled"] >= 7
        assert report["counts"]["succeeded"] + report["counts"]["cancelled"] == 8
        assert [row["planned_s"] for row in report["requests"]] == list(range(8))
        assert "exception calling callback" not in caplog.text
