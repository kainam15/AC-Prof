"""Application factories use loaded dependencies; startup keeps its timed phases."""
import importlib.util
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.container import server
from acprof.container.model_session import ModelLoadConfig, ModelSession

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("flask") is None, reason="requires the runtime Flask dependency",
)


class LoadedHandler:
    def load(self, *args, **kwargs):
        raise AssertionError("application construction must not load a model")

    def preprocess(self, context, payload):
        return {"_effective_input_scale": 1, "_probe_reason": context["identity"]}

    def predict(self, context, processed):
        return [[3.0]]

    def postprocess(self, context, output):
        return {"task": "tabular-regression", "output_type": "regression",
                "output_shape": [1, 1], "n_results": 1, "identity": context["identity"]}

    def get_scale_metadata(self, context, payload):
        return {"identity": context["identity"]}

    def validate_output(self, *args):
        raise AssertionError("formal HTTP requests must not run output validation")


def loaded_session(identity, *, use_gpu=False):
    config = ModelLoadConfig(
        identity, "structured", "tabular-regression", "onnxruntime",
        model_revision="fixture-revision", use_gpu=use_gpu,
    )
    context = {"identity": identity, "task_type": config.task_type}
    execution = SimpleNamespace(inference_context=nullcontext, metadata=lambda: {})
    return ModelSession(config, LoadedHandler(), context, execution, "cuda" if use_gpu else "cpu")


def test_apps_keep_loaded_sessions_and_service_identity_separate():
    first, second = loaded_session("first"), loaded_session("second")
    first_app = server.create_app(first, service_token="token-first", startup_timing={"model_load_s": 1.2345})
    second_app = server.create_app(second, service_token="token-second")

    for app, identity, token in [
        (first_app, "first", "token-first"), (second_app, "second", "token-second"),
        (first_app, "first", "token-first"),
    ]:
        with app.test_client() as client:
            ready = client.get("/ready").get_json()
            assert (ready["model_id"], ready["service_token"]) == (identity, token)
            assert client.get("/meta").get_json()["model_id"] == identity
            assert client.post("/probe", json={}).get_json()["reason"] == identity
            assert client.get("/scale_meta").get_json()["identity"] == identity
            result = client.post("/predict", json={"input_scale": 1, "features": [[1.0, 2.0]]})
            assert result.status_code == 200
            assert result.get_json()["identity"] == identity

    assert first_app.test_client().get("/ready").get_json()["load_time_s"] == 1.234
    assert second_app.test_client().get("/ready").get_json()["load_time_s"] is None


def test_probe_keeps_empty_reason_when_preprocessing_has_no_diagnostic():
    session = loaded_session("probe")
    session.handler.preprocess = lambda context, payload: {}
    app = server.create_app(session)

    response = app.test_client().post("/probe", json={})

    assert response.status_code == 200
    assert response.get_json() == {
        "effective_input_scale": None, "truncated_by_limit": None, "reason": "",
    }


@pytest.mark.parametrize(("use_gpu", "expected_setup", "expected_cuda"), [(False, 12.0, 0.0), (True, 5.0, 7.0)])
def test_runtime_startup_keeps_setup_device_and_load_timing_separate(
    monkeypatch, use_gpu, expected_setup, expected_cuda,
):
    import flask

    from acprof.container import model_session

    session = loaded_session("timed", use_gpu=use_gpu)
    elapsed, events = [0.0], []
    monkeypatch.setattr(server, "SERVER_PROCESS_STARTED_AT", 1000.0)
    monkeypatch.setattr(server, "SERVER_PROCESS_STARTED_PERF", 100.0)
    monkeypatch.setattr(server, "time", SimpleNamespace(
        time=lambda: 1000.0 + elapsed[0], perf_counter=lambda: 100.0 + elapsed[0],
    ))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.setenv("ACPROF_SERVICE_TOKEN", "timed-token")
    monkeypatch.setenv("ACPROF_RUNTIME_PROFILE", "fixture-profile")
    monkeypatch.setattr(model_session.ModelLoadConfig, "from_environment", lambda: session.config)

    class TimedFlask(flask.Flask):
        def __init__(self, *args, **kwargs):
            events.append("app")
            elapsed[0] += 2.0
            super().__init__(*args, **kwargs)

    def select(config):
        events.append("handler")
        elapsed[0] += 3.0
        return session.handler

    def prepare(config):
        events.append("device")
        elapsed[0] += 7.0
        return session.execution, session.device

    def load(config, execution, device, **kwargs):
        events.append("load")
        assert kwargs == {"handler": session.handler, "model_source": "timed"}
        elapsed[0] += 11.0
        return session

    monkeypatch.setattr(flask, "Flask", TimedFlask)
    monkeypatch.setattr(model_session, "select_handler", select)
    monkeypatch.setattr(model_session, "prepare_execution", prepare)
    monkeypatch.setattr(model_session, "load_model_session", load)

    app = server.create_runtime_app()
    ready = app.test_client().get("/ready").get_json()

    assert events == ["app", "handler", "device", "load"]
    assert ready["startup_timing"] == {
        "server_process_started_at_epoch_s": 1000.0,
        "server_setup_s": expected_setup, "cuda_init_s": expected_cuda,
        "model_load_started_at_epoch_s": 1012.0,
        "model_load_completed_at_epoch_s": 1023.0, "model_load_s": 11.0,
    }
    assert ready["load_time_s"] == 11.0
    assert ready["service_token"] == "timed-token"
    assert server.os.environ["HF_HUB_OFFLINE"] == "1"
    assert server.os.environ["TRANSFORMERS_OFFLINE"] == "1"


@pytest.mark.parametrize("descriptor", [None, "17"])
def test_main_keeps_standard_and_inherited_socket_launch_paths(monkeypatch, descriptor):
    from werkzeug import serving

    app = Mock()
    initialize = Mock(return_value=app)
    monkeypatch.setattr(server, "create_runtime_app", initialize)
    make_server = Mock()
    close = Mock()
    monkeypatch.setattr(serving, "make_server", make_server)
    monkeypatch.setattr(server.os, "close", close)
    if descriptor is None:
        monkeypatch.delenv("ACPROF_SERVER_FD", raising=False)
    else:
        monkeypatch.setenv("ACPROF_SERVER_FD", descriptor)
        monkeypatch.setenv("ACPROF_SERVER_PORT", "9182")

    server.main()

    initialize.assert_called_once_with()
    if descriptor is None:
        app.run.assert_called_once_with(host="0.0.0.0", port=8002)
        make_server.assert_not_called()
    else:
        make_server.assert_called_once_with("127.0.0.1", 9182, app, threaded=True, fd=17)
        close.assert_called_once_with(17)
        make_server.return_value.serve_forever.assert_called_once_with()
        app.run.assert_not_called()
