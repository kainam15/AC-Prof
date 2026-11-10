"""真实本地 HTTP 服务的完成边界；可控异步替身，不宣称 GPU 验证。"""
import importlib.util
import json
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from functools import partial
from threading import Event, Thread
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest

pytestmark = pytest.mark.runtime


@pytest.mark.skipif(not (importlib.util.find_spec('flask')), reason='requires Flask in the runtime container')
class TestONNXServerCompletion:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        monkeypatch.setenv('ACPROF_REQUEST_TIMEOUT_S', '0.2')
    def start_server(self, future):
        from werkzeug.serving import make_server

        from acprof.container.handlers import BaseHandler
        from acprof.container.model_session import ModelLoadConfig, ModelSession
        from acprof.container.server import create_app

        events, entered = [], Event()

        class AsyncHandler(BaseHandler):
            def load(self, *args, adapter="family-default"):
                return {'task_type': 'tabular-regression', 'device': 'cpu'}

            def preprocess(self, context, payload):
                return {'_effective_input_scale': 1}

            def predict(self, context, processed):
                events.append('submitted')
                return future

            def postprocess(self, context, output):
                if isinstance(output, Future):
                    raise AssertionError('unfinished output reached postprocess')
                events.append('postprocessed')
                return {'task': 'tabular-regression', 'output_type': 'regression',
                        'output_shape': [1, 1], 'n_results': 1}

            def validate_output(self, *args):
                events.append('validated_inside_request')
                raise AssertionError('formal HTTP request must not invoke output validation')

        def wait_for_completion(context, output, *, timeout_s):
            entered.set()
            value = output.result(timeout=timeout_s)
            events.append('completed')
            return value

        runtime = SimpleNamespace(inference_context=nullcontext, metadata=lambda: {},
                                  wait_for_completion=wait_for_completion)
        config = ModelLoadConfig(
            model_id='local/async-fixture', model_revision='fixture',
            task_family='structured', task_type='tabular-regression', backend='onnxruntime',
        )
        session = ModelSession(config, AsyncHandler(),
                               {'task_type': config.task_type, 'device': 'cpu'}, runtime, 'cpu')
        self.server_app = create_app(session)
        server = make_server('127.0.0.1', 0, self.server_app, threaded=True)
        thread = Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
        thread.start()

        def cleanup():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self._request.addfinalizer(partial(cleanup))
        base = f'http://127.0.0.1:{server.server_port}'
        with build_opener(ProxyHandler({})).open(base + '/ready', timeout=2) as response:
            assert (json.load(response)['status']) == ('ok')
        return base, events, entered

    @staticmethod
    def request(base):
        payload = {'input_scale': 1, 'batch_size': 1, 'features': [[1., 2.]]}
        request = Request(base + '/predict', data=json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with build_opener(ProxyHandler({})).open(request, timeout=3) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            with error:
                return error.code, json.load(error)

    def test_oversized_json_is_rejected_before_preprocess_or_inference(self):
        _base, events, _entered = self.start_server(Future())
        limit = self.server_app.config['MAX_CONTENT_LENGTH']
        assert limit == 32 * 1024 * 1024
        with self.server_app.test_client() as client:
            response = client.post('/predict', data=b'{' + b' ' * limit + b'}',
                                   content_type='application/json')
        assert response.status_code == 413
        assert response.is_json
        assert response.json['limit_bytes'] == limit
        assert 'exceeds' in response.json['error']
        assert events == []

    def test_http_waits_for_completion_and_does_not_validate_inside_request(self):
        future = Future()
        base, events, entered = self.start_server(future)
        with ThreadPoolExecutor(max_workers=1) as executor:
            request = executor.submit(self.request, base)
            try:
                assert (entered.wait(1)), 'server never entered completion hook'
                assert not (request.done())
                assert (events) == (['submitted'])
            finally:
                future.set_result([[3.]])
            status, body = request.result(timeout=2)
        assert (status) == (200), body
        assert (body['output_shape']) == ([1, 1])
        assert (body['workload_contract']['input']['rows']) == (1)
        assert (events) == (['submitted', 'completed', 'postprocessed'])

    def test_background_error_becomes_failed_http_response(self):
        future = Future()
        future.set_exception(RuntimeError('background execution failed'))
        base, events, _ = self.start_server(future)
        status, body = self.request(base)
        assert (status) == (500)
        assert ('background execution failed') in (body['error'])
        assert (body['failure']['reason_code']) == ('inference_failed')
        assert (body['failure']['stage']) == ('completion')
        assert ('workload_contract') not in (body)
        assert (events) == (['submitted'])

    def test_timeout_never_returns_success_or_runs_postprocess(self):
        base, events, entered = self.start_server(Future())
        status, body = self.request(base)
        assert (entered.is_set())
        assert (status) == (500)
        assert (body['error']), 'timeout needs a diagnostic instead of an empty error'
        assert (body['failure']['reason_code']) == ('request_timeout')
        assert (body['failure']['evidence']['timeout_seconds']) == (0.2)
        assert (body['failure']['evidence']['service_alive']) is (True)
        assert ('workload_contract') not in (body)
        assert (events) == (['submitted'])
