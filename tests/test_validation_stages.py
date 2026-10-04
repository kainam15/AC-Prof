"""Independent validation identifies failure stages without producing measurements."""
import io
import json
import os
import tempfile
from contextlib import ExitStack, nullcontext, redirect_stdout
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from acprof.container import runtime_validate


class TestValidationStage:
    def fixtures(self, stack, *, failed=None):
        handlers = import_module('acprof.container.handlers')
        execution_module = import_module('acprof.container.execution')
        handler = Mock()
        handler.load.return_value = {}
        handler.preprocess.return_value = {"_effective_input_scale": 1}
        handler.predict.return_value = [1.0]
        handler.postprocess.return_value = {"task": "text-classification"}
        handler.validate_output.return_value = {"protocol": {"status": "verified"},
                                                "task": {"status": "verified"}, "workload_contract": {}}
        if failed:
            getattr(handler, failed).side_effect = ValueError("bad input contract")
        execution = SimpleNamespace(inference_context=nullcontext, metadata=lambda: {})
        stack.enter_context(patch.dict(os.environ, {"TASK_FAMILY": "nlp", "TASK_TYPE": "text-classification",
                                                   "RUNTIME_BACKEND": "transformers_pipeline", "MODEL_ID": "fixture",
                                                   "MODEL_REVISION": "a" * 40, "USE_GPU": "0"}))
        stack.enter_context(patch.object(handlers.HandlerRegistry, "get", return_value=handler))
        stack.enter_context(patch.object(handlers, "load_handler", side_effect=lambda *args: handler.load()))
        stack.enter_context(patch.object(execution_module, "configured_execution", return_value=(execution, "cpu")))
        stack.enter_context(patch.object(execution_module, "complete_prediction", side_effect=lambda e, c, output: output))
        return handler

    def test_success_records_each_phase_separately(self):
        with ExitStack() as stack:
            self.fixtures(stack)
            result = runtime_validate.validate({"text": "hello"})
        assert (result["status"]) == ("ok")
        assert ([item["stage"] for item in result["stages"]]) == (["execution", "load", "preprocess", "predict", "completion", "postprocess", "validate_output", "metadata"])
        assert (all(item["status"] == "verified" for item in result["stages"]))

    def test_preprocess_failure_is_reported_without_running_prediction(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            payload = Path(directory) / "input.json"
            payload.write_text('{"text":"hello"}')
            stack.enter_context(patch("sys.argv", ["runtime_validate", str(payload)]))
            handler = self.fixtures(stack, failed="preprocess")
            output = io.StringIO()
            stack.enter_context(redirect_stdout(output))
            stack.enter_context(patch("acprof.container.runtime_validate.traceback.print_exc"))
            assert (runtime_validate.main()) == (1)
            record = json.loads(output.getvalue().split(runtime_validate.RESULT_PREFIX)[1])
            handler.predict.assert_not_called()
        assert (record["failed_stage"]) == ("preprocess")
        assert (record["stages"][-1]["status"]) == ("error")
        assert ("ValueError: bad input contract") in (record["error"])

    @pytest.mark.parametrize('stale_parent', (False, True))
    def test_completion_timeout_records_actual_request_budget(self, stale_parent, monkeypatch):
        if stale_parent:
            # Import-isolation tests can restore sys.modules while retaining an
            # older module on its package; Python 3.10 dotted patch uses that alias.
            execution = import_module('acprof.container.execution')
            stale = ModuleType(execution.__name__)
            stale.__dict__.update(execution.__dict__)
            monkeypatch.setattr(import_module('acprof.container'), 'execution', stale)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            payload = Path(directory) / "input.json"
            payload.write_text('{"text":"hello"}')
            stack.enter_context(patch("sys.argv", ["runtime_validate", str(payload)]))
            handler = self.fixtures(stack)
            stack.enter_context(patch.dict(os.environ, {"ACPROF_REQUEST_TIMEOUT_S": "2.5"}))
            stack.enter_context(patch.object(import_module('acprof.container.execution'), "complete_prediction",
                                            side_effect=TimeoutError("fixture timeout")))
            stack.enter_context(patch("acprof.container.runtime_validate.traceback.print_exc"))
            output = stack.enter_context(redirect_stdout(io.StringIO()))
            assert (runtime_validate.main()) == (1)
            record = json.loads(output.getvalue().split(runtime_validate.RESULT_PREFIX)[1])
            handler.postprocess.assert_not_called()
        assert (record["failure"]["reason_code"]) == ("request_timeout")
        assert (record["failure"]["evidence"]["timeout_seconds"]) == (2.5)
        assert (record["failure"]["evidence"]["request_phase"]) == ("completion")
        assert (record["failure"]["evidence"]["model_loaded"]) is (True)
