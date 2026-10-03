import contextlib
import types
from unittest.mock import Mock, patch

import numpy as np
import pytest

from acprof.container.handlers.timeseries import ChronosHandler, TimeseriesTransformersHandler
from acprof.workloads.timeseries import TimeseriesWorkloadGenerator


class Tensor:
    def __init__(self, value):
        self.values = np.asarray(value)
        self.ndim = self.values.ndim
        self.shape = self.values.shape
        self.device = "cpu"

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.values, axis))

    def __getitem__(self, index):
        item = Tensor(self.values[index])
        item.device = self.device
        return item

    def to(self, device):
        self.device = device
        return self


def test_chronos_exposes_context_limit_and_rejects_hidden_truncation():
    handler = ChronosHandler()
    ctx = {"device": "cpu", "pipeline": types.SimpleNamespace(model_context_length=2)}
    metadata = handler.get_scale_metadata(ctx, {})
    assert (metadata["max_effective_input_scale"]) == (2)
    assert (metadata["input_scale_type"]) == ("context_length")
    torch = types.SimpleNamespace(tensor=lambda value, **kwargs: Tensor(value), float32="float32")
    with patch.dict("sys.modules", {"torch": torch}), pytest.raises(ValueError, match="context.*2"):
        handler.preprocess(ctx, {"context": [[1, 2, 3]]})

def test_chronos_preserves_cpu_series_for_native_pinning_and_device_transfer():
    torch = types.SimpleNamespace(tensor=lambda value, **kwargs: Tensor(value), float32="float32",
                                  inference_mode=contextlib.nullcontext)
    ctx = {"device": "cuda", "task_type": "time-series-forecasting", "pipeline": Mock()}
    handler = ChronosHandler()
    with patch.dict("sys.modules", {"torch": torch}):
        processed = handler.preprocess(ctx, {"context": [[1.0, 2.0, 3.0]], "prediction_length": 2})
        assert (processed["context"].device) == ("cpu")
        assert (all(series.device == "cpu" for series in processed["series"]))
        assert (processed["_effective_input_scale"]) == (3)
        processed["context"].to = Mock(side_effect=AssertionError("predict moved input"))
        handler.predict(ctx, processed)
        ctx["pipeline"].predict.assert_called_once_with(processed["series"], prediction_length=2)

@pytest.mark.parametrize('payload_case', range(6), ids=["{'context': []}", "{'context': [[1], [1, 2]]}", "{'context': [[float('nan')]]}", "{'context': [[True]]}", "{'context': [[1]], 'prediction_length': 0}", "{'context': [[1]], 'prediction_length': 1.5}"])
def test_invalid_context_and_prediction_length_fail(payload_case):
    handler = ChronosHandler()
    torch = types.SimpleNamespace(tensor=lambda value, **kwargs: Tensor(value), float32="float32")
    ctx = {"device": "cpu", "task_type": "time-series-forecasting", "pipeline": Mock()}
    payload = tuple(({'context': []}, {'context': [[1], [1, 2]]}, {'context': [[float('nan')]]}, {'context': [[True]]}, {'context': [[1]], 'prediction_length': 0}, {'context': [[1]], 'prediction_length': 1.5}))[payload_case]
    with patch.dict("sys.modules", {"torch": torch}), pytest.raises(ValueError):
        handler.preprocess(ctx, payload)

def test_non_chronos_pipeline_fails_with_explicit_support_boundary():
    torch = types.ModuleType("torch")
    transformers = types.SimpleNamespace(pipeline=Mock())
    with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}):
        with pytest.raises(ValueError, match="Chronos"):
            TimeseriesTransformersHandler().load("unsupported/model", "time-series-forecasting", "transformers_pipeline", "cpu")
    transformers.pipeline.assert_not_called()

@pytest.mark.parametrize('value', (0, -1, 1.5, True, 2049))
def test_workload_rejects_invalid_scale_instead_of_slicing(value):
    generator = TimeseriesWorkloadGenerator("demo", "time-series-forecasting", 1)
    with pytest.raises(ValueError):
        generator.generate(value)
