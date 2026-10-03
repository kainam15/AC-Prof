from __future__ import annotations

import contextlib
import json
import tempfile
import types
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from acprof.container.handlers.structured import StructuredHandler
from acprof.workloads.structured import StructuredWorkloadGenerator

TASKS = ("tabular-classification", "tabular-regression", "reinforcement-learning", "robotics", "graph-ml")


@pytest.mark.parametrize('task,key,width', (('tabular-classification', 'features', 8), ('tabular-regression', 'features', 8), ('reinforcement-learning', 'observations', 4), ('robotics', 'observations', 7)))
def test_dense_scale_counts_rows_and_keeps_feature_width_fixed(task, key, width):
    generator = StructuredWorkloadGenerator("demo", task, 2)
    small, large = generator.generate(2), generator.generate(3)
    assert (len(small[key])) == (4)
    assert (len(large[key])) == (6)
    assert ({len(row) for row in large[key]}) == ({width})
    assert (small) == (generator.generate(2))
    assert (small[key][:2]) == (large[key][:2])
    assert (generator.effective_input_scale(100, small)) == (2)
    assert (generator.input_metadata(2, small)["input_num_samples"]) == (4)

def test_graph_scale_is_per_graph_nodes_with_valid_ring_edges():
    generator = StructuredWorkloadGenerator("demo", "graph-ml", 2)
    payload = generator.generate(5)
    assert (len(payload["graphs"])) == (2)
    for graph in payload["graphs"]:
        assert (len(graph["node_features"])) == (5)
        assert (len(graph["edge_index"][0])) == (10)
        assert (max(max(row) for row in graph["edge_index"])) == (4)
    assert (generator.effective_input_scale(100, payload)) == (5)
    metadata = generator.input_metadata(5, payload)
    assert (metadata["node_count"]) == (10)
    assert (metadata["edge_count"]) == (20)
    assert (metadata["input_num_samples"]) == (2)

@pytest.mark.parametrize('value_case', range(7), ids=['0', '-1', '1.5', 'True', "float('nan')", "float('inf')", '4097'])
def test_invalid_scales_fail_instead_of_silent_rounding_or_truncation(value_case):
    generator = StructuredWorkloadGenerator("demo", "tabular-classification", 1)
    value = tuple((0, -1, 1.5, True, float('nan'), float('inf'), 4097))[value_case]
    with pytest.raises(ValueError):
        generator.generate(value)

def test_spec_controls_shape_scales_seed_and_records_provenance():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "workload.json"
        path.write_text(json.dumps({"schema_version": 1, "task": "robotics", "feature_dim": 12,
                                    "input_scales": [2, 4], "seed": 42,
                                    "provenance": {"source": "test"}}))
        generator = StructuredWorkloadGenerator("demo", "robotics", 1, workload_spec_path=str(path))
        assert (len(generator.generate(2)["observations"][0])) == (12)
        assert (generator.default_input_scales()) == ([2.0, 4.0])
        assert (generator.plan_metadata()["feature_dim"]) == (12)
        assert (generator.plan_metadata()["seed"]) == (42)
        assert (generator.plan_metadata()["provenance"]) == ({"source": "test"})
        assert (len(generator.plan_metadata()["workload_spec_sha256"])) == (64)


class TestStructuredHandler:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.handler = StructuredHandler()
        self.fake_torch = types.SimpleNamespace(
            tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
            float32=np.float32, int64=np.int64, inference_mode=contextlib.nullcontext,
        )

    def context(self, task="tabular-classification", backend="torchscript", feature_dim=2):
        return {"task_type": task, "backend": backend, "tensor_inputs": backend == "torchscript", "feature_dim": feature_dim,
                "device": "cpu", "model": Mock(return_value=np.ones((2, 1), np.float32))}

    @pytest.mark.parametrize('payload_case', range(5), ids=["{'features': [[1]], 'input_scale': 1}", "{'features': [[1, 2]], 'input_scale': 2}", "{'features': [[1, float('nan')]], 'input_scale': 1}", "{'features': [[True, 2]], 'input_scale': 1}", "{'features': [[1, 2]], 'batch_size': 2, 'input_scale': 1}"])
    def test_preprocess_validates_actual_scale_and_dimension(self, payload_case):
        ctx = self.context()
        with patch.dict("sys.modules", {"torch": self.fake_torch}):
            processed = self.handler.preprocess(ctx, {"features": [[1, 2], [3, 4]],
                                                       "batch_size": 1, "input_scale": 2})
            assert (processed["_effective_input_scale"]) == (2)
            np.testing.assert_array_equal(processed["args"][0], [[1, 2], [3, 4]])
            payload = tuple(({'features': [[1]], 'input_scale': 1}, {'features': [[1, 2]], 'input_scale': 2}, {'features': [[1, float('nan')]], 'input_scale': 1}, {'features': [[True, 2]], 'input_scale': 1}, {'features': [[1, 2]], 'batch_size': 2, 'input_scale': 1}))[payload_case]
            with pytest.raises(ValueError):
                self.handler.preprocess(ctx, payload)

    def test_graph_preprocess_forms_disjoint_batch_and_offsets_edges(self):
        ctx = self.context("graph-ml")
        graph = {"node_features": [[1, 2], [3, 4]], "edge_index": [[0, 1], [1, 0]]}
        with patch.dict("sys.modules", {"torch": self.fake_torch}):
            processed = self.handler.preprocess(ctx, {"graphs": [graph, graph], "batch_size": 2, "input_scale": 2})
        assert (processed["_effective_input_scale"]) == (2)
        np.testing.assert_array_equal(processed["args"][1], [[0, 1, 2, 3], [1, 0, 3, 2]])
        np.testing.assert_array_equal(processed["args"][2], [0, 0, 1, 1])

    @pytest.mark.parametrize('edges', ([[0], [2]], [[-1], [0]], [[0.5], [1]], [[True], [1]], [[0], []]))
    def test_graph_rejects_invalid_indices_and_mixed_node_counts(self, edges):
        ctx = self.context("graph-ml")
        with patch.dict("sys.modules", {"torch": self.fake_torch}), pytest.raises(ValueError):
            self.handler.preprocess(ctx, {"graphs": [{"node_features": [[1, 2], [3, 4]], "edge_index": edges}],
                                          "input_scale": 2})

    def test_predict_uses_preprocessed_inputs_and_disables_gradients(self):
        ctx = self.context()
        tensor = np.array([[1, 2]], np.float32)
        self.fake_torch.tensor = Mock(side_effect=AssertionError("predict rebuilt the input"))
        self.fake_torch.inference_mode = Mock(return_value=contextlib.nullcontext())
        with patch.dict("sys.modules", {"torch": self.fake_torch}):
            output = self.handler.predict(ctx, {"args": (tensor,)})
        ctx["model"].assert_called_once_with(tensor)
        self.fake_torch.inference_mode.assert_called_once_with()
        assert (output.shape) == ((2, 1))

    @pytest.mark.parametrize('task_case', range(5))
    def test_postprocess_reports_tensor_results_for_each_task(self, task_case):
        task = tuple(TASKS)[task_case]
        result = self.handler.postprocess(self.context(task), np.ones((3, 2)))
        assert (result["task"]) == (task)
        assert (result["output_shape"]) == ([3, 2])
        assert (result["n_results"]) == (3)
        with pytest.raises(ValueError):
            self.handler.postprocess(self.context(), {"loss": 0})

    def test_torchscript_loads_only_declared_local_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "model.pt").write_bytes(b"test")
            manifest = {"schema_version": 1, "task": "robotics", "format": "torchscript",
                        "model_file": "model.pt", "feature_dim": 7}
            (path / "acprof_model.json").write_text(json.dumps(manifest))
            model = Mock()
            model.eval.return_value = model
            self.fake_torch.jit = types.SimpleNamespace(load=Mock(return_value=model))
            with patch.dict("sys.modules", {"torch": self.fake_torch}):
                ctx = self.handler.load(directory, "robotics", "torchscript", "cpu")
                assert (ctx["feature_dim"]) == (7)
                self.fake_torch.jit.load.assert_called_once_with(str(path / "model.pt"), map_location="cpu")
                with pytest.raises(ValueError, match="task"):
                    self.handler.load(directory, "graph-ml", "torchscript", "cpu")
                manifest["model_file"] = "../model.pt"
                (path / "acprof_model.json").write_text(json.dumps(manifest))
                with pytest.raises(ValueError, match="model_file"):
                    self.handler.load(directory, "robotics", "torchscript", "cpu")

    def test_missing_manifest_and_unknown_backend_fail_with_format_guidance(self):
        with tempfile.TemporaryDirectory() as directory:
            with pytest.raises(ValueError, match="acprof_model.json"):
                self.handler.load(directory, "robotics", "torchscript", "cpu")
            with pytest.raises(ValueError, match="backend"):
                self.handler.load(directory, "robotics", "transformers_pipeline", "cpu")
            with pytest.raises(ValueError, match="CPU"):
                self.handler.load(directory, "tabular-classification", "skops", "cuda:0")

    def test_skops_loads_single_artifact_without_trusting_custom_types(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.skops"
            path.write_bytes(b"test")
            estimator = Mock(n_features_in_=2)
            loader = Mock(return_value=estimator)
            modules = {"skops": types.ModuleType("skops"), "skops.io": types.SimpleNamespace(load=loader),
                       "sklearn": types.SimpleNamespace(base=types.SimpleNamespace(
                           is_classifier=lambda model: True, is_regressor=lambda model: False))}
            with patch.dict("sys.modules", modules):
                ctx = self.handler.load(directory, "tabular-classification", "skops", "cpu")
            loader.assert_called_once_with(str(path), trusted=[])
            assert (ctx["feature_dim"]) == (2)
            processed = self.handler.preprocess(ctx, {"features": [[1, 2]], "input_scale": 1})
            self.handler.predict(ctx, processed)
            np.testing.assert_array_equal(estimator.predict.call_args.args[0], [[1, 2]])
