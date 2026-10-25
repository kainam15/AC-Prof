"""Metric declarations also drive the pure row assembly consumers."""
import subprocess
import sys
from dataclasses import replace
from textwrap import dedent

from acprof.metric_registry import METRICS


def test_row_assembly_roles_do_not_change_persisted_metric_metadata():
    metric = METRICS["cpu_idle_power_w"]
    assert metric.row_group == "cpu_energy"
    assert metric.to_dict() == replace(metric, row_group="", row_order=None).to_dict()
    assert "row_group" not in metric.to_dict()
    assert "row_order" not in metric.to_dict()


def test_new_grouped_metric_is_initialized_by_row_consumer():
    # Import in isolation so adding a future declaration cannot leak into other
    # tests or require reloading the production measurement modules in-process.
    program = dedent("""
        import math
        from types import MappingProxyType, SimpleNamespace
        from acprof import metric_registry

        new_metric = metric_registry.Metric(
            "future_resource_metric", "count", "cgroup", "request_window",
            row_group="resource_usage",
        )
        metric_registry.METRICS = MappingProxyType({
            **metric_registry.METRICS, new_metric.name: new_metric,
        })

        from acprof.host.client_metrics import WindowMetrics

        window = WindowMetrics(
            SimpleNamespace(batch_size=1), {"input_scale": 1}, use_energy=False,
        )
        assert math.isnan(window.resource["future_resource_metric"])
        assert list(window.resource)[-1] == "future_resource_metric"
    """)
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
