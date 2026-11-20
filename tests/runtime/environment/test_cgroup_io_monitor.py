"""Missing target cgroup I/O counters must not abort unrelated measurements."""

import math

import pytest

from acprof.monitors import resource_readers, resource_usage


@pytest.mark.parametrize(
    ("content", "expected_fatal"),
    [
        ("", False),
        ("8:0 rbytes=10 wbytes=20\n", False),
        ("8:0 rbytes=10 wbytes=20 rios=1 wios=2\n", False),
        ("8:0 rbytes=-1 wbytes=20 rios=1 wios=2\n", True),
    ],
)
def test_target_io_unavailable_is_observed_without_abort(tmp_path, monkeypatch, content, expected_fatal):
    path = tmp_path / "io.stat"
    path.write_text(content, encoding="utf-8")
    readers = resource_readers.ResourceMetricReaders(
        cpu=lambda: 0.1,
        memory=lambda: 4096,
        io=lambda: resource_readers._read_cgroup_v2_io_stat(str(path)),
        io_operations=lambda: resource_readers._read_cgroup_v2_io_operations(str(path)),
    )
    monkeypatch.setattr(resource_readers, "resolve_container_metric_readers", lambda *_a, **_kw: readers)
    monkeypatch.setattr(resource_readers, "prepare_cpu_frequency_reader", lambda *_a, **_kw: lambda: (None, None))

    monitor = resource_usage.ResourceUsageMonitor(sample_hz=10, container_name="fixture")
    try:
        monitor.start()
        result, error, _ = monitor.stop()
        diagnostics = monitor.telemetry_snapshot()["reader_errors"]
    finally:
        monitor.close()

    assert result.resource_usage_iters >= 1
    assert result.container_mem_usage_avg_bytes == 4096
    if expected_fatal:
        assert "negative rbytes counter" in error
        assert "negative rbytes counter" in diagnostics["io"]
    else:
        assert error == ""
    if not content:
        assert math.isnan(result.container_io_read_bytes)
        assert math.isnan(result.container_io_write_bytes)
        assert "io.stat missing counters" in diagnostics["io"]
        assert "io.stat missing counters" in diagnostics["io_operations"]
    elif not expected_fatal:
        # Both byte counters were observed: equal start/end is a measured zero.
        assert result.container_io_read_bytes == 0
        assert result.container_io_write_bytes == 0
        if "rios=" not in content:
            assert "io.stat missing counters" in diagnostics["io_operations"]
        else:
            assert "io_operations" not in diagnostics
