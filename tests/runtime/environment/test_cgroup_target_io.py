"""The cgroup collecting a workload, not the launcher, owns I/O availability."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from acprof.experiments.configuration import RunConfig
from acprof.host.measurement.quality import window_quality
from acprof.host.measurement.requests import WindowRequestProgress
from acprof.host.runtimes.capabilities import probe_collection_plan
from acprof.monitors import resource_readers
from acprof.platform import Environment
from acprof.results.profile import metric_observation
from acprof.tui.i18n import translate
from acprof.tui.run.diagnostics import _automatic_checks
from acprof.tui.run.environment_view import check_presentation, global_check_view
from acprof.tui.run.global_status import build_status


@pytest.mark.parametrize(("runtime", "service_cgroup", "prefix"), [
    ("docker", None, "container_io_"),
    ("process", True, "service_cgroup_io_"),
])
def test_collection_plan_never_uses_host_cgroup_to_deny_target_io(
    tmp_path, monkeypatch, runtime, service_cgroup, prefix,
):
    # Host cgroup is readable but has no io.stat; a future workload may.
    (tmp_path / "cpu.stat").write_text("usage_usec 1\n")
    monkeypatch.setattr(
        "acprof.host.runtimes.resources.inherited_resources",
        lambda: {"cgroup": str(tmp_path), "cgroup_error": ""},
    )
    plan = probe_collection_plan(
        runtime, environment=Environment("wsl2", "Linux"), service_cgroup=service_cgroup,
    )
    assert plan["probes"]["cgroup"]["status"] == "available"
    assert any(field.startswith(prefix) for field in plan["fields"]["cgroup"])
    assert not any(field.startswith(prefix) for field in plan["field_probes"])


def test_pending_target_io_is_neutral_and_does_not_block_start(tmp_path, monkeypatch):
    plan = {"probes": {"cgroup": {"status": "available", "detail": "host v2 visible"}},
            "fields": {"cgroup": [
                "container_io_read_bytes_per_request", "container_io_write_bytes_per_request",
                "container_io_read_ops_per_request", "container_io_write_ops_per_request",
            ]}, "field_probes": {}}
    monkeypatch.setattr(
        "acprof.host.runtimes.capabilities.probe_collection_plan", lambda *a, **kw: plan,
    )
    config = replace(RunConfig.smoke("fixture/model"), runtime="docker", profiling_mode="best_effort")
    row, = _automatic_checks(config, "docker", None, project_dir=tmp_path)
    assert row.check_id == "cgroup_metrics"
    assert row.status == "declared"
    assert row.reason_code == "runtime_pending"
    assert len(row.affected_fields) == 4
    status = build_status(config, issues=(), validation_pending=False, checks=(row,), inventory=(),
                          completed=True, current=True, running=False, inventory_running=False)
    item, = (entry for entry in status.items if entry.check.check_id == "cgroup_metrics")
    assert item.requested is True
    assert not item.blocks
    assert status.can_start
    assert status.counts["warning"] == 0
    assert status.counts["pending"] >= 1
    assert translate(check_presentation(item.check).label, "zh") == "待 Runtime 验证"
    view = global_check_view(item, technical_detail=row.detail,
                             tr=lambda value: translate(value, "zh"),
                             check_allowed=True, navigate_allowed=True)
    assert view.facts is not None
    assert "目标容器" in translate(view.facts.rows[1].value, "zh")


@pytest.mark.parametrize(("host_has_io", "target_has_io"), [(False, True), (True, False)])
def test_readers_follow_target_container_not_host(
    tmp_path, monkeypatch, host_has_io, target_has_io,
):
    proc_root = tmp_path / "proc"
    host_cgroup = tmp_path / "cgroup"
    target = host_cgroup / "docker" / "case"
    target.mkdir(parents=True)
    (proc_root / "42").mkdir(parents=True)
    (proc_root / "42" / "cgroup").write_text("0::/docker/case\n")
    (target / "cpu.stat").write_text("usage_usec 1000\n")
    if host_has_io:
        (host_cgroup / "io.stat").write_text("8:0 rbytes=9 wbytes=7 rios=1 wios=1\n")
    if target_has_io:
        (target / "io.stat").write_text("8:0 rbytes=100 wbytes=200 rios=2 wios=3\n")
    monkeypatch.setattr(resource_readers, "docker_container_pid", lambda _name: 42)

    readers = resource_readers.resolve_container_metric_readers(
        "case", cgroup_root=str(host_cgroup), proc_root=str(proc_root),
    )
    assert readers.cpu is not None
    if target_has_io:
        assert readers.io is not None
        assert readers.io() == (100, 200)
        assert readers.io_operations is not None
        assert readers.io_operations() == {"read_ops": 2.0, "write_ops": 3.0}
        assert not readers.reader_errors
    else:
        assert readers.io is None
        assert readers.io_operations is None
        assert "io.stat" in readers.reader_errors["io"]
        assert "io.stat" in readers.reader_errors["io_operations"]


def test_target_io_denied_is_distinguished_from_missing(tmp_path, monkeypatch):
    path = tmp_path / "io.stat"
    path.write_text("8:0 rbytes=1 wbytes=2 rios=3 wios=4\n")
    real_stat = resource_readers.os.stat

    def denied(target, *args, **kwargs):
        if str(target) == str(path):
            raise PermissionError(13, "Permission denied", str(path))
        return real_stat(target, *args, **kwargs)

    monkeypatch.setattr(resource_readers.os, "stat", denied)
    readers = resource_readers.resolve_cgroup_metric_readers(str(tmp_path))
    assert readers.io is None
    assert "PermissionError" in readers.reader_errors["io"]
    assert "Permission denied" in readers.reader_errors["io_operations"]


@pytest.mark.parametrize("content", ["", "8:0 rbytes=0 wbytes=0\n"])
def test_empty_io_stat_is_not_a_measured_zero(tmp_path, content):
    path = tmp_path / "io.stat"
    path.write_text(content)
    if not content:
        with pytest.raises(ValueError, match="io.stat missing counters"):
            resource_readers._read_cgroup_v2_io_stat(str(path))
        with pytest.raises(ValueError, match="io.stat missing counters"):
            resource_readers._read_cgroup_v2_io_operations(str(path))
    else:
        assert resource_readers._read_cgroup_v2_io_stat(str(path)) == (0, 0)
        with pytest.raises(ValueError, match="io.stat missing counters"):
            resource_readers._read_cgroup_v2_io_operations(str(path))


@pytest.mark.parametrize(("has_value", "has_runtime_error", "expected"), [
    (True, False, "complete"), (False, True, "unavailable"),
])
def test_post_run_quality_ignores_old_host_cgroup_advisory(
    has_value, has_runtime_error, expected,
):
    field = "container_io_read_bytes_per_request"
    legacy = {"status": "unsupported", "detail": "host io.stat absent", "scope": "preflight_namespace"}
    config = SimpleNamespace(
        collection_plan={"probes": {"cgroup": {"status": "available", "detail": ""}},
                         "fields": {"cgroup": [field]}, "field_probes": {field: legacy}},
        profiling_mode="best_effort", sample_hz=20, sampling_policy="burst",
        task_family="nlp", idle_debug=False,
    )
    resource = {"status": "recorded"}
    if has_runtime_error:
        resource["reader_errors"] = {"io": "target cgroup io.stat: FileNotFoundError: No such file"}
    observations = SimpleNamespace(
        resource_result=SimpleNamespace(metric_evidence={}),
        collector_evidence={"resource": resource}, cpu_result=None,
    )
    progress = WindowRequestProgress(count=1, payload_sizes=[4], response_sizes=[8])
    evidence = window_quality(
        config, progress, observations, use_gpu=False, status="ok", error="", latency_slo=False,
    )
    assert field in evidence["collectors"]["cgroup"]["preflight_advisories"]
    assert "preflight_missing" not in evidence["collectors"]["cgroup"]
    row = {"runtime_kind": "docker", "resource_metric_quality": json.dumps(evidence),
           field: 128.0 if has_value else float("nan")}
    result = metric_observation(row, field)
    assert result["quality"] == expected
    assert ("target cgroup" in result["reason"]) == has_runtime_error


def test_missing_target_reader_reaches_resource_telemetry(tmp_path, monkeypatch):
    from acprof.monitors.resource_usage import ResourceUsageMonitor

    (tmp_path / "cpu.stat").write_text("usage_usec 1000\n")
    readers = resource_readers.resolve_cgroup_metric_readers(str(tmp_path))
    monkeypatch.setattr(
        resource_readers, "resolve_container_metric_readers", lambda *a, **kw: readers,
    )
    monitor = ResourceUsageMonitor(container_name="case")
    try:
        evidence = monitor.telemetry_snapshot()
    finally:
        monitor.close()
    assert "io.stat" in evidence["reader_errors"]["io"]
    assert "io.stat" in evidence["reader_errors"]["io_operations"]
    assert "io" not in evidence["reader_errors"].get("cpu", "")
