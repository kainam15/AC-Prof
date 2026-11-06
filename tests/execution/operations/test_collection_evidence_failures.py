"""CLI failures preserve collection errors, independent evidence and resumability."""
import io
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from acprof.host.measurement import packet_processing
from acprof.host.measurement.capture import PacketLatencyError
from acprof.host.orchestrator import EnergyProfilingError, MIPSProfilingError
from acprof.quality import loading_quality
from acprof.results.layout import ArtifactLayout, case_sidecar
from tests.support.run_recovery_fixtures import RunRecoveryFixture


@pytest.fixture
def collection(request, tmp_path):
    fixture = RunRecoveryFixture()
    fixture.build(request, tmp_path)
    return fixture


@pytest.mark.parametrize("failure_type", [None, RuntimeError, KeyboardInterrupt],
                         ids=["completed-matrix", "failed-matrix", "cancelled-matrix"])
@pytest.mark.parametrize("broken", [("quality_checks",), ("runtime_failures",),
                                   ("quality_checks", "runtime_failures")],
                         ids=["quality", "failures", "both"])
def test_evidence_failures_keep_primary_other_reports_and_resume(collection, failure_type, broken):
    primary = failure_type("fixture matrix failure") if failure_type else None
    quality = loading_quality({"unexpected_keys": ["fixture.unused"]}, source="fixture")
    failures = [{"stage": "postprocess", "reason_code": "inference_failed", "detail": "fixture diagnostic"}]
    payloads = {"quality_checks": {"schema_version": 1, "checks": quality},
                "runtime_failures": {"schema_version": 1, "failures": failures}}
    reports, originals, sidecars, measurements = {}, {}, {}, {}

    def measure(request, **kwargs):
        if request.cpu == 2 and primary is not None:
            raise primary
        case_csv = Path(collection.write_case(request, **kwargs))
        measurements[case_csv] = case_csv.read_bytes()
        if request.cpu == 1:
            case_layout = ArtifactLayout.discover(collection.directory)
            for evidence_kind, payload in payloads.items():
                sidecar = case_sidecar(case_csv, evidence_kind)
                sidecar.write_text("{broken" if evidence_kind in broken else json.dumps(payload))
                sidecars[evidence_kind] = sidecar
                aggregate = case_layout.path(evidence_kind + ".json")
                aggregate.write_text('{"schema_version": 1, "checks": [], "failures": []}\n')
                reports[evidence_kind], originals[evidence_kind] = aggregate, aggregate.read_bytes()
        return str(case_csv)

    errors = io.StringIO()
    with pytest.raises(type(primary) if primary is not None else SystemExit) as caught:
        collection.invoke(case=measure, error_output=errors)

    if primary is not None:
        assert caught.value is primary
        diagnostic = "\n".join((str(caught.value), *caught.value.__notes__))
    else:
        assert isinstance(caught.value, SystemExit)
        assert caught.value.code == 2
        diagnostic = errors.getvalue()
    for kind in broken:
        assert str(sidecars[kind]) in diagnostic
    for kind, report in reports.items():
        if kind in broken:
            assert report.read_bytes() == originals[kind]
        else:
            assert json.loads(report.read_text()) == payloads[kind]
    for path, original in measurements.items():
        assert path.read_bytes() == original
    layout = ArtifactLayout.discover(collection.directory)
    state_path = layout.path("run_state.json")
    previous = json.loads(state_path.read_text())
    assert previous["status"] == ("interrupted" if failure_type is KeyboardInterrupt else "failed")

    # A fresh CLI owner must reacquire the locks and reuse completed case results.
    for kind in broken:
        sidecars[kind].write_text(json.dumps(payloads[kind]))
    collection.calls.clear()
    collection.invoke("--resume")
    assert collection.calls == ([2] if primary is not None else [])
    resumed = json.loads(state_path.read_text())
    assert resumed["status"] == "complete"
    assert resumed["run_id"] == previous["run_id"]
    for name, record in previous["cases"].items():
        if record["status"] == "complete":
            assert resumed["cases"][name]["sha256"] == record["sha256"]


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_evidence_read_cancellation_keeps_prior_failure_and_collects_other_report(collection, failure_type):
    primary = failure_type("fixture matrix failure")
    cancellation = KeyboardInterrupt("fixture evidence read cancelled")
    quality_path: Path | None = None
    original_open = io.open

    def measure(request, **kwargs):
        nonlocal quality_path
        if request.cpu == 2:
            raise primary
        path = Path(collection.write_case(request, **kwargs))
        quality = case_sidecar(path, "quality_checks")
        quality.write_text('{"schema_version":1,"checks":[]}')
        case_sidecar(path, "runtime_failures").write_text('{"schema_version":1,"failures":[{"detail":"retained"}]}')
        quality_path = quality
        return str(path)

    def open_file(file: str | bytes | Path | int, *args, **kwargs):
        if file == quality_path:
            raise cancellation
        return original_open(file, *args, **kwargs)

    with patch("io.open", open_file), pytest.raises(KeyboardInterrupt) as caught:
        collection.invoke(case=measure)
    assert caught.value is (primary if failure_type is KeyboardInterrupt else cancellation)
    diagnostic = "\n".join((str(caught.value), *caught.value.__notes__))
    assert "fixture matrix failure" in diagnostic
    assert "fixture evidence read cancelled" in diagnostic
    if failure_type is RuntimeError:
        assert caught.value.__cause__ is primary
    layout = ArtifactLayout.discover(collection.directory)
    assert json.loads(layout.path("runtime_failures.json").read_text())["failures"] == [{"detail": "retained"}]
    assert json.loads(layout.path("run_state.json").read_text())["status"] == "interrupted"
    collection.calls.clear()
    collection.invoke("--resume")
    assert collection.calls == [2]


@pytest.mark.parametrize(("error_type", "prefix"), [(PacketLatencyError, "sniff"),
                                                   (EnergyProfilingError, "energy"),
                                                   (MIPSProfilingError, "mips")])
def test_required_measurement_error_notes_survive_cli_translation(collection, error_type, prefix):
    failure = error_type("fixture required measurement failed")
    failure.add_note("fixture owned process cleanup failed")

    def measure(request, **kwargs):
        if request.cpu == 2:
            raise failure
        return collection.write_case(request, **kwargs)

    errors = io.StringIO()
    with pytest.raises(SystemExit) as caught:
        collection.invoke(case=measure, error_output=errors)
    assert caught.value.code == 1
    diagnostic = errors.getvalue()
    assert f"[{prefix}][ERROR]" in diagnostic
    assert "fixture required measurement failed" in diagnostic
    assert "fixture owned process cleanup failed" in diagnostic


@pytest.mark.parametrize("stage", ["pcap parsing", "archived pcap parsing", "packet latency merge"])
def test_packet_timeout_cleanup_notes_reach_cli_without_raw_command(collection, monkeypatch, stage):
    command = ["fixture-parser", "--token", "fixture-raw-secret"]
    timeout = subprocess.TimeoutExpired(command, 30, output=b"fixture partial output")
    timeout.add_note("fixture SIGKILL refusal; child reap unconfirmed")
    monkeypatch.setenv("ACPROF_POSTPROCESS_TIMEOUT_S", "30")

    def measure(request, **kwargs):
        if request.cpu == 2:
            # Patch only the OS command boundary while the public packet wrapper runs.
            with monkeypatch.context() as scoped:
                scoped.setattr(packet_processing.host_command, "run_command", Mock(side_effect=timeout))
                packet_processing.run_packet_command(command, stage=stage)
        return collection.write_case(request, **kwargs)

    errors = io.StringIO()
    with pytest.raises(SystemExit) as caught:
        collection.invoke(case=measure, error_output=errors)
    assert caught.value.code == 1
    diagnostic = errors.getvalue()
    assert f"{stage} timed out after 30s" in diagnostic
    assert "ACPROF_POSTPROCESS_TIMEOUT_S" in diagnostic
    assert "fixture SIGKILL refusal; child reap unconfirmed" in diagnostic
    assert "fixture-raw-secret" not in diagnostic
    assert timeout.output == b"fixture partial output"
