import contextlib
import csv
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from acprof.analysis.audit import audit_result
from acprof.failures import Failure, RuntimeFailure
from acprof.tui.progress import RunProgressTracker


@pytest.mark.parametrize('status', (401, 403, 429))
def test_hub_access_errors_preserve_http_status_through_exception_chains(status):
    import requests
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    from acprof.failures import compatibility_status, failure_from_exception

    response = requests.Response()
    response.status_code = status
    error = (GatedRepoError if status == 401 else HfHubHTTPError)("opaque server detail", response=response)
    wrapped = RuntimeError("artifact planning failed")
    wrapped.__cause__ = error
    failure = failure_from_exception(wrapped, stage="artifact_planning")
    if status in (401, 403):
        assert (failure.reason_code) == ("access_denied")
        assert (failure.evidence["http_status"]) == (status)
        assert (failure.stage) == ("artifact_planning")
        assert (compatibility_status(failure)) == ("access_denied")
    else:
        assert (failure.reason_code) != ("access_denied")

def test_actual_dtype_reports_mixed_parameters_instead_of_primary_property():
    from acprof.container.load_policy import actual_dtype
    parameters = [SimpleNamespace(dtype=dtype, is_floating_point=lambda: True) for dtype in ("torch.float16", "torch.float32")]
    model = SimpleNamespace(dtype="torch.float16", parameters=lambda: iter(parameters))
    assert (actual_dtype({"model": model})) == ("mixed[torch.float16, torch.float32]")

def test_all_consumers_preserve_the_same_reason_code():
    from acprof.analysis.compatibility import write_compatibility_report
    from acprof.tui.reports import read_report
    failure = Failure("preflight", "runtime_task_unsupported", "本地化说明", runtime_profile="cv-transformers560-cpu")
    row = {"model_id": "fixture/model", "failure": failure.to_dict()}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_compatibility_report(root, [row])
        with (root / "models.csv").open() as stream:
            assert (next(csv.DictReader(stream))["reason_code"]) == (failure.reason_code)
        assert (failure.reason_code) in ((root / "REPORT.md").read_text())
        source = root / "coverage.json"
        source.write_text(json.dumps({"schema_version": 1, "scope": "selected_sample_only; no_formal_measurement", "rows": [row]}))
        assert (read_report(source).rows[0].cells[2]) == (failure.reason_code)
    tracker = RunProgressTracker()
    for line in str(RuntimeFailure(failure)).splitlines():
        tracker.feed(line)
    assert (tracker.snapshot.failure) == (failure.to_dict())

def test_compatibility_report_publication_is_durable(tmp_path):
    from acprof.analysis.compatibility import write_compatibility_report

    with patch("os.fsync") as fsync:
        write_compatibility_report(tmp_path, [{"model_id": "fixture/model"}])

    assert fsync.call_count >= 4
    assert (tmp_path / "models.csv").read_text(encoding="utf-8").startswith("model_id,")
    assert (tmp_path / "REPORT.md").read_text(encoding="utf-8").startswith("# Compatibility report\n")


@pytest.mark.parametrize('plan', ({'total_selected_bytes': 150}, {'selected_bytes': None}))
def test_budget_uses_selected_files_and_never_claims_measured_oom(plan):
    from acprof.analysis.compatibility import result_status
    from acprof.resource_budget import assess_resource_budget
    small = assess_resource_budget(plan={"selected_bytes": 50, "repository_bytes": 5000}, max_download_bytes=100)
    assert (small["status"]) == ("within_budget")
    result = assess_resource_budget(plan=plan, max_download_bytes=100)
    assert (result_status(result)) == ("unverified")
    assert not (result["failure"]["evidence"]["measured_oom"])

def test_recorded_result_report_preserves_warnings_and_legacy_unknown():
    from acprof.analysis.compatibility import report_results, result_status
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sources = [root / "current", root / "legacy"]
        for source in sources:
            source.mkdir()
            (source / "static_meta.json").write_text(json.dumps({"model_id": "fixture/model", "model_revision": "a" * 40}))
            (source / "capability_report.json").write_text('{"full_profile_complete": true}')
        (sources[0] / "quality_checks.json").write_text(json.dumps({"schema_version": 1, "checks": [{
            "code": "unused_checkpoint_weights", "severity": "warning", "observed": ["unused.weight"],
            "threshold": 0, "detail": "Unused checkpoint weight", "evidence": {"source": "from_pretrained"}}]}))
        before = {path: path.read_bytes() for source in sources for path in source.iterdir()}
        report = report_results(sources, root / "report")
        assert (result_status(report["rows"][0])) == ("full_success_with_warnings")
        assert (result_status(report["rows"][1])) == ("full_success_quality_unknown")
        assert (before) == ({path: path.read_bytes() for path in before})

@pytest.mark.parametrize('content', (
    '{',
    '[]',
    '{"full_profile_complete": NaN}',
    '{"full_profile_complete": 1e999}',
    json.dumps({"padding": "x" * (4 * 1024 * 1024)}),
))
def test_recorded_result_report_marks_invalid_artifact_inconclusive(tmp_path, content):
    from acprof.analysis.compatibility import report_results, result_status
    source = tmp_path / 'fixture'
    source.mkdir()
    (source / 'static_meta.json').write_text(json.dumps({
        'model_id': 'fixture/model', 'model_revision': 'a' * 40,
    }))
    artifact = source / 'capability_report.json'
    artifact.write_text(content)

    report = report_results([source], tmp_path / 'report')

    row = report['rows'][0]
    assert (result_status(row)) == ('inconclusive')
    assert not row['full_profile_complete']
    assert (row['failure']['reason_code']) == ('recorded_evidence_invalid')
    assert (row['failure']['evidence']['artifact']) == (str(artifact.resolve()))
    assert (tmp_path / 'report' / 'coverage.json').is_file()


@pytest.mark.parametrize(('name', 'content'), (
    ('runtime_failures.json', '{"failures": {}}'),
    ('runtime_failures.json', '{"failures": [{}]}'),
    ('runtime_validation.json', '{"devices": []}'),
    ('runtime_validation.json', '{"devices": {"off": {"failure": {"reason_code": "invalid"}}}}'),
    ('model_resolution.json', '{"failure": []}'),
    ('model_resolution.json', '{"failure": {"reason_code": "invalid"}}'),
))
def test_recorded_result_report_rejects_invalid_nested_evidence(tmp_path, name, content):
    from acprof.analysis.compatibility import report_results, result_status
    source = tmp_path / 'fixture'
    source.mkdir()
    (source / 'static_meta.json').write_text(json.dumps({'model_id': 'fixture/model'}))
    (source / name).write_text(content)

    row = report_results([source], tmp_path / 'report')['rows'][0]

    assert (result_status(row)) == ('inconclusive')
    assert (row['failure']['reason_code']) == ('recorded_evidence_invalid')
    assert (row['failure']['evidence']['artifact_name']) == (name)


def test_recorded_result_report_accepts_failure_only_directory(tmp_path):
    from acprof.analysis.compatibility import report_results, result_status
    source = tmp_path / 'fixture'
    source.mkdir()
    failure = Failure('preflight', 'runtime_task_unsupported', 'unsupported task').to_dict()
    (source / 'runtime_failures.json').write_text(json.dumps({'failures': [failure]}))

    row = report_results([source], tmp_path / 'report')['rows'][0]

    assert (row['model_id']) == ('fixture')
    assert (row['failure']) == (failure)
    assert (result_status(row)) == ('failed')


def test_recorded_result_report_keeps_runtime_failure_ahead_of_invalid_artifact(tmp_path):
    from acprof.analysis.compatibility import report_results, result_status
    source = tmp_path / 'fixture'
    source.mkdir()
    failure = Failure('predict', 'inference_failed', 'request failed').to_dict()
    (source / 'static_meta.json').write_text(json.dumps({'model_id': 'fixture/model'}))
    (source / 'runtime_failures.json').write_text(json.dumps({'failures': [failure]}))
    (source / 'capability_report.json').write_text('{')

    row = report_results([source], tmp_path / 'report')['rows'][0]

    assert (result_status(row)) == ('failed')
    assert (row['failure']) == (failure)
    assert [item['reason_code'] for item in row['failures']] == [
        'inference_failed', 'recorded_evidence_invalid',
    ]

def test_quality_does_not_revoke_verified_capability():
    from acprof.analysis.compatibility import result_status
    from acprof.capabilities import Capability, CapabilityReport, apply_runtime_validation
    check = {"code": "weights_reinitialized", "severity": "warning"}
    report = CapabilityReport("full", measurement={"latency": Capability("verified")}, requested={"latency"},
                              collection_complete=True, identity={"comparability_class": "native_linux"})
    apply_runtime_validation(report, {"devices": {"off": {"status": "ok", "quality_checks": [check]}}})
    snapshot = report.to_dict()
    assert (snapshot["full_profile_complete"])
    assert ("weights_reinitialized") not in (json.dumps(snapshot))
    assert (result_status({**snapshot, "quality_checks": [check]})) == ("full_success_with_warnings")

def test_server_failure_survives_client_and_is_written_after_request():
    from acprof.host.client import ClientRunner
    from acprof.host.client_config import ClientConfig
    failure = Failure("predict", "inference_failed", "arbitrary detail", evidence={"model_loaded": True})
    runner = ClientRunner.__new__(ClientRunner)
    runner.runtime_failures = []
    runner.config = ClientConfig()
    runner._proxy_options = {}
    response_body = {"error": "different text", "failure": failure.to_dict()}
    response = SimpleNamespace(status_code=500, text=json.dumps(response_body))
    with patch("acprof.host.client.requests.post", return_value=response):
        with pytest.raises(RuntimeFailure) as caught:
            runner._one_request(1, "case_w0:0", {"input_scale": 1})
    failure = caught.value.failure
    assert (failure.reason_code) == ("inference_failed")
    assert (failure.evidence["input_scale"]) == (1)
    assert (failure.evidence["request_id"]) == ("case_w0:0")
    with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
        runner.config.out_csv = str(Path(directory, "case.csv"))
        runner.main = Mock(side_effect=caught.value)
        with pytest.raises(SystemExit):
            runner.run_cli()
        saved = json.loads(Path(directory, "case.csv.runtime_failures.json").read_text())
    assert (saved["failures"]) == ([failure.to_dict()])

def test_collect_failures_rejects_nonfinite_case_evidence(tmp_path):
    from acprof.artifact_layout import case_sidecar
    from acprof.failures import collect_failures

    case_csv = tmp_path / "case.csv"
    case_sidecar(case_csv, "runtime_failures").write_text(
        '{"failures": [], "corrupt_metric": NaN}'
    )

    with pytest.raises(ValueError, match="invalid runtime failure evidence JSON"):
        collect_failures(tmp_path, [case_csv])

    assert not (tmp_path / "runtime_failures.json").exists()


def test_tui_keeps_typed_failure_instead_of_classifying_detail():
    failure = Failure("preflight", "runtime_dependency_missing", "任意本地化文本", evidence={"module": "skimage"})
    tracker = RunProgressTracker()
    for line in str(RuntimeFailure(failure)).splitlines():
        tracker.feed(line)
    assert (tracker.snapshot.failure["reason_code"]) == ("runtime_dependency_missing")

def test_audit_consumes_quality_and_failures_without_logs():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "result_all.csv").write_text("status,warmup\nok,0\n")
        failure = Failure("load", "processor_incompatible", "localized").to_dict()
        (root / "runtime_validation.json").write_text(json.dumps({"devices": {"off": {"failure": failure}}}))
        check = {"code": "weights_reinitialized", "severity": "warning", "observed": ["head.weight"],
                 "threshold": 0, "detail": "loading info", "evidence": {"source": "from_pretrained"}}
        (root / "quality_checks.json").write_text(json.dumps({"schema_version": 1, "checks": [check]}))
        report = audit_result(root)
    assert (report.get("failures")) == ([failure])
    assert (report.get("quality_checks")) == ([{**check, "evidence": {
        **check["evidence"], "artifact": str(root / "quality_checks.json")}}])
