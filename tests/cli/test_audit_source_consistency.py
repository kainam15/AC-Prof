"""A standalone audit checks the source it actually read after examining its windows."""
import json

import pytest

from acprof.analysis import audit
from acprof.cli.audit import main


@pytest.fixture
def experiment(tmp_path):
    path = tmp_path / "result_all.csv"
    path.write_text("cpu_cores,mem_cap_gb,gpu_mode,input_scale,warmup,repeat_idx,status,latency_app_s,environment_class\n"
                    "2,8,off,1,0,0,ok,1,unknown\n")
    (tmp_path / "run_state.json").write_text(json.dumps({
        "status": "complete", "runtime": {"planned": {"scales": [1]}},
        "options": {"cpus": "2", "mems": "8", "gpus": "off", "warmup": 0, "repeat": 1},
    }))
    return path


def strict(source, capsys):
    status = main([str(source), "--require-complete", "--require-ok", "--json"])
    return status, json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("change", ["state", "repaired-quality", "window", "removed-csv", "same-bytes"])
def test_strict_audit_rechecks_observations_after_counting(experiment, change, monkeypatch, capsys):
    state = experiment.with_name("run_state.json")
    quality = experiment.with_name("quality_checks.json")
    if change == "repaired-quality":
        quality.write_text("{")
    original = audit._audit_rows

    def scan_then_change(*args, **kwargs):
        original(*args, **kwargs)
        if change == "state":
            state.write_text('{"status":"failed"}')
        elif change == "repaired-quality":
            quality.write_text('{"schema_version":1,"checks":[]}')
        elif change == "window":
            experiment.write_text(experiment.read_text().replace("off,1,0,0", "off,1,0,1"))
        elif change == "removed-csv":
            experiment.unlink()
        else:
            for path in (state, experiment):
                replacement = path.with_suffix(".new")
                replacement.write_bytes(path.read_bytes())
                replacement.replace(path)

    monkeypatch.setattr(audit, "_audit_rows", scan_then_change)
    status, report = strict(experiment, capsys)
    assert report["counts"]["formal_ok"] == 1
    assert report["coverage"]["missing"] == 0
    if change == "same-bytes":
        assert status == 0
        assert report["accepted"]
    else:
        assert status == 1
        assert not report["accepted"]
        path = {"state": state, "repaired-quality": quality}.get(change, experiment)
        assert any(i["code"] == "changing_snapshot" and path.name in i["message"] for i in report["issues"])


def test_transient_alias_cannot_bind_a_different_table_to_the_first_selector(experiment, monkeypatch, capsys):
    alias = experiment.with_name("selected.csv")
    alias.symlink_to(experiment)
    alternate = experiment.with_name("alternate.csv")
    alternate.write_bytes(experiment.read_bytes())
    read = audit.read_result_csv_snapshot
    scan = audit._audit_rows

    def select(target):
        alias.unlink()
        alias.symlink_to(target)

    def read_alternate(*args, **kwargs):
        select(alternate)
        return read(*args, **kwargs)

    def restore_after_counting(*args, **kwargs):
        scan(*args, **kwargs)
        select(experiment)

    monkeypatch.setattr(audit, "read_result_csv_snapshot", read_alternate)
    monkeypatch.setattr(audit, "_audit_rows", restore_after_counting)
    status, report = strict(alias, capsys)
    assert status == 1
    assert not report["accepted"]
    assert any(i["code"] == "changing_snapshot" and alias.name in i["message"] for i in report["issues"])


def test_selector_change_while_reading_quality_returns_a_structured_failure(experiment, monkeypatch, capsys):
    alias = experiment.with_name("selected.csv")
    alias.symlink_to(experiment)
    other = experiment.with_name("other.csv")
    other.write_bytes(experiment.read_bytes())
    read = audit.read_quality

    def replace_before_quality(*args, **kwargs):
        alias.unlink()
        alias.symlink_to(other)
        return read(*args, **kwargs)

    monkeypatch.setattr(audit, "read_quality", replace_before_quality)
    status, report = strict(alias, capsys)
    assert status == 1
    assert not report["valid"]
    assert any(i["code"] == "changing_snapshot" and alias.name in i["message"] for i in report["issues"])
