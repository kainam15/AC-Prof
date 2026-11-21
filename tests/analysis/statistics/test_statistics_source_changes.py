"""Window statistics keep their selected result and observed metadata through bootstrap."""
import pytest

from acprof.analysis import statistics_report
from acprof.cli.stats import main
from acprof.results.layers import publish_result_layers


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "result_all.csv"
    path.write_text(
        "cpu_cores,mem_cap_gb,gpu_mode,input_scale,warmup,repeat_idx,status,latency_app_s\n"
        "2,8,off,1,0,0,ok,1\n2,8,off,1,0,1,ok,3\n2,8,off,1,0,2,ok,5\n"
    )
    return path


@pytest.fixture
def after_bootstrap(monkeypatch):
    summarize = statistics_report.summarize_windows

    def install(change):
        def summarize_then_change(*args, **kwargs):
            report = summarize(*args, **kwargs)
            change()
            return report

        monkeypatch.setattr(statistics_report, "summarize_windows", summarize_then_change)

    return install


@pytest.mark.parametrize("change", ["state", "new_quality", "removed", "repaired"])
def test_changed_audit_evidence_prevents_cli_publication(source, after_bootstrap, tmp_path, capsys, change):
    if change == "state":
        path = tmp_path / "run_state.json"
        path.write_text('{"status":"complete"}')
        after_bootstrap(lambda: path.write_text('{"status":"failed"}'))
    else:
        path = tmp_path / "quality_checks.json"
        quality = '{"schema_version":1,"checks":[]}'
        if change == "removed":
            path.write_text(quality)
            after_bootstrap(path.unlink)
        else:
            if change == "repaired":
                path.write_text("{")
            after_bootstrap(lambda: path.write_text(quality))

    output = tmp_path / "statistics.json"
    with pytest.raises(SystemExit) as stopped:
        main([str(source), "--metric", "latency_app_s", "--resamples", "5", "--output", str(output)])
    assert stopped.value.code == 1
    result = capsys.readouterr()
    assert not result.out
    assert path.name in result.err
    assert not output.exists()


def test_same_metadata_bytes_allow_publication_after_atomic_replacement(source, after_bootstrap, tmp_path):
    path = tmp_path / "run_state.json"
    path.write_text('{"status":"complete"}')
    replacement = tmp_path / "replacement"
    replacement.write_bytes(path.read_bytes())
    after_bootstrap(lambda: replacement.replace(path))
    report = statistics_report.build_statistics_report(source, ["latency_app_s"], resamples=5)
    assert (report["run_status"], report["groups"][0]["mean"]) == ("complete", 3)


@pytest.mark.parametrize("mode", ["alias", "directory"])
def test_selected_result_changes_prevent_cli_publication(source, after_bootstrap, tmp_path, capsys, mode):
    if mode == "alias":
        selection = tmp_path / "current.csv"
        selection.symlink_to(source)
        replacement = tmp_path / "replacement.csv"
        replacement.write_text(source.read_text().replace("ok,5", "ok,17"))
        next_link = tmp_path / "next.csv"
        next_link.symlink_to(replacement)
        after_bootstrap(lambda: next_link.replace(selection))
    else:
        selection = tmp_path
        after_bootstrap(lambda: publish_result_layers(source))
    output = tmp_path / "statistics.json"
    with pytest.raises(SystemExit) as stopped:
        main([str(selection), "--metric", "latency_app_s", "--resamples", "5", "--output", str(output)])
    assert stopped.value.code == 1
    assert not capsys.readouterr().out
    assert not output.exists()


def test_snapshot_from_transient_alias_target_cannot_be_reported(source, after_bootstrap, tmp_path, monkeypatch):
    from acprof.analysis import audit

    alias = tmp_path / "current.csv"
    alias.symlink_to(source)
    other = tmp_path / "other.csv"
    other.write_text(source.read_text().replace("ok,5", "ok,17"))
    read_snapshot = audit.read_result_csv_snapshot

    def retarget(target):
        alias.unlink()
        alias.symlink_to(target)

    def read_transient_target(*args, **kwargs):
        retarget(other)
        return read_snapshot(*args, **kwargs)

    monkeypatch.setattr(audit, "read_result_csv_snapshot", read_transient_target)
    after_bootstrap(lambda: retarget(source))
    with pytest.raises(ValueError, match="current.csv"):
        statistics_report.build_statistics_report(alias, ["latency_app_s"], resamples=5)
