"""Offline commands cannot overlap collection or publish while exclusion is lost."""

import json
import selectors
import subprocess
import sys
from types import SimpleNamespace

import pytest

from acprof.host import run_state

CASES = [
    ("plot", ["plot", "missing.csv"]),
    ("stats", ["stats", "missing.csv", "--output", "statistics.json"]),
    ("compare", ["compare", "--left", "missing.csv", "--right", "other.csv", "--output", "comparison.json"]),
    ("report", ["report", "missing.csv", "--output", "report.html"]),
    ("audit", ["audit", "missing.csv", "--json"]),
    ("verify", ["results", "verify", "missing"]),
    ("export", ["results", "export", "missing", "wide.csv"]),
]


def run_cli(tmp_path, arguments):
    code = '''
import json
import sys
from pathlib import Path
from acprof.host import run_state
run_state.MEASUREMENT_LOCK_ROOT = Path(sys.argv[1])
run_state.MEASUREMENT_LOCK_NAMESPACE = bytes.fromhex(sys.argv[2])
from acprof.cli.main import main
raise SystemExit(main(json.loads(sys.argv[3])))
'''
    return subprocess.run(
        [sys.executable, "-c", code, str(run_state.MEASUREMENT_LOCK_ROOT),
         run_state.MEASUREMENT_LOCK_NAMESPACE.hex(), json.dumps(arguments)],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )


@pytest.mark.parametrize(("name", "arguments"), CASES, ids=[case[0] for case in CASES])
def test_busy_commands_fail_before_reading_results_or_publishing(tmp_path, name, arguments):
    with run_state.MeasurementLock():
        result = run_cli(tmp_path, arguments)

    assert result.returncode == 1, result.stderr
    assert "测量锁" in result.stderr
    assert "Traceback" not in result.stderr
    assert not result.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("command", ["plot", "stats", "compare", "report", "audit", "results"])
def test_help_remains_available_during_collection(tmp_path, command):
    with run_state.MeasurementLock():
        result = run_cli(tmp_path, [command, "--help"])
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert not result.stderr
    assert list(tmp_path.iterdir()) == []


def test_static_metric_catalog_remains_available_during_collection(tmp_path):
    with run_state.MeasurementLock():
        result = run_cli(tmp_path, ["audit", "--metrics"])
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("system", ["Darwin", "Windows"])
def test_report_on_non_linux_does_not_request_linux_lock(tmp_path, monkeypatch, system):
    from acprof import platform
    from acprof.analysis import model
    from acprof.cli import report
    from acprof.plotting import report as rendering

    monkeypatch.setattr(platform.platform, "system", lambda: system)

    def unexpected_lock():
        pytest.fail("offline report requested a Linux-only measurement lock")

    monkeypatch.setattr(run_state, "MeasurementLock", unexpected_lock)
    monkeypatch.setattr(model, "load_analysis", lambda _: SimpleNamespace(configs=[]))
    target = tmp_path / "report.html"
    monkeypatch.setattr(rendering, "write_report", lambda _model, output, **_kwargs: output.write_text("report"))
    assert report.main([str(tmp_path / "archive.csv"), "--output", str(target)]) == 0
    assert target.read_text() == "report"


@pytest.mark.parametrize(("name", "arguments", "module"), [
    (*CASES[0], "acprof.plotting.pipeline"),
    (*CASES[1], "acprof.analysis.statistics_report"),
    (*CASES[2], "acprof.analysis.independent_comparison"),
    (*CASES[3], "acprof.analysis.model"),
    (*CASES[4], "acprof.analysis.audit"),
    (*CASES[5], "acprof.result_layers"),
    (*CASES[6], "acprof.result_layers"),
])
@pytest.mark.parametrize("failure", [ImportError, KeyboardInterrupt])
def test_command_imports_under_lock_and_releases_it_on_failure(tmp_path, monkeypatch, name, arguments, module, failure):
    import builtins

    from acprof.cli.main import main

    original = builtins.__import__
    reached = []

    def import_module(target, *args, **kwargs):
        if target == module:
            with pytest.raises(run_state.RunStateError, match="测量锁"):
                with run_state.MeasurementLock():
                    pytest.fail("analysis module loaded without exclusion")
            reached.append(target)
            raise failure("injected offline import failure")
        return original(target, *args, **kwargs)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(builtins, "__import__", import_module)
    with pytest.raises((failure, SystemExit)) as error:
        main(arguments)
    if isinstance(error.value, SystemExit):
        assert error.value.code == 1
    assert reached == [module]
    assert list(tmp_path.iterdir()) == []
    with run_state.MeasurementLock():
        pass


@pytest.mark.parametrize(("name", "arguments"), CASES, ids=[case[0] for case in CASES])
def test_malformed_lock_is_not_treated_as_an_unlocked_host(tmp_path, monkeypatch, name, arguments):
    from acprof.cli.main import main

    monkeypatch.setattr(run_state, "MEASUREMENT_LOCK_ROOT", tmp_path)
    lock = run_state.MeasurementLock()
    lock.path.mkdir()
    monkeypatch.chdir(tmp_path)
    try:
        try:
            code = main(arguments)
        except SystemExit as error:
            code = error.code
        assert code == 1
        assert list(tmp_path.iterdir()) == [lock.path]
    finally:
        lock.path.rmdir()
    with run_state.MeasurementLock():
        pass


@pytest.mark.parametrize("exit_mode", ["normal", "killed"])
def test_statistics_process_excludes_collection_and_other_commands_until_exit(tmp_path, exit_mode):
    source = tmp_path / "source.csv"
    source.write_text(
        "cpu_cores,mem_cap_gb,gpu_mode,input_scale,warmup,repeat_idx,status,latency_app_s\n"
        "2,8,off,64,0,0,ok,0.01\n2,8,off,64,0,1,ok,0.02\n2,8,off,64,0,2,ok,0.03\n",
    )
    original = source.read_bytes()
    target = tmp_path / "statistics.json"
    code = '''
import sys
from pathlib import Path
from acprof.host import run_state
run_state.MEASUREMENT_LOCK_ROOT = Path(sys.argv[1])
run_state.MEASUREMENT_LOCK_NAMESPACE = bytes.fromhex(sys.argv[2])
from acprof.analysis import statistics_report
summarize = statistics_report.summarize_windows
def controlled_summary(*args, **kwargs):
    print("computing", flush=True)
    sys.stdin.read(1)
    return summarize(*args, **kwargs)
statistics_report.summarize_windows = controlled_summary
from acprof.cli import stats
publish = stats.atomic_write_json
def checked_publication(*args, **kwargs):
    try:
        with run_state.MeasurementLock():
            raise AssertionError("publication lost the measurement lock")
    except run_state.RunStateError:
        pass
    return publish(*args, **kwargs)
stats.atomic_write_json = checked_publication
raise SystemExit(stats.main([sys.argv[3], "--resamples", "30", "--output", sys.argv[4]]))
'''
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(run_state.MEASUREMENT_LOCK_ROOT),
         run_state.MEASUREMENT_LOCK_NAMESPACE.hex(), str(source), str(target)],
        cwd=tmp_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        with selectors.DefaultSelector() as ready:
            ready.register(process.stdout, selectors.EVENT_READ)
            assert ready.select(timeout=10), "statistics child did not become ready"
        assert process.stdout.readline().strip() == "computing"
        with pytest.raises(run_state.RunStateError, match="测量锁"):
            with run_state.MeasurementLock():
                pytest.fail("collection overlapped statistics")
        contender = run_cli(tmp_path, ["compare", "--left", "absent", "--right", "absent"])
        assert contender.returncode == 1
        assert "测量锁" in contender.stderr
        assert not contender.stdout
        if exit_mode == "killed":
            process.kill()
        stdout, stderr = process.communicate("\n" if exit_mode == "normal" else None, timeout=10)
        assert not stderr
        assert process.returncode == (0 if exit_mode == "normal" else -9)
        if exit_mode == "normal":
            assert "已保存" in stdout
            assert json.loads(target.read_text())["groups"][0]["mean"] == pytest.approx(.02)
        else:
            assert not target.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
    assert source.read_bytes() == original
    with run_state.MeasurementLock():
        pass
