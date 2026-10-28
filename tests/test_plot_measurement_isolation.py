"""Plot entry points acquire exclusion before loading or running renderers."""

import json
import selectors
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from acprof.host import run_state


def _run_cli(tmp_path, invocation, *, check_imports=False):
    code = '''
import json
import sys
from pathlib import Path
from acprof.host import run_state
run_state.MEASUREMENT_LOCK_ROOT = Path(sys.argv[1])
run_state.MEASUREMENT_LOCK_NAMESPACE = bytes.fromhex(sys.argv[2])
from acprof.cli.main import main
status = main(json.loads(sys.argv[3]))
if sys.argv[4] == "check":
    heavy = sorted({name.split(".")[0] for name in sys.modules
                    if name.split(".")[0] in {"matplotlib", "pandas", "numpy", "scipy"}})
    assert not heavy, heavy
raise SystemExit(status)
'''
    return subprocess.run(
        [sys.executable, "-c", code, str(run_state.MEASUREMENT_LOCK_ROOT),
         run_state.MEASUREMENT_LOCK_NAMESPACE.hex(), json.dumps(invocation),
         "check" if check_imports else ""],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )


def test_busy_plot_fails_before_importing_renderers_or_reading_input(tmp_path):
    missing = tmp_path / "not-created" / "result_all.csv"
    with run_state.MeasurementLock():
        result = _run_cli(tmp_path, ["plot", str(missing)], check_imports=True)

    assert result.returncode == 1, result.stderr
    assert "测量锁" in result.stderr
    assert "Traceback" not in result.stderr
    assert not missing.parent.exists()


def test_help_works_during_measurement_without_loading_renderers(tmp_path):
    with run_state.MeasurementLock():
        result = _run_cli(tmp_path, ["plot", "--help"], check_imports=True)

    assert result.returncode == 0, result.stderr
    assert "usage: acprof plot" in result.stdout
    assert not result.stderr


@pytest.mark.parametrize("outcome", ["success", "error", "interrupt"])
def test_plot_excludes_measurements_until_rendering_finishes(tmp_path, monkeypatch, outcome):
    import pandas as pd

    from acprof.cli import plot
    from acprof.plotting import data

    seen = []

    def prepare(source, **_kwargs):
        seen.append(source)
        with pytest.raises(run_state.RunStateError, match="测量锁"):
            with run_state.MeasurementLock():
                pytest.fail("measurement was allowed while plotting")
        if outcome == "error":
            raise ValueError("injected invalid result")
        if outcome == "interrupt":
            raise KeyboardInterrupt
        return pd.DataFrame()

    monkeypatch.setattr(data, "prepare_df", prepare)
    monkeypatch.setattr(data, "read_static_meta", lambda _: {"runtime_kind": "process"})
    from acprof.plotting import process

    def render(*_args, **_kwargs):
        with pytest.raises(run_state.RunStateError, match="测量锁"):
            with run_state.MeasurementLock():
                pytest.fail("measurement was allowed during rendering")
        return Path("synthetic.png")

    monkeypatch.setattr(process, "plot_process_results", render)
    source = str(tmp_path / "result_all.csv")
    if outcome == "success":
        assert plot.main([source]) == 0
    else:
        exception = ValueError if outcome == "error" else KeyboardInterrupt
        with pytest.raises(exception):
            plot.main([source])
    assert seen == [source]
    with run_state.MeasurementLock():
        pass


def test_renderer_import_failure_releases_measurement_lock(monkeypatch):
    import builtins

    from acprof.cli import plot

    original = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == "acprof.plotting.pipeline":
            with pytest.raises(run_state.RunStateError, match="测量锁"):
                with run_state.MeasurementLock():
                    pytest.fail("renderer import started without exclusion")
            raise ImportError("injected rendering dependency failure")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    with pytest.raises(ImportError, match="injected rendering dependency failure"):
        plot.main(["unused.csv"])
    with run_state.MeasurementLock():
        pass


@pytest.mark.parametrize("system", ["Darwin", "Windows"])
def test_offline_plotting_does_not_require_linux_locks_on_other_systems(monkeypatch, system):
    from acprof import platform
    from acprof.cli import plot

    rendered = []
    pipeline = ModuleType("acprof.plotting.pipeline")
    pipeline.plot_results = rendered.append
    monkeypatch.setitem(sys.modules, pipeline.__name__, pipeline)
    monkeypatch.setattr(platform.platform, "system", lambda: system)

    def unexpected_lock():
        pytest.fail("non-Linux offline rendering requested a Linux lock")

    monkeypatch.setattr(run_state, "MeasurementLock", unexpected_lock)
    assert plot.main(["archived.csv"]) == 0
    assert rendered == ["archived.csv"]


@pytest.mark.parametrize("exit_mode", ["normal", "killed"])
def test_plot_process_blocks_contenders_until_exit(tmp_path, exit_mode):
    code = '''
import sys
from pathlib import Path
from types import ModuleType
from acprof.host import run_state
run_state.MEASUREMENT_LOCK_ROOT = Path(sys.argv[1])
run_state.MEASUREMENT_LOCK_NAMESPACE = bytes.fromhex(sys.argv[2])
pipeline = ModuleType("acprof.plotting.pipeline")
def render(_source):
    print("rendering", flush=True)
    sys.stdin.read(1)
pipeline.plot_results = render
sys.modules[pipeline.__name__] = pipeline
from acprof.cli.main import main
raise SystemExit(main(["plot", "synthetic.csv"]))
'''
    first = subprocess.Popen(
        [sys.executable, "-c", code, str(run_state.MEASUREMENT_LOCK_ROOT),
         run_state.MEASUREMENT_LOCK_NAMESPACE.hex()],
        cwd=tmp_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        with selectors.DefaultSelector() as ready:
            ready.register(first.stdout, selectors.EVENT_READ)
            assert ready.select(timeout=10), "plot child did not become ready"
        assert first.stdout.readline().strip() == "rendering"
        contender = _run_cli(tmp_path, ["plot", "unused.csv"], check_imports=True)
        assert contender.returncode == 1, contender.stderr
        assert "测量锁" in contender.stderr
        assert "Traceback" not in contender.stderr
        with pytest.raises(run_state.RunStateError, match="测量锁"):
            with run_state.MeasurementLock():
                pytest.fail("measurement overlapped the plot process")
        if exit_mode == "killed":
            first.kill()
        _, stderr = first.communicate("\n" if exit_mode == "normal" else None, timeout=10)
        assert not stderr
        assert first.returncode == (0 if exit_mode == "normal" else -9)
    finally:
        if first.poll() is None:
            first.kill()
            first.communicate(timeout=10)
    with run_state.MeasurementLock():
        pass
