"""The minimum-memory probe shares the collection lock across output directories."""
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.cli import probe
from acprof.host import run_state
from acprof.host.detect import TaskInfo


def test_competing_probe_stops_before_docker_or_output(tmp_path):
    script = """
import sys
from pathlib import Path
from unittest.mock import Mock
from acprof.cli import main, probe
from acprof.host import run_state
run_state.MEASUREMENT_LOCK_ROOT = Path(sys.argv[1])
run_state.MEASUREMENT_LOCK_NAMESPACE = bytes.fromhex(sys.argv[2])
probe.bootstrap_project_env = Mock()
probe.require_collection_host = Mock()
probe.require_native_docker = Mock(side_effect=AssertionError("busy probe reached Docker"))
sys.exit(main.main(["probe", "--model", "fixture/model", "--output-dir", sys.argv[3]]))
"""
    output = tmp_path / "other-results"
    with run_state.MeasurementLock():
        attempt = subprocess.run(
            [sys.executable, "-c", script, str(run_state.MEASUREMENT_LOCK_ROOT),
             run_state.MEASUREMENT_LOCK_NAMESPACE.hex(), str(output)],
            capture_output=True, text=True, timeout=10, check=False,
        )
    assert attempt.returncode == 2, attempt.stderr
    assert "无法取得本机测量锁" in attempt.stderr
    assert "Traceback" not in attempt.stderr
    assert not output.exists()


@pytest.mark.parametrize("interrupt", [False, True])
def test_probe_owns_lock_through_finalization_and_releases_on_exit(tmp_path, monkeypatch, interrupt):
    stages = []

    def check(stage):
        with pytest.raises(run_state.RunStateError, match="测量锁"):
            with run_state.MeasurementLock():
                pytest.fail(f"{stage} ran without the host measurement lock")
        stages.append(stage)

    task = TaskInfo("fixture/model", "fill-mask", "nlp", "transformers_pipeline", "transformers", "a" * 40, "manual")
    image = SimpleNamespace(runtime_environment={})

    def resolve(**kwargs):
        check("resolution")
        return task

    def build(*args, **kwargs):
        check("image")
        return image

    def execute(**kwargs):
        check("request")
        if interrupt:
            raise KeyboardInterrupt
        return {"status": "ok", "timing": {}, "artifacts": {"summary": str(tmp_path / "summary.json")}}

    monkeypatch.setattr(probe, "bootstrap_project_env", Mock())
    monkeypatch.setattr(probe, "require_collection_host", Mock())
    monkeypatch.setattr(probe, "require_native_docker", lambda: check("docker"))
    monkeypatch.setattr(probe, "require_cgroup_prerequisites", lambda: "v2")
    monkeypatch.setattr("acprof.host.detect.detect_task", resolve)
    monkeypatch.setattr(probe, "prepare_image", build)
    monkeypatch.setattr(probe, "plan_input_scales", lambda **kwargs: check("input"))
    monkeypatch.setattr(probe, "run_largest_scale_probe", execute)
    monkeypatch.setattr(probe, "write_probe_summary", lambda *_args: check("summary"))
    args = ["--model", task.model_id, "--output-dir", str(tmp_path / "output")]
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            probe.main(args)
    else:
        assert probe.main(args) == 0
    expected = ["docker", "resolution", "image", "input", "request"]
    assert stages == (expected if interrupt else [*expected, "summary"])
    with run_state.MeasurementLock():
        pass
