from types import SimpleNamespace

from acprof.host.images import environment, runtime
from acprof.host.profiling.tools import execution_environment
from acprof.system import command


def test_image_identity_and_manifest_reads_have_separate_budgets(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"image_id":"sha256:abc"}', stderr="")

    monkeypatch.setattr(command, "run_command", run)
    assert runtime.inspect_identity("fixture")["image_id"] == "sha256:abc"
    environment.read_image_manifest("sha256:abc", "/manifest.json")
    assert [kwargs.get("timeout") for _, kwargs in calls] == [15, 120]


def test_profiler_queries_are_bounded(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"Id":"sha256:' + 'a' * 64 + '"}', stderr="")

    monkeypatch.setattr(execution_environment, "run_command", run)
    monkeypatch.setattr(execution_environment, "find_nsys_importer", lambda _: "/nsys/importer")
    execution_environment._inspect_execution_image("fixture")
    execution_environment.get_massif_version("fixture")
    execution_environment.get_nsys_version("/nsys")
    execution_environment.validate_nsys_container_runtime("fixture", "/nsys")
    assert [kwargs.get("timeout") for _, kwargs in calls] == [15, 120, 15, 120]
