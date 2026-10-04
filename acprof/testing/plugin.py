"""pytest lifecycle for evidence, platform policy and offline lock isolation."""

from fnmatch import fnmatch
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.platform import detect_environment
from acprof.testing.evidence import Evidence
from acprof.testing.sharding import select_shard

EVIDENCE = pytest.StashKey[Evidence]()


def pytest_addoption(parser):
    group = parser.getgroup("acprof", "AC-Prof evidence and deterministic shards")
    group.addoption("--report", type=Path, help="Write schema v1 test evidence JSON")
    group.addoption("--pattern", action="append", help="Required test filename glob; repeatable")
    group.addoption("--require-no-skips", action="store_true", help="Fail on skip or xfail")
    group.addoption("--shard-index", type=int, default=0)
    group.addoption("--shard-count", type=int, default=1)


def pytest_configure(config):
    if not 0 <= config.getoption("shard_index") < config.getoption("shard_count"):
        raise pytest.UsageError("需要 0 <= --shard-index < --shard-count")
    if getattr(config.option, "numprocesses", None):
        raise pytest.UsageError("AC-Prof requires serial execution; use deterministic CI shards")
    for pattern in config.getoption("pattern") or []:
        config.addinivalue_line("python_files", pattern)
    # The wrapper also supports standalone fixture directories without pyproject.toml.
    registered = {marker.split(":", 1)[0].split("(", 1)[0] for marker in config.getini("markers")}
    for name in ("unit", "wsl", "native_linux", "hardware", "integration", "runtime", "visual"):
        if name not in registered:
            config.addinivalue_line("markers", f"{name}: AC-Prof test category")
    config.stash[EVIDENCE] = Evidence(config)


def pytest_ignore_collect(collection_path, config):
    patterns = config.getoption("pattern")
    if patterns and collection_path.suffix == ".py":
        return not any(fnmatch(collection_path.name, pattern) for pattern in patterns)
    return None


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_collection_modifyitems(config, items):
    environment = detect_environment()
    for item in items:
        if not any(item.get_closest_marker(name) for name in
                   ("wsl", "native_linux", "hardware", "integration", "runtime", "visual")):
            item.add_marker(pytest.mark.unit)
        if item.get_closest_marker("native_linux") and not environment.native:
            item.add_marker(pytest.mark.skip(reason="requires Native Linux validation"))
        elif item.get_closest_marker("wsl") and environment.environment != "wsl2":
            item.add_marker(pytest.mark.skip(reason="requires WSL2 integration environment"))
    # pytest applies -k/-m before hashing the selected suite and assigning shards.
    yield
    evidence = config.stash[EVIDENCE]
    patterns = config.getoption("pattern") or ["test_*.py"]
    evidence.discovery_counts = {pattern: sum(fnmatch(item.path.name, pattern) for item in items)
                                 for pattern in patterns}
    candidates = [item for item in items if any(fnmatch(item.path.name, p) for p in patterns)]
    ordered, selected = select_shard(candidates, config.getoption("shard_index"),
                                     config.getoption("shard_count"))
    evidence.collected = [item.nodeid for item in ordered]
    evidence.selected = [item.nodeid for item in selected]
    evidence.capture_inputs(sorted({item.path for item in ordered}))
    selected_set = set(selected)
    deselected = [item for item in items if item not in selected_set]
    items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    result = yield
    report = result.get_result()
    if report.failed:
        report.acprof_outcome = ("failed" if call.when == "call" and call.excinfo is not None
                                and isinstance(call.excinfo.value, (AssertionError, pytest.fail.Exception))
                                else "error")
    # Keep the phase reports for domain fixtures such as browser failure artifacts.
    item.stash.setdefault(PHASE_REPORTS, {})[report.when] = report


PHASE_REPORTS = pytest.StashKey[dict]()


def pytest_sessionstart(session):
    session.config.pluginmanager.register(_Reports(session.config.stash[EVIDENCE]), "acprof-reports")


class _Reports:
    def __init__(self, evidence):
        self.evidence = evidence

    def pytest_runtest_logreport(self, report):
        self.evidence.record(report)

    def pytest_collectreport(self, report):
        if report.failed:
            self.evidence.collection_errors.append({"id": report.nodeid, "reason": report.longreprtext})
        elif report.skipped:
            self.evidence.collection_skips.append({"id": report.nodeid, "reason": str(report.longrepr)})


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    session.config.stash[EVIDENCE].finish(session, exitstatus)


@pytest.fixture(scope="session", autouse=True)
def measurement_lock_root(tmp_path_factory):
    """Cover setup, call and teardown, including class/module scoped fixtures."""
    root = tmp_path_factory.mktemp("acprof-measurement-lock")
    with patch("acprof.host.run_state.MEASUREMENT_LOCK_ROOT", root):
        yield root
