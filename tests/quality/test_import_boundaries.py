"""Guard the dependency direction between presentation, collection and analysis."""
from pathlib import Path

import pytest

from scripts.check_import_boundaries import scan_import_violations, violations_in_text


@pytest.mark.parametrize(
    ("module", "source"),
    [
        ("acprof.analysis.report", "from acprof.host import hardware_conditions"),
        ("acprof.analysis.report", "from ..host.hardware_conditions import conditions_path"),
        ("acprof.host.orchestrator", "import acprof.cli.run"),
        ("acprof.container.server", "from acprof.tui.app import AcprofTui"),
        ("acprof.monitors.common", "from acprof.host.other import execute"),
        ("acprof.models.contract", "from acprof.host.detect import detect_task"),
        ("acprof.models.inspection.source", "from acprof.container.model_probe import validate"),
        ("acprof.hub.repository", "from acprof.host.model_store.store import store_root"),
        ("acprof.system.command", "from acprof.host.orchestrator import execute"),
    ],
)
def test_reverse_dependencies_are_rejected(module, source):
    assert violations_in_text(module, source)


def test_runtime_type_only_imports_are_excluded():
    source = "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from acprof.system.command import run_command\n"
    assert violations_in_text("acprof.analysis.report", source) == []


def test_shared_runner_has_no_monitor_host_exception():
    assert violations_in_text("acprof.monitors.common", "from acprof.system.command import run_command") == []
    assert violations_in_text("acprof.monitors.resource_usage", "from acprof.system.command import run_command") == []
    assert violations_in_text("acprof.monitors.common", "from acprof.host.command import run_command")


def test_actual_repository_does_not_reverse_import():
    assert scan_import_violations(Path(__file__).resolve().parents[2]) == []
