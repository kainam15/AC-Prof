"""Protect explicit UI ownership and the single App worker/lifecycle boundary."""

from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

import pytest
from textual.message_pump import MessagePump

from acprof.tui.app import AcprofTui
from acprof.tui.configuration_controller import ConfigurationController
from acprof.tui.experiment_controller import ExperimentController
from acprof.tui.preflight_controller import PreflightController
from acprof.tui.process_controller import ProcessController
from acprof.tui.profile_controller import ProfileController
from acprof.tui.result_controller import ResultController
from acprof.tui.slash_controller import SlashController


@pytest.mark.parametrize(("attribute", "owner", "method"), (
    ("configuration", ConfigurationController, "collect_config"),
    ("configuration", ConfigurationController, "refresh_command_preview"),
    ("configuration", ConfigurationController, "save_ui_settings"),
    ("experiments", ExperimentController, "request_run"),
    ("experiments", ExperimentController, "request_probe"),
    ("preflight", PreflightController, "quick_check"),
    ("preflight", PreflightController, "preparation_event"),
    ("process", ProcessController, "execute_command"),
    ("process", ProcessController, "watch_failed_process"),
    ("process", ProcessController, "safe_process_callback"),
    ("results", ResultController, "execute_summary_read"),
    ("results", ResultController, "execute_report_read"),
    ("profiles", ProfileController, "profile_run_button"),
    ("slash", SlashController, "slash_command_submitted"),
))
def test_business_controller_has_one_explicit_owner(tmp_path, attribute, owner, method):
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    controller = getattr(app, attribute)
    assert isinstance(controller, owner)
    assert controller.app is app
    assert not issubclass(owner, MessagePump)
    assert owner not in AcprofTui.__mro__
    assert not hasattr(app, method)
    assert getattr(controller, method).__self__ is controller


def test_report_worker_uses_the_app_worker_manager(tmp_path):
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    token = Event()
    report = tmp_path / "report.json"
    worker = Mock()
    with patch.object(app, "run_worker", return_value=worker) as schedule:
        result = app.results.execute_report_read(report, token)
    assert result is worker
    operation = schedule.call_args.args[0]
    assert operation.func.__self__ is app.results
    assert operation.args == (report, token)
    assert schedule.call_args.kwargs["group"] == "report"
    assert schedule.call_args.kwargs["thread"] is True


def test_controller_state_is_isolated_between_apps(tmp_path):
    first = AcprofTui(settings_path=tmp_path / "first.json")
    second = AcprofTui(settings_path=tmp_path / "second.json")
    first.images.selected_ids.add("image-1")
    first.catalog.picker_open = True
    assert second.images.selected_ids == set()
    assert not second.catalog.picker_open
    assert first.lifecycle is not second.lifecycle


def test_app_retains_process_and_measurement_lifecycle():
    for name in ("launch", "process_finished", "stop_process_gracefully", "on_unmount"):
        assert name in AcprofTui.__dict__


def test_main_controller_does_not_regrow_into_giant_module():
    path = Path(__file__).resolve().parents[1] / "acprof/tui/app.py"
    assert len(path.read_text(encoding="utf-8").splitlines()) <= 1100
