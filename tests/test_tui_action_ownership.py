"""Keep Textual actions in focused owners without duplicating App handlers."""
from pathlib import Path

import pytest

from acprof.tui.app import AcprofTui
from acprof.tui.configuration_actions import ConfigurationActions
from acprof.tui.experiment_actions import ExperimentActions
from acprof.tui.preflight_actions import PreflightActions
from acprof.tui.profile_actions import ProfileActions
from acprof.tui.result_actions import ResultActions
from acprof.tui.slash_actions import SlashCommandActions

OWNER_METHODS = (
    (ConfigurationActions, "_collect_config"),
    (ConfigurationActions, "_refresh_command_preview"),
    (ConfigurationActions, "save_ui_settings"),
    (ExperimentActions, "action_request_run"),
    (ExperimentActions, "action_request_probe"),
    (PreflightActions, "action_quick_check"),
    (PreflightActions, "_preparation_event"),
    (ResultActions, "_execute_summary_read"),
    (ResultActions, "_execute_report_read"),
    (ProfileActions, "profile_run_button"),
    (SlashCommandActions, "slash_command_submitted"),
)


@pytest.mark.parametrize(("owner", "name"), OWNER_METHODS, ids=lambda item: getattr(item, "__name__", item))
def test_action_has_exact_single_owner(owner, name):
    assert name not in AcprofTui.__dict__
    assert getattr(AcprofTui, name) is owner.__dict__[name]


def test_app_retains_process_and_measurement_lifecycle():
    for name in ("_launch", "_execute_command", "_process_finished", "_stop_process_gracefully", "on_unmount"):
        assert name in AcprofTui.__dict__


def test_main_controller_does_not_regrow_into_giant_module():
    path = Path(__file__).resolve().parents[1] / "acprof/tui/app.py"
    assert len(path.read_text(encoding="utf-8").splitlines()) <= 1100
