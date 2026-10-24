"""Application ownership, configuration transport and failure contracts."""
from unittest.mock import Mock, patch

import pytest

from acprof.application.notifications import RunNotifications
from acprof.application.run import RunApplication, RunOutcome
from acprof.experiment import RunConfig, RunConfigError, build_run_command
from acprof.run_args import build_parser


def test_two_applications_do_not_share_state_notifications_or_terminal_logs():
    first, second = RunApplication(), RunApplication()
    assert first.notifications is not second.notifications
    first_state, second_state = Mock(), Mock()
    first_log = ("%1", "first.part", "first.log")
    second_log = ("%2", "second.part", "second.log")
    second.run_state, second.terminal_log = second_state, second_log
    stop = Mock(return_value=True)
    first.terminal_stop = stop

    def collect(*args, **kwargs):
        first.run_state, first.terminal_log = first_state, first_log
        return RunOutcome("ok", "first.csv", 1)

    with patch.object(first, "_execute", side_effect=collect):
        outcome = first.run(RunConfig.smoke("fixture/model"))

    assert outcome.final_csv == "first.csv"
    stop.assert_called_once_with(first_log)
    first_state.close.assert_called_once()
    second_state.close.assert_not_called()
    assert second.run_state is second_state
    assert second.terminal_log == second_log
    assert first.run_state is None and first.terminal_log is None


def test_cleanup_failure_releases_state_and_emits_failed_terminal_event():
    state = Mock()
    notifications = Mock(spec=RunNotifications)
    application = RunApplication(notifications=notifications,
                                 terminal_stop=Mock(side_effect=OSError("log unavailable")))

    def collect(*args, **kwargs):
        application.run_state = state
        application.terminal_log = ("%1", "partial", "final")
        return RunOutcome("ok")

    with patch.object(application, "_execute", side_effect=collect), patch(
        "acprof.application.run.emit_event"
    ) as emitted, pytest.raises(OSError, match="log unavailable"):
        application.run(RunConfig.smoke("fixture/model"))

    state.close.assert_called_once()
    notifications.deliver.assert_called_once()
    emitted.assert_called_once_with("run_finished", status="failed")


def test_application_rejects_invalid_config_without_system_exit():
    with pytest.raises(RunConfigError):
        RunApplication().run(RunConfig(model=""))


def test_cli_config_round_trip_retains_unmapped_and_sampling_options(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "--model", "fixture/model", "--matrix-seed", "73", "--matrix-order", "declared",
        "--latency-slo", "1.5", "--torch-profiler-repeat", "7", "--sampling-policy", "skip",
        "--raw-sample-limit", "321",
    ])
    config = RunConfig.from_namespace(args).validate(project_dir=tmp_path)
    restored = config.to_namespace()
    assert (restored.matrix_seed, restored.matrix_order, restored.torch_profiler_repeat) == (73, "declared", 7)
    assert restored.latency_slo == ["1.5"]
    assert (restored.sampling_policy, restored.raw_sample_limit) == ("skip", 321)
    command = build_run_command(config, project_dir=tmp_path)
    assert command[command.index("--matrix-seed") + 1] == "73"
    assert command[command.index("--sampling-policy") + 1] == "skip"
    assert command[command.index("--raw-sample-limit") + 1] == "321"


@pytest.mark.parametrize("values", [{"sampling_policy": "unknown"}, {"raw_sample_limit": 0},
                                   {"raw_sample_limit": -1}, {"raw_sample_limit": 1},
                                   {"raw_sample_limit": 2.5}, {"raw_sample_limit": True}])
def test_sampling_options_are_validated_by_shared_config(values):
    with pytest.raises(RunConfigError):
        RunConfig(model="fixture/model", **values).validate()


def test_absent_cli_transport_options_preserve_configured_environment(monkeypatch):
    from acprof.cli import run as run_cli

    monkeypatch.setenv("HF_DOWNLOAD_MODE", "mirror-only")
    monkeypatch.setenv("ACPROF_MAX_DOWNLOAD", "1GB")
    monkeypatch.setenv("ACPROF_MODEL_STORE", "/configured/store")
    options = build_parser().parse_args(["--model", "fixture/model"])
    with patch("acprof.application.run.bootstrap_project_env"), patch(
        "acprof.application.run.require_collection_host", side_effect=RuntimeError("preflight boundary")
    ), pytest.raises(RuntimeError, match="preflight boundary"):
        run_cli.main(args=options)
    import os
    assert os.environ["HF_DOWNLOAD_MODE"] == "mirror-only"
    assert os.environ["ACPROF_MAX_DOWNLOAD"] == "1GB"
    assert os.environ["ACPROF_MODEL_STORE"] == "/configured/store"
