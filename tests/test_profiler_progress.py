import io
from contextlib import redirect_stderr
from unittest.mock import Mock

import pytest

from acprof.host.profiler_progress import report_profiler_completion


def test_counts_source_samples_once_regardless_of_repeat() -> None:
    callback = Mock()
    report_profiler_completion(
        callback,
        profiler="Nsys",
        profiles=[
            {"repeat": 10, "entries": [{"error": ""}, {"error": ""}]},
            {"repeat": 10, "entries": [{"error": "timeout"}, {"error": ""}]},
        ],
        elapsed_seconds=65.0,
    )
    callback.assert_called_once()
    event = callback.call_args.args[0]
    assert (event.profiler) == ("Nsys")
    assert (event.status) == ("partial")
    assert (event.total_samples) == (4)
    assert (event.error_samples) == (1)
    assert (event.elapsed_seconds) == (65.0)
    assert (event.detail) == ("timeout")

@pytest.mark.parametrize('profiles,status,total,failed', (([{'entries': [{'error': ''}]}], 'success', 1, 0), ([{'entries': []}], 'no_results', 0, 0), ([{'error': 'tool_missing', 'entries': []}], 'failed', 0, 0), ([{'error': 'tool_failed', 'entries': [{}]}], 'failed', 1, 1), ([{'entries': [{'error': 'OOM'}]}], 'failed', 1, 1), ([{'error': 'OOM', 'entries': [{'error': 'OOM'}, {}]}], 'partial', 2, 1)))
def test_reports_empty_and_failed_results_without_claiming_success(profiles, status, total, failed) -> None:
    callback = Mock()
    report_profiler_completion(
        callback, profiler="CPU Torch", profiles=profiles, elapsed_seconds=0.5
    )
    event = callback.call_args.args[0]
    assert (event.status) == (status)
    assert (event.total_samples) == (total)
    assert (event.error_samples) == (failed)

def test_disabled_callback_does_not_inspect_results() -> None:
    profiles = Mock()
    report_profiler_completion(
        None, profiler="NCU", profiles=profiles, elapsed_seconds=1.0
    )
    assert (profiles.mock_calls) == ([])

def test_callback_error_does_not_escape_or_log_its_contents() -> None:
    stderr = io.StringIO()
    with redirect_stderr(stderr):
        report_profiler_completion(
            Mock(side_effect=RuntimeError("private request details")),
            profiler="Massif",
            profiles=[{"entries": [{"error": ""}]}],
            elapsed_seconds=1.0,
        )
    assert ("RuntimeError") in (stderr.getvalue())
    assert ("private request details") not in (stderr.getvalue())
