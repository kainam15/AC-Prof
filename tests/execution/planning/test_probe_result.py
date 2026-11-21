"""A successful process cannot substitute for current, matching probe evidence."""
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from acprof.events.progress import encode_event
from acprof.experiments.configuration import RunConfig
from acprof.host.largest_scale_probe import (
    PROBE_SUMMARY_NAME,
    create_probe_output_dir,
    write_probe_summary,
)
from acprof.tui.run.progress import RunProgressTracker
from acprof.tui.run.results import inspect_probe_result


@pytest.fixture
def probe_evidence(tmp_path):
    started = time.time_ns()
    config = replace(RunConfig.smoke("fixture/model"), output_dir=str(tmp_path))
    directory = create_probe_output_dir(tmp_path, config.model)
    path = directory / PROBE_SUMMARY_NAME
    payload = {
        "schema_version": 3, "status": "ok", "model": {"id": config.model},
        "resource": {"cpu_cores": 1, "mem_gb": 4, "gpu_mode": "off"},
        "input": {"planned_scale": 512, "effective_scale": 512},
        "timing": {"request_s": 2.0, "ready_plus_request_s": 7.0},
        "cold_start": {"total_s": 5.0},
    }
    write_probe_summary(path, payload)
    tracker = RunProgressTracker()
    tracker.feed(encode_event("probe_started", "largest_scale_probe", cpu=1, gpu="off",
                              memory_candidates=[4], input_scale=512))
    tracker.feed(encode_event("probe_finished", "largest_scale_probe", status="ok", cpu=1,
                              mem=4, gpu="off", input_scale=512, request_s=2.0,
                              ready_plus_request_s=7.0, cold_start_s=5.0))
    snapshot = tracker.feed(encode_event("probe_summary", path=str(path)))
    return snapshot, config, Path.cwd(), os.getpid(), started, path, payload


def test_current_probe_summary_matches_events_and_launch(probe_evidence):
    inspect_probe_result(*probe_evidence[:5])


@pytest.mark.parametrize("missing", ["terminal", "path"])
def test_zero_exit_without_terminal_or_summary_cannot_succeed(probe_evidence, missing):
    snapshot, *rest = probe_evidence[:5]
    snapshot = replace(snapshot, **({"probe_result": None} if missing == "terminal" else {"probe_summary": ""}))
    with pytest.raises(ValueError, match="without"):
        inspect_probe_result(snapshot, *rest)


@pytest.mark.parametrize("change", ["wrong_model", "wrong_pid", "old_file", "timing", "version"])
def test_unrelated_or_changed_probe_artifacts_are_rejected(probe_evidence, change):
    snapshot, config, project, pid, started, path, payload = probe_evidence
    if change == "wrong_pid":
        pid += 1
    elif change == "old_file":
        started = path.stat().st_mtime_ns + 1
    else:
        if change == "wrong_model":
            payload["model"]["id"] = "other/model"
        elif change == "timing":
            payload["timing"]["request_s"] = 99.0
        else:
            payload["schema_version"] = 99
        write_probe_summary(path, payload)
    with pytest.raises(ValueError, match="Probe completion event disagrees|Probe summary|unsupported largest-scale probe summary version"):
        inspect_probe_result(snapshot, config, project, pid, started)


def test_configured_probe_requires_persisted_configuration_identity(probe_evidence):
    snapshot, config, project, pid, started = probe_evidence[:5]
    with pytest.raises(ValueError, match="missing the requested inference configuration"):
        inspect_probe_result(snapshot, replace(config, confirm_inference="a" * 64), project, pid, started)


def test_cpu_set_must_match_the_launch_in_both_directions(probe_evidence):
    snapshot, config, project, pid, started, path, payload = probe_evidence
    restricted = replace(config, cpuset_cpus="3,1-2")
    with pytest.raises(ValueError, match="CPU set differs from the launch configuration"):
        inspect_probe_result(snapshot, restricted, project, pid, started)

    payload["resource"]["cpuset_cpus"] = "1-3"
    payload["memory_probe"] = {"attempts": [{"status": "ok", "mem_gb": 4, "cpu_affinity": ["1", "2-3"]}]}
    write_probe_summary(path, payload)
    inspect_probe_result(snapshot, restricted, project, pid, started)
    with pytest.raises(ValueError, match="CPU set differs from the launch configuration"):
        inspect_probe_result(snapshot, config, project, pid, started)


@pytest.mark.parametrize("affinities", [None, [], ["1", "invalid"], ["2-4"]])
def test_unverified_thread_masks_cannot_support_a_successful_probe(probe_evidence, affinities):
    snapshot, config, project, pid, started, path, payload = probe_evidence
    payload["resource"]["cpuset_cpus"] = "1-3"
    payload["memory_probe"] = {"attempts": [{"status": "ok", "mem_gb": 4, "cpu_affinity": affinities}]}
    write_probe_summary(path, payload)
    with pytest.raises(ValueError, match="CPU affinity"):
        inspect_probe_result(snapshot, replace(config, cpuset_cpus="1-3"), project, pid, started)


def test_saved_sha_does_not_replace_the_current_interactive_confirmation(probe_evidence):
    from acprof.artifacts import atomic_write_json
    from acprof.host.largest_scale_probe import PROBE_CONFIGURATION_NAME
    from acprof.tui.i18n import error_message, translate
    from tests.execution.operations.test_inference_configuration_review import frozen

    snapshot, config, project, pid, started, path, summary = probe_evidence
    configuration = frozen()
    digest = configuration["identity_sha256"]
    atomic_write_json(path.with_name(PROBE_CONFIGURATION_NAME), configuration)
    summary["model"].update(revision=configuration["model"]["revision"], runtime_backend=configuration["backend"]["id"])
    summary["inference_configuration_sha256"] = digest
    write_probe_summary(path, summary)
    with pytest.raises(ValueError, match="本次探测没有收到") as error:
        inspect_probe_result(snapshot, replace(config, confirm_inference=digest), project, pid, started)
    assert translate(error_message(error.value), "en") == (
        "This probe did not receive confirmation of an inference configuration SHA-256."
    )
