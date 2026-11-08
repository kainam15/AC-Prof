"""Payload, resource availability and build work must never share a denominator."""
import base64
import json
import subprocess

import pytest

from acprof.events.image import ImageProgress
from acprof.host.images import operations
from acprof.host.images.errors import DockerPreparationError
from acprof.host.images.progress import DockerOutput


@pytest.fixture
def capture(monkeypatch):
    events = []
    monkeypatch.setattr("acprof.host.images.progress.emit_image_progress",
                        lambda component, image, phase, **fields: events.append({"phase": phase, **fields}))
    return events


def feed(output, identifier, status, current=None, total=None):
    output.feed(json.dumps({"id": identifier, "status": status,
                            "progressDetail": {"current": current, "total": total}}))


def test_concurrent_layers_duplicates_and_extraction_are_separate(capture):
    output = DockerOutput("platform", "fixture:image", "pulling")
    feed(output, "a", "Downloading", 40, 100)
    feed(output, "b", "Downloading", 20, 50)
    feed(output, "a", "Downloading", 40, 100)
    feed(output, "b", "Downloading", 50, 50)
    feed(output, "a", "Downloading", 100, 100)
    feed(output, "a", "Extracting", 1000, 1000)
    output.finish()
    last = capture[-1]
    assert last["transferred_bytes"] == 150
    assert last["known_bytes"] == 150
    assert last["unknown_items"] == 0
    assert last["activity"] == "extract"


def test_retry_resume_and_duplicate_countdown_keep_actual_payload(capture):
    output = DockerOutput("platform", "fixture:image", "pulling")
    feed(output, "a", "Downloading", 40, 100)
    feed(output, "a", "Retrying in 2 seconds")
    feed(output, "a", "Retrying in 1 second")
    # Moby resumes its file, but reports NEW bytes and remaining total in this attempt.
    feed(output, "a", "Downloading", 20, 60)
    feed(output, "a", "Downloading", 60, 60)
    feed(output, "a", "Download complete")
    output.finish()
    assert capture[-1]["transferred_bytes"] == 100
    assert capture[-1]["known_bytes"] == 100


def test_retry_redownload_counts_bytes_again_without_inflating_resource_size(capture):
    output = DockerOutput("platform", "fixture:image", "pulling")
    feed(output, "a", "Downloading", 40, 100)
    feed(output, "a", "Retrying in 1 second")
    feed(output, "a", "Downloading", 100, 100)
    output.finish()
    assert capture[-1]["transferred_bytes"] == 140
    assert capture[-1]["known_bytes"] == 100


def test_unknown_total_cache_and_missing_id_do_not_invent_percentage(capture):
    output = DockerOutput("platform", "fixture:image", "pulling")
    feed(output, "a", "Downloading", 40)
    feed(output, "b", "Already exists")
    feed(output, "b", "Already exists")
    feed(output, "", "Downloading", 900, 1000)
    output.finish()
    assert capture[-1]["transferred_bytes"] == 40
    assert capture[-1]["unknown_items"] == 2
    assert capture[-1]["cached_items"] == 1
    assert capture[-1]["current"] is None
    assert capture[-1]["total"] is None


def test_buildkit_digest_availability_and_work_units_are_not_network_bytes(capture):
    output = DockerOutput("environment", "fixture:image", "building")
    output.feed(json.dumps({"statuses": [
        {"id": "sha256:" + "a" * 64, "vertex": "v1", "current": 100, "total": 100},
        {"id": "sha256:" + "a" * 64, "vertex": "v2", "current": 100, "total": 100},
        {"id": "work", "name": "compiling", "current": 4, "total": 8},
    ], "vertexes": [{"digest": "v3", "name": "RUN install", "cached": True}]}))
    output.finish()
    assert capture[-1]["known_bytes"] == 100
    assert capture[-1]["transferred_bytes"] is None
    assert capture[-1]["reused_steps"] == 1
    assert capture[-1]["total"] is None


def test_unannounced_counter_regression_keeps_evidence_and_marks_partial(capture):
    output = DockerOutput("platform", "fixture:image", "pulling")
    for current in (40, 20, 40, 60):
        feed(output, "a", "Downloading", current, 100)
    output.finish()
    assert capture[-1]["transferred_bytes"] == 60
    assert capture[-1]["transfer_status"] == "partial"


def test_flat_buildx_statuses_do_not_disappear(capture):
    output = DockerOutput("platform", "fixture:image", "building")
    output.feed(json.dumps({"id": "layer", "vertex": "v", "name": "downloading",
                            "current": 20, "total": 50}))
    output.finish()
    assert capture[-1]["transferred_bytes"] == 20
    assert capture[-1]["current"] == 20


def test_interleaved_vertex_logs_keep_cumulative_dependency_payload(capture):
    output = DockerOutput("environment", "fixture:image", "building")
    def record(key, received, total):
        return "ACPROF_DEPENDENCY_PROGRESS " + json.dumps({
            "version": 1, "id": key, "transfer_id": key, "received": received,
            "current": received, "total": total, "cached": 0, "state": "downloading",
        }) + "\n"

    a, b = record("a", 40, 100), record("b", 20, None)
    for vertex, fragment in (("a", a[:50]), ("b", b[:50]), ("a", a[50:]), ("b", b[50:]), ("a", a)):
        output.feed(json.dumps({"logs": [{"vertex": vertex, "stream": 1,
                                          "data": base64.b64encode(fragment.encode()).decode()}]}))
    output.finish()
    assert capture[-1]["transferred_bytes"] == 60
    assert capture[-1]["known_bytes"] == 100
    assert capture[-1]["unknown_items"] == 1


@pytest.mark.parametrize("error", [TimeoutError("network timeout"), PermissionError("Docker permission denied"),
                                   subprocess.TimeoutExpired("docker", 1), KeyboardInterrupt()])
def test_operation_failure_flushes_throttled_progress(capture, monkeypatch, error):
    def run(command, **kwargs):
        kwargs["output_line"](json.dumps({"id": "a", "status": "Downloading", "progressDetail": {"current": 40, "total": 100}}))
        kwargs["output_line"](json.dumps({"id": "a", "status": "Downloading", "progressDetail": {"current": 60, "total": 100}}))
        raise error
    monkeypatch.setattr(operations.host_command, "run_command", run)
    with pytest.raises(KeyboardInterrupt if isinstance(error, KeyboardInterrupt) else DockerPreparationError):
        operations.run_docker(["docker", "pull", "fixture"], component="platform", image="fixture", operation="pull")
    assert capture[-1]["transferred_bytes"] == 60


def test_resource_size_is_discovered_later_without_losing_received_bytes(capture):
    output = DockerOutput("platform", "fixture:image", "pulling")
    feed(output, "a", "Downloading", 20)
    output.finish()
    assert capture[-1]["unknown_items"] == 1
    feed(output, "a", "Downloading", 30, 100)
    feed(output, "b", "Downloading", 10, 50)
    output.finish()
    assert capture[-1]["unknown_items"] == 0
    assert capture[-1]["known_bytes"] == 150
    assert capture[-1]["transferred_bytes"] == 40


@pytest.mark.parametrize("fields", [{"transferred_bytes": -1}, {"known_bytes": True},
                                   {"unknown_items": 1.5}, {"reused_steps": None},
                                   {"current": 20, "total": 10}, {"activity": "fake"}])
def test_invalid_accounting_evidence_is_rejected(fields):
    with pytest.raises(ValueError, match="invalid Docker"):
        ImageProgress("platform", "fixture", "pulling", **fields).validate()
