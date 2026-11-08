"""Evidence boundaries for Docker download states and streamed progress."""
import base64
import json

import httpx
import pytest

from acprof.events.image import ImageProgress
from acprof.host.images import operations, progress
from acprof.host.images.errors import DockerPreparationError
from acprof.host.images.progress import DockerOutput

IMAGE = "registry.example/acprof:test"
DIGEST = "sha256:" + "a" * 64
STARTED = "2026-01-01T00:00:00Z"
COMPLETED = "2026-01-01T00:00:01Z"


@pytest.fixture
def events(monkeypatch):
    captured = []

    def emit(component, image, phase, **fields):
        captured.append(ImageProgress(component, image, phase, **fields).to_dict())

    monkeypatch.setattr(progress, "emit_image_progress", emit)
    return captured


def engine(output, identifier, status, **counts):
    output.feed(json.dumps({
        "id": identifier, "status": status, "progressDetail": counts,
    }))


class ScriptedStream(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    def __iter__(self):
        for chunk in self.chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk


def test_pull_lifecycle_distinguishes_waiting_unavailable_partial_and_reported(events):
    output = DockerOutput("platform", IMAGE, "pulling")
    output.start(detail="docker pull")
    assert events[0]["transfer_status"] == "waiting"
    assert events[0]["transferred_bytes"] is None
    assert events[-1]["transfer_status"] == "unavailable"
    assert events[-1]["transferred_bytes"] is None

    engine(output, "a", "Downloading", current=30, total=100)
    assert output.accounting.snapshot()["transfer_status"] == "partial"
    engine(output, "a", "Downloading", current=100, total=100)
    # Reaching a layer total does not complete the operation.
    assert output.accounting.snapshot()["transfer_status"] == "partial"
    output.finish(succeeded=True)
    assert events[-1]["transfer_status"] == "reported"
    assert events[-1]["transferred_bytes"] == 100
    assert "metadata" in events[-1]["uncovered_sources"]


@pytest.mark.parametrize(("records", "status", "transferred"), [
    ([], "unavailable", None),
    ([{"status": "Status: Image is up to date for registry.example/acprof:test"}],
     "unavailable", None),
    ([{"id": "a", "status": "Pull complete"}], "unavailable", None),
    ([{"id": "a", "status": "Already exists"}], "cached", 0),
    ([{"id": "a", "status": "Already exists"},
      {"id": "b", "status": "Waiting"}], "unavailable", None),
    ([{"id": "a", "status": "Downloading",
       "progressDetail": {"current": 0, "total": 100}}], "partial", 0),
])
def test_zero_requires_explicit_cache_or_numeric_evidence(events, records, status, transferred):
    output = DockerOutput("platform", IMAGE, "pulling")
    output.start()
    for record in records:
        output.feed(json.dumps(record))
    output.finish(succeeded=True)
    assert events[-1]["transfer_status"] == status
    assert events[-1]["transferred_bytes"] == transferred


def test_cache_savings_include_only_reported_sizes(events):
    output = DockerOutput("platform", IMAGE, "pulling")
    engine(output, "a", "Already exists", total=75)
    engine(output, "b", "Already exists")
    engine(output, "a", "Already exists", total=75)
    output.finish(succeeded=True)
    assert events[-1]["transfer_status"] == "cached"
    assert events[-1]["transferred_bytes"] == 0
    assert events[-1]["cached_bytes"] == 75
    assert events[-1]["cached_items"] == 2
    assert events[-1]["unknown_items"] == 1


@pytest.mark.parametrize("phase", ["pulling", "building"])
def test_failure_or_build_cache_does_not_prove_zero_download(events, phase):
    output = DockerOutput("platform", IMAGE, phase)
    if phase == "pulling":
        engine(output, "a", "Already exists", total=100)
    else:
        output.feed(json.dumps({
            "vertexes": [{"digest": "v", "name": "RUN install", "cached": True}],
        }))
    output.finish(succeeded=phase == "building")
    assert events[-1]["transfer_status"] == "unavailable"
    assert events[-1]["transferred_bytes"] is None
    assert events[-1]["reused_steps"] == (1 if phase == "building" else 0)


@pytest.mark.parametrize("tail", [
    {"id": "b", "status": "Pull complete"},
    {"id": "b", "status": "Downloading", "progressDetail": {"current": 20}},
    {"id": "b", "status": "Downloading", "progressDetail": {"current": 20, "total": 50}},
])
def test_success_with_unreported_or_unfinished_layer_keeps_partial_evidence(events, tail):
    output = DockerOutput("platform", IMAGE, "pulling")
    engine(output, "a", "Downloading", current=100, total=100)
    output.feed(json.dumps(tail))
    output.finish(succeeded=True)
    assert events[-1]["transfer_status"] == "partial"
    assert events[-1]["transferred_bytes"] == 100 + tail.get("progressDetail", {}).get("current", 0)


@pytest.mark.parametrize(("name", "activity"), [
    ("resolve image config for docker-image://docker.io/docker/dockerfile:1", "frontend"),
    ("[internal] load metadata for ghcr.io/example/base:latest", "metadata"),
])
def test_metadata_status_bytes_are_excluded_from_payload_and_resource_totals(events, name, activity):
    output = DockerOutput("environment", IMAGE, "building")
    output.feed(json.dumps({
        "vertexes": [{"digest": "metadata", "name": name, "started": STARTED}],
        "statuses": [{"id": DIGEST, "vertex": "metadata", "name": "downloading",
                      "current": 500, "total": 500}],
    }))
    output.finish(succeeded=True)
    assert events[-1]["activity"] == activity
    assert events[-1]["transferred_bytes"] is None
    assert events[-1]["known_bytes"] == 0
    assert events[-1]["transfer_status"] == "unavailable"
    assert events[-1]["source_image"] in name
    assert events[-1]["network_owner"] == "buildkit"


@pytest.mark.parametrize("parent", [{"vertex": "unknown"}, {}])
def test_unrelated_status_does_not_inherit_previous_frontend_metadata_or_source(events, parent):
    output = DockerOutput("environment", IMAGE, "building")
    output.feed(json.dumps({"vertexes": [{
        "digest": "frontend",
        "name": "resolve image config for docker-image://docker.io/docker/dockerfile:1",
        "started": STARTED,
    }]}))
    output.feed(json.dumps({
        "id": DIGEST, "name": "downloading", "current": 25, "total": 100, **parent,
    }))
    # Completion of the old metadata step must not restore its activity or source.
    output.feed(json.dumps({"digest": "frontend", "completed": COMPLETED}))
    output.finish(succeeded=True)
    assert events[-1]["transferred_bytes"] == 25
    assert events[-1]["known_bytes"] == 100
    assert events[-1]["activity"] == "download"
    assert events[-1]["source_image"] == ""
    assert events[-1]["source_registry"] == ""
    assert "docker/dockerfile" not in events[-1]["detail"]


def test_metadata_status_uses_its_parent_after_another_build_step_becomes_active(events):
    output = DockerOutput("environment", IMAGE, "building")
    output.feed(json.dumps({"vertexes": [
        {"digest": "metadata", "name": "[internal] load metadata for ghcr.io/example/base:latest",
         "started": STARTED},
        {"digest": "build", "name": "RUN install", "started": STARTED},
    ]}))
    output.feed(json.dumps({
        "id": DIGEST, "vertex": "metadata", "name": "downloading", "current": 25, "total": 100,
    }))
    output.finish(succeeded=True)
    assert events[-1]["transferred_bytes"] is None
    assert events[-1]["known_bytes"] == 0
    assert events[-1]["activity"] == "metadata"
    assert events[-1]["source_image"] == "ghcr.io/example/base:latest"


def test_null_optional_batches_do_not_make_complete_payload_partial(events):
    output = DockerOutput("platform", IMAGE, "pulling")
    engine(output, "a", "Downloading", current=100, total=100)
    output.feed(json.dumps({"vertexes": None, "statuses": None, "logs": None, "warnings": None}))
    output.finish(succeeded=True)
    assert events[-1]["transfer_status"] == "reported"
    assert "incomplete_stream" not in events[-1]["uncovered_sources"]


@pytest.mark.parametrize("malformed", [False, 0, "", {}, [None, 7, []]])
def test_malformed_batch_keeps_prior_payload_but_marks_missing_coverage(events, malformed):
    output = DockerOutput("platform", IMAGE, "pulling")
    engine(output, "a", "Downloading", current=100, total=100)
    output.feed(json.dumps({"statuses": malformed}))
    output.finish(succeeded=True)
    assert events[-1]["transferred_bytes"] == 100
    assert events[-1]["transfer_status"] == "partial"
    assert "incomplete_stream" in events[-1]["uncovered_sources"]


def test_bad_batch_item_does_not_discard_later_valid_status(events):
    output = DockerOutput("environment", IMAGE, "building")
    output.feed(json.dumps({"statuses": [
        None, {"id": DIGEST, "vertex": "v", "name": "downloading", "current": 25, "total": 100},
    ]}))
    output.finish(succeeded=True)
    assert events[-1]["transferred_bytes"] == 25
    assert "incomplete_stream" in events[-1]["uncovered_sources"]


@pytest.mark.parametrize("identity", ["digest", "id"])
def test_sparse_flat_vertex_completion_retains_name_and_cache_evidence(events, identity):
    output = DockerOutput("environment", IMAGE, "building")
    output.feed(json.dumps({identity: "v", "name": "RUN install", "started": STARTED}))
    output.feed(json.dumps({identity: "v", "completed": COMPLETED, "cached": True}))
    output.finish(succeeded=True)
    assert events[-1]["reused_steps"] == 1
    assert events[-1]["detail"] == "RUN install"
    assert "incomplete_stream" not in events[-1]["uncovered_sources"]


@pytest.mark.parametrize("vertex", [{"vertex": "v"}, {}])
@pytest.mark.parametrize("counts", [{"current": 25, "total": 100}, {"total": 100}])
def test_sparse_flat_content_status_keeps_size_without_inventing_network_payload(
    events, vertex, counts,
):
    output = DockerOutput("environment", IMAGE, "building")
    output.feed(json.dumps({"id": DIGEST, "timestamp": STARTED, **vertex, **counts}))
    output.finish(succeeded=True)
    assert events[-1]["known_bytes"] == 100
    assert events[-1]["transferred_bytes"] is None
    assert "incomplete_stream" not in events[-1]["uncovered_sources"]


def test_flat_named_status_without_optional_vertex_reports_payload(events):
    output = DockerOutput("environment", IMAGE, "building")
    output.feed(json.dumps({
        "id": "layer", "name": "downloading", "current": 25, "total": 100,
        "timestamp": STARTED,
    }))
    output.finish(succeeded=True)
    assert events[-1]["transferred_bytes"] == 25
    assert events[-1]["transfer_status"] == "partial"


def test_pre_2024_pretty_rawjson_retains_step_and_log_across_lines(events, capsys):
    output = DockerOutput("environment", IMAGE, "building")
    records = [
        {"vertexes": [{"digest": "v", "name": "RUN install", "started": STARTED}]},
        {"logs": [{"vertex": "v", "stream": 1,
                   "data": base64.b64encode(b"dependency resolver ready\n").decode()}]},
    ]
    for record in records:
        for line in json.dumps(record, indent=2).splitlines():
            output.feed(line)
    output.finish(succeeded=True)
    assert events[-1]["detail"] == "RUN install"
    assert "incomplete_stream" not in events[-1]["uncovered_sources"]
    assert "dependency resolver ready" in capsys.readouterr().out


@pytest.mark.parametrize("units", ["items", "layers"])
def test_non_byte_engine_units_are_not_counted_as_download(events, units):
    output = DockerOutput("platform", IMAGE, "pulling")
    engine(output, "a", "Downloading", current=5, total=10, units=units)
    output.finish(succeeded=True)
    assert events[-1]["transferred_bytes"] is None
    assert events[-1]["known_bytes"] == 0
    assert events[-1]["current"] is None
    assert "incomplete_stream" in events[-1]["uncovered_sources"]


def test_engine29_extraction_seconds_preserve_activity_without_downgrading_payload(events):
    # Moby docker-v29.4.0 daemon/containerd/progress.go reports extraction elapsed seconds.
    output = DockerOutput("platform", IMAGE, "pulling")
    engine(output, "a", "Downloading", current=100, total=100)
    engine(output, "a", "Extracting", current=7, units="s")
    output.finish(succeeded=True)
    assert events[-1]["activity"] == "extract"
    assert events[-1]["current"] is None
    assert events[-1]["total"] is None
    assert events[-1]["transferred_bytes"] == 100
    assert events[-1]["transfer_status"] == "reported"
    assert "incomplete_stream" not in events[-1]["uncovered_sources"]


def test_engine29_byte_less_completion_does_not_fill_missing_download_bytes(events):
    output = DockerOutput("platform", IMAGE, "pulling")
    engine(output, "a", "Pulling fs layer")
    engine(output, "a", "Downloading", current=30, total=100)
    engine(output, "a", "Download complete", hidecounts=True)
    engine(output, "a", "Pull complete", hidecounts=True)
    output.finish(succeeded=True)
    assert events[-1]["transferred_bytes"] == 30
    assert events[-1]["transfer_status"] == "partial"


def test_small_engine_line_is_delivered_before_requesting_the_next_network_chunk():
    line = '{"status":"Pulling fs layer","id":"a"}'
    response = httpx.Response(200, stream=ScriptedStream([
        (line + "\n").encode(),
        AssertionError("the first progress event must be yielded before the next read"),
    ]))
    lines = operations._response_lines(response)
    try:
        assert next(lines) == line
    finally:
        lines.close()
        response.close()


def test_engine_lines_preserve_split_utf8_and_final_unterminated_line():
    first = '{"status":"正在下载","id":"a"}'
    encoded = (first + "\n" + '{"status":"Pull complete","id":"a"}').encode()
    split = encoded.index("正".encode()) + 1
    response = httpx.Response(200, stream=ScriptedStream([encoded[:split], encoded[split:]]))
    try:
        assert list(operations._response_lines(response)) == [
            first, '{"status":"Pull complete","id":"a"}',
        ]
    finally:
        response.close()


@pytest.mark.parametrize("failure", [
    httpx.ReadError("daemon stream interrupted"),
    KeyboardInterrupt(),
    b'{"errorDetail":{"message":"connection reset"}}\n',
])
def test_interrupted_engine_stream_preserves_payload_and_next_operation_is_distinct(
    events, monkeypatch, failure,
):
    streams = iter([
        ScriptedStream([
            b'{"id":"a","status":"Downloading","progressDetail":{"current":40,"total":100}}\n',
            failure,
        ]),
        ScriptedStream([
            b'{"id":"a","status":"Downloading","progressDetail":{"current":60,"total":60}}\n',
        ]),
    ])
    monkeypatch.setattr(operations, "_anonymous_socket", lambda: "/fixture/docker.sock")
    def respond(request):
        if request.url.path == "/info":
            return httpx.Response(200, json={"Driver": "overlay2", "DriverStatus": []})
        return httpx.Response(200, stream=next(streams))

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(operations.httpx, "HTTPTransport", lambda **kwargs: transport)
    expected_error = KeyboardInterrupt if isinstance(failure, KeyboardInterrupt) else DockerPreparationError
    with pytest.raises(expected_error):
        operations.pull_image(IMAGE, component="platform")
    failed = events[-1]
    assert failed["transferred_bytes"] == 40
    assert failed["transfer_status"] == "partial"
    assert "incomplete_stream" in failed["uncovered_sources"]
    if isinstance(failure, KeyboardInterrupt):
        assert failed["phase"] == "cancelled"

    operations.pull_image(IMAGE, component="platform")
    recovered = events[-1]
    assert recovered["operation_id"] != failed["operation_id"]
    assert recovered["transfer_status"] == "reported"
    assert recovered["transferred_bytes"] == 60
    latest_by_operation = {event["operation_id"]: event for event in events}
    assert sum(event["transferred_bytes"] for event in latest_by_operation.values()) == 100


@pytest.mark.parametrize(("daemon_info", "expected_bytes", "expected_status"), [
    ({"Driver": "overlay2", "DriverStatus": [],
      "HttpProxy": "http://fixture:private-daemon-setting@proxy.invalid"}, 100, "reported"),
    ({"Driver": "overlayfs", "DriverStatus": [
        ["driver-type", "io.containerd.snapshotter.v1"],
    ]}, 60, "partial"),
    ({}, 60, "partial"),
    ({"Driver": []}, 60, "partial"),
    ("unavailable", 60, "partial"),
    ("interrupted", 60, "partial"),
])
def test_engine_counter_origin_is_checked_on_same_daemon_without_exposing_settings(
    events, monkeypatch, capsys, daemon_info, expected_bytes, expected_status,
):
    requested = []

    def respond(request):
        requested.append((request.method, request.url.host, request.url.path))
        timeout = request.extensions["timeout"]
        assert timeout["connect"] == 15.0
        if request.url.path == "/info":
            assert timeout["read"] == 15.0
            if daemon_info == "unavailable":
                return httpx.Response(503, json={"message": "daemon info unavailable"})
            if daemon_info == "interrupted":
                raise httpx.ReadError("daemon info stream interrupted", request=request)
            return httpx.Response(200, json=daemon_info)
        assert request.url.path == "/images/create"
        assert timeout["read"] is None
        return httpx.Response(200, content=(
            b'{"id":"a","status":"Downloading","progressDetail":{"current":40,"total":100}}\n'
            b'{"id":"a","status":"Downloading","progressDetail":{"current":100,"total":100}}\n'
        ))

    def transport(**kwargs):
        assert kwargs["uds"] == "/fixture/docker.sock"
        return httpx.MockTransport(respond)

    monkeypatch.setattr(operations, "_anonymous_socket", lambda: "/fixture/docker.sock")
    monkeypatch.setattr(operations.httpx, "HTTPTransport", transport)
    operations.pull_image(IMAGE, component="platform")
    assert requested == [("GET", "docker", "/info"), ("POST", "docker", "/images/create")]
    assert events[-1]["transferred_bytes"] == expected_bytes
    assert events[-1]["transfer_status"] == expected_status
    assert ("resume_baseline" in events[-1]["uncovered_sources"]) == (expected_status == "partial")
    assert "private-daemon-setting" not in capsys.readouterr().out
    assert "private-daemon-setting" not in json.dumps(events)


@pytest.mark.parametrize(("samples", "expected_bytes", "expected_status"), [
    ([100], None, "unavailable"),
    ([40, 40], None, "unavailable"),
    ([40, 70, 20, 50], 60, "partial"),
    ([40, 70, "Retrying in 1 second", 20, 50], 60, "partial"),
])
def test_content_offsets_count_only_growth_after_each_observed_baseline(
    events, samples, expected_bytes, expected_status,
):
    output = DockerOutput("platform", IMAGE, "pulling")
    output.accounting.engine_offsets = True
    for sample in samples:
        if isinstance(sample, str):
            engine(output, "a", sample)
        else:
            engine(output, "a", "Downloading", current=sample, total=100)
    output.finish(succeeded=True)
    assert events[-1]["activity"] == "content"
    assert events[-1]["transferred_bytes"] == expected_bytes
    assert events[-1]["transfer_status"] == expected_status
    assert "resume_baseline" in events[-1]["uncovered_sources"]


def test_containerd_cache_evidence_does_not_require_a_download_baseline(events):
    output = DockerOutput("platform", IMAGE, "pulling")
    output.accounting.engine_offsets = True
    engine(output, "a", "Already exists")
    output.finish(succeeded=True)
    assert events[-1]["transfer_status"] == "cached"
    assert events[-1]["transferred_bytes"] == 0
    assert "resume_baseline" not in events[-1]["uncovered_sources"]
