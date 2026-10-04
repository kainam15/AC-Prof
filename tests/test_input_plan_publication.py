"""Input plan publication preserves frozen payload bytes and complete files."""
import hashlib
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from acprof import artifacts
from acprof.host import input_plan
from acprof.host.detect import TaskInfo

# Written independently of the serializer under test, including escaped Unicode,
# a literal newline in a value, field order, native types, and the final newline.
EXPECTED_LF = br"""{
  "schema_version": 2,
  "model_id": "org/model",
  "task_family": "cv",
  "pipeline_tag": "image-classification",
  "workload": {
    "name": "caf\u00e9"
  },
  "model_constraints": {
    "max": 8
  },
  "scenario": {
    "type": "serial"
  },
  "entries": [
    {
      "input_scale": 1.5,
      "scale_label": "small",
      "input_metadata": {
        "enabled": true
      },
      "payload": {
        "text": "\u4e2d\u6587\ud83d\ude00\n",
        "values": [
          1,
          -0.0,
          null,
          false
        ]
      }
    }
  ]
}
"""
EXPECTED_SHA256 = {
    "\n": "c8e9ff6d4b887cf71bb2b864edc8cac67e9a090616ab2d50b0f39f3b0a484ae9",
    "\r\n": "faccf4111ee27ddb3989790a1a1a24283f028851bd2b6fa967b97a21062943dd",
}
OLD_PLAN = b'{"schema_version": 2, "entries": [], "old": true}\n'


@pytest.fixture
def task_info():
    return TaskInfo(
        "org/model", "image-classification", "cv",
        "transformers_pipeline", "transformers", "revision", "test",
    )


@pytest.fixture
def existing_plan(tmp_path):
    path = tmp_path / "input_scale_plan.json"
    path.write_bytes(OLD_PLAN)
    path.chmod(0o640)
    return path, stat.S_IMODE(path.stat().st_mode)


def _entries():
    return [{
        "input_scale": 1.5,
        "scale_label": "small",
        "input_metadata": {"enabled": True},
        "payload": {"text": "中文😀\n", "values": [1, -0.0, None, False]},
    }]


def _write_fixture(path, task_info, entries=None):
    return input_plan._write_scale_plan_file(
        str(path), task_info, _entries() if entries is None else entries,
        workload={"name": "café"}, model_constraints={"max": 8},
    )


def _assert_old_plan_preserved(path, mode):
    assert path.read_bytes() == OLD_PLAN
    assert stat.S_IMODE(path.stat().st_mode) == mode
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


@pytest.mark.parametrize("newline", ("\n", "\r\n"), ids=("lf", "crlf"))
def test_publication_preserves_exact_historical_bytes_and_digest(
    existing_plan, task_info, monkeypatch, newline,
):
    path, mode = existing_plan
    monkeypatch.setattr(input_plan.os, "linesep", newline)

    digest = _write_fixture(path, task_info)

    assert path.read_bytes() == EXPECTED_LF.replace(b"\n", newline.encode("ascii"))
    assert digest == EXPECTED_SHA256[newline]
    assert stat.S_IMODE(path.stat().st_mode) == mode
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []
    assert artifacts.read_input_scale_plan(path)["entries"] == _entries()


@pytest.mark.parametrize("newline", ("\n", "\r\n"), ids=("lf", "crlf"))
@pytest.mark.parametrize("limit_delta", (0, -1), ids=("exact_limit", "newline_exceeds_limit"))
def test_publication_size_limit_includes_platform_newlines(
    existing_plan, task_info, monkeypatch, newline, limit_delta,
):
    path, mode = existing_plan
    expected = EXPECTED_LF.replace(b"\n", newline.encode("ascii"))
    limit = len(expected) + limit_delta
    monkeypatch.setattr(input_plan.os, "linesep", newline)
    monkeypatch.setattr(input_plan, "MAX_JSON_ARTIFACT_BYTES", limit)
    monkeypatch.setattr(artifacts, "MAX_JSON_ARTIFACT_BYTES", limit)

    if limit_delta:
        with pytest.raises(ValueError, match="input scale plan.*4 MiB"):
            _write_fixture(path, task_info)
        _assert_old_plan_preserved(path, mode)
    else:
        assert _write_fixture(path, task_info) == EXPECTED_SHA256[newline]
        assert path.read_bytes() == expected
        assert artifacts.read_input_scale_plan(path)["entries"] == _entries()


@pytest.mark.parametrize(
    "invalid", ("unsupported", "circular", "nan", "positive_infinity", "negative_infinity"),
)
def test_serialization_failure_preserves_existing_plan(
    existing_plan, task_info, invalid,
):
    path, mode = existing_plan
    if invalid == "unsupported":
        value = object()
        error = TypeError
    elif invalid == "circular":
        value = []
        value.append(value)
        error = ValueError
    else:
        value = {"nan": float("nan"), "positive_infinity": float("inf"),
                 "negative_infinity": -float("inf")}[invalid]
        error = ValueError
    entries = _entries()
    entries[0]["payload"]["invalid"] = value

    with pytest.raises(error):
        _write_fixture(path, task_info, entries)
    _assert_old_plan_preserved(path, mode)


def test_oversized_plan_is_rejected_before_publication(existing_plan, task_info):
    path, mode = existing_plan
    entries = _entries()
    entries[0]["payload"]["text"] = "x" * artifacts.MAX_JSON_ARTIFACT_BYTES

    with pytest.raises(ValueError, match="input scale plan.*4 MiB"):
        _write_fixture(path, task_info, entries)
    _assert_old_plan_preserved(path, mode)


@pytest.mark.parametrize("failure", ("file_fsync", "replace"))
def test_publication_io_failure_preserves_existing_plan(
    existing_plan, task_info, monkeypatch, failure,
):
    path, mode = existing_plan

    def fail(*args):
        raise OSError(f"fixture {failure} failure")

    monkeypatch.setattr(artifacts.os, "fsync" if failure == "file_fsync" else "replace", fail)

    with pytest.raises(OSError, match=f"fixture {failure} failure"):
        _write_fixture(path, task_info)
    _assert_old_plan_preserved(path, mode)


@pytest.mark.skipif(artifacts.os.name != "posix", reason="directory fsync is POSIX-only")
def test_directory_fsync_failure_reports_error_after_complete_publication(
    existing_plan, task_info, monkeypatch,
):
    path, _ = existing_plan
    real_fsync = artifacts.os.fsync

    def fsync(fd):
        if stat.S_ISDIR(artifacts.os.fstat(fd).st_mode):
            raise OSError("fixture directory fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(input_plan.os, "linesep", "\n")
    monkeypatch.setattr(artifacts.os, "fsync", fsync)

    with pytest.raises(OSError, match="fixture directory fsync failure"):
        _write_fixture(path, task_info)
    assert path.read_bytes() == EXPECTED_LF
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []
    assert artifacts.read_input_scale_plan(path)["entries"] == _entries()


def _stub_generator(monkeypatch, generate):
    monkeypatch.setattr(input_plan, "_get_task_generator", lambda *args, **kwargs: SimpleNamespace(
        generate=generate,
        effective_input_scale=lambda scale, payload: scale,
        scale_label=lambda scale: "small",
    ))


@pytest.mark.parametrize("failure", ("generation", "serialization"))
def test_failed_planner_retains_existing_plan(
    existing_plan, task_info, monkeypatch, failure,
):
    path, mode = existing_plan

    def generate(scale):
        if failure == "generation":
            raise RuntimeError("fixture generation failure")
        return {"invalid": object()}

    _stub_generator(monkeypatch, generate)
    error = RuntimeError if failure == "generation" else TypeError
    with pytest.raises(error):
        input_plan.plan_input_scales(
            task_info, SimpleNamespace(), [1], [1], ["off"], 1,
            str(path.parent), input_scales="1",
        )
    _assert_old_plan_preserved(path, mode)


def test_successful_planner_replaces_existing_plan_with_matching_digest(
    existing_plan, task_info, monkeypatch,
):
    path, mode = existing_plan
    _stub_generator(monkeypatch, lambda scale: {"text": "new payload"})

    planned = input_plan.plan_input_scales(
        task_info, SimpleNamespace(), [1], [1], ["off"], 1,
        str(path.parent), input_scales="1",
    )

    assert Path(planned.plan_file) == path
    assert path.read_bytes() != OLD_PLAN
    assert planned.plan_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert artifacts.read_input_scale_plan(path)["entries"][0]["payload"] == {"text": "new payload"}
    assert stat.S_IMODE(path.stat().st_mode) == mode
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []
