"""Hash checks must cover the exact bounded bytes decoded for post-hoc work."""
import hashlib
import json
from pathlib import Path

import pytest
from test_posthoc import TestPosthocProfile as _PosthocFixture

from acprof import artifacts
from acprof.host.posthoc import context as posthoc_context


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_expected_hash_accepts_original_bytes_without_rewriting(tmp_path, encoding):
    path = tmp_path / "plan.json"
    payload = {"model": "示例/model", "scale": 8.0}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding=encoding)
    content = path.read_bytes()
    expected = hashlib.sha256(content).hexdigest()
    assert artifacts.read_json_object(path, expected_sha256=expected) == payload
    assert path.read_bytes() == content


def test_hash_mismatch_is_rejected_before_decoding(tmp_path, monkeypatch):
    path = tmp_path / "plan.json"
    path.write_bytes(b'{"payload": "changed"}')

    def must_not_decode(_content):
        raise AssertionError("unverified bytes must not reach the JSON decoder")

    monkeypatch.setattr(artifacts, "loads_finite_json", must_not_decode)
    with pytest.raises(ValueError, match="hash does not match"):
        artifacts.read_json_object(path, label="input plan", expected_sha256="0" * 64)


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_matching_hash_does_not_bypass_finite_json_validation(tmp_path, number):
    path = tmp_path / "plan.json"
    content = ('{"value": ' + number + '}').encode()
    path.write_bytes(content)
    with pytest.raises(ValueError, match="non-finite"):
        artifacts.read_json_object(path, expected_sha256=hashlib.sha256(content).hexdigest())


def test_matching_hash_still_requires_an_object(tmp_path):
    path = tmp_path / "plan.json"
    path.write_bytes(b'[]')
    with pytest.raises(ValueError, match="must be an object"):
        artifacts.read_json_object(path, expected_sha256=hashlib.sha256(b'[]').hexdigest())


def test_matching_hash_still_obeys_byte_limit(tmp_path, monkeypatch):
    path = tmp_path / "plan.json"
    content = b'{}' + b' ' * 32
    path.write_bytes(content)
    monkeypatch.setattr(artifacts, "MAX_JSON_ARTIFACT_BYTES", 32)
    with pytest.raises(ValueError, match="read limit"):
        artifacts.read_json_object(path, expected_sha256=hashlib.sha256(content).hexdigest())


def test_bom_is_included_in_expected_byte_identity(tmp_path):
    path = tmp_path / "plan.json"
    path.write_bytes(b'\xef\xbb\xbf{}')
    with pytest.raises(ValueError, match="hash does not match"):
        artifacts.read_json_object(path, expected_sha256=hashlib.sha256(b'{}').hexdigest())


def test_posthoc_rejects_unverified_parse_even_if_path_is_restored(tmp_path, monkeypatch):
    _PosthocFixture()._write_fixture(tmp_path)
    path = tmp_path / posthoc_context.INPUT_SCALE_PLAN_NAME
    original = path.read_bytes()
    changed = json.loads(original)
    changed["entries"][0]["payload"]["text"] = "unverified replacement"
    path.write_text(json.dumps(changed), encoding="utf-8")
    reader = posthoc_context._load_json_object

    def restore_after_parse(candidate, label, **kwargs):
        value = reader(candidate, label, **kwargs)
        if candidate == path:
            path.write_bytes(original)
        return value

    monkeypatch.setattr(posthoc_context, "_load_json_object", restore_after_parse)
    with pytest.raises(posthoc_context.PosthocError, match="hash does not match"):
        posthoc_context.load_result_context(tmp_path)


@pytest.mark.parametrize("record_hash", [False, True])
def test_posthoc_parses_and_verifies_one_bounded_read(tmp_path, monkeypatch, record_hash):
    _PosthocFixture()._write_fixture(tmp_path)
    path = tmp_path / posthoc_context.INPUT_SCALE_PLAN_NAME
    if not record_hash:
        metadata_path = tmp_path / posthoc_context.STATIC_META_NAME
        metadata = json.loads(metadata_path.read_text())
        metadata.pop("input_scale_plan_sha256")
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    open_path = Path.open
    read_bytes = Path.read_bytes
    opened = []

    def track_open(candidate, *args, **kwargs):
        if candidate == path:
            opened.append(args[0] if args else kwargs.get("mode", "r"))
        return open_path(candidate, *args, **kwargs)

    def reject_second_read(candidate):
        if candidate == path:
            raise AssertionError("post-hoc must hash the bounded parse buffer, not reopen the path")
        return read_bytes(candidate)

    monkeypatch.setattr(Path, "open", track_open)
    monkeypatch.setattr(Path, "read_bytes", reject_second_read)
    context = posthoc_context.load_result_context(tmp_path)
    assert opened == ["rb"]
    assert context.input_scale_plan["entries"][0]["payload"] == {"text": "hello"}
