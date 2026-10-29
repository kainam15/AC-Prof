"""Host and image-staged Transformers pins must stay distinct."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.container.load_policy import load_policy
from acprof.installation import resource_root
from acprof.runtime_profiles import PROFILES, locked_transformers_version


@pytest.fixture(autouse=True)
def clear_version_cache():
    locked_transformers_version.cache_clear()
    yield
    locked_transformers_version.cache_clear()


def test_host_transformers_pin_still_uses_project_resource_lock():
    environment = PROFILES["nlp-cu128"].environment
    assert locked_transformers_version(environment) == "4.57.6"


@pytest.mark.parametrize(
    ("task", "backend", "profile"),
    [
        ("text-ranking", "cross_encoder", "nlp-cu128"),
        ("image-classification", "transformers_pipeline", "cv-cpu"),
    ],
)
def test_container_reads_staged_lock_not_host_resource(
    tmp_path, monkeypatch, task, backend, profile,
):
    original_lock = resource_root() / PROFILES[profile].environment.requirements_lock
    image_lock = tmp_path / "requirements.lock"
    image_lock.write_bytes(original_lock.read_bytes())
    monkeypatch.setattr("acprof.installation.resource_root", lambda: tmp_path / "missing-source-tree")
    monkeypatch.setattr("acprof.container.load_policy.CONTAINER_REQUIREMENTS_LOCK", image_lock, raising=False)
    monkeypatch.setenv("ACPROF_DEPENDENCY_LOCK_SHA256", "image-build-fingerprint")
    monkeypatch.setenv("ACPROF_RUNTIME_PROFILE", profile)
    with patch.dict(sys.modules, {
        "torch": SimpleNamespace(float32="float32", float16="float16", bfloat16="bfloat16"),
        "transformers": SimpleNamespace(__version__="4.57.6"),
    }):
        policy = load_policy("fixture/model", task, backend, "cpu")
    assert policy["runtime_profile"] == profile


def test_container_missing_staged_lock_never_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr("acprof.container.load_policy.CONTAINER_REQUIREMENTS_LOCK", tmp_path / "missing.lock", raising=False)
    monkeypatch.setenv("ACPROF_DEPENDENCY_LOCK_SHA256", "image-build-fingerprint")
    monkeypatch.setenv("ACPROF_RUNTIME_PROFILE", "cv-cpu")
    with patch.dict(sys.modules, {
        "torch": SimpleNamespace(float32="float32"),
        "transformers": SimpleNamespace(__version__="4.57.6"),
    }), pytest.raises(FileNotFoundError, match="missing.lock"):
        load_policy("fixture/model", "image-classification", "transformers_pipeline", "cpu")
