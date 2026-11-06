"""Opt-in Hub lookup settings for tests that replace SDK network boundaries."""
import pytest


@pytest.fixture
def hf_lookup_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ACPROF_MODEL_SOURCE", "huggingface")
    monkeypatch.setenv("ACPROF_MODEL_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("HF_DOWNLOAD_MODE", "mirror-only")
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    monkeypatch.setenv("HF_FALLBACK_ENDPOINTS", "")
