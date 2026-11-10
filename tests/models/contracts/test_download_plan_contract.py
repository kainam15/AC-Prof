"""Freeze the public download-plan contract across its domain-layer move."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.container.download_model import prepare_repository_plan, verify_download
from acprof.models.download_plan import (
    plan_download,
    seal_plan,
    validate_plan,
    validate_plan_request,
)

# Written independently, then checked against the old planner at 5b1b48b8.
# Null metadata and absent keys deliberately remain different JSON shapes.
_NATIVE_PLAN_JSON = """
{
  "schema_version": 1,
  "model_id": "example/model",
  "model_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "task_family": "nlp",
  "backend": "transformers_pipeline",
  "adapter": "family-default",
  "requested_policy": "auto",
  "effective_policy": "selected",
  "reason": "native_transformers",
  "library_versions": {"transformers": "4.57.6"},
  "weights": [
    {
      "component": ".",
      "format": "safetensors",
      "variant": null,
      "files": ["model.safetensors"]
    }
  ],
  "files": [
    {"path": "config.json", "size": 2, "blob_id": null},
    {
      "path": "model.safetensors",
      "size": 7,
      "lfs_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    },
    {"path": "tokenizer.json", "size": null},
    {"path": "配置.json"}
  ],
  "excluded_files": [{"path": "pytorch_model.bin", "size": 11}],
  "selected_bytes": null,
  "plan_sha256": "144be4a3b3e3e445b4d4d5c3f3a64f0e4abe23590d517c9d6a2a46b03b17e939"
}
"""


@pytest.fixture
def native_plan():
    return plan_download(
        model_id="example/model",
        revision="a" * 40,
        family="nlp",
        backend="transformers_pipeline",
        files={
            "配置.json": {},
            "pytorch_model.bin": {"size": 11},
            "tokenizer.json": {"size": None},
            "model.safetensors": {"size": 7, "lfs_sha256": "b" * 64},
            "config.json": {"size": 2, "blob_id": None},
        },
        read_json={"config.json": {"model_type": "bert"}}.__getitem__,
        native_model_types={"bert"},
        library_versions={"transformers": "4.57.6"},
    )


def test_native_plan_preserves_complete_json_and_frozen_sha256(native_plan):
    expected = json.loads(_NATIVE_PLAN_JSON)
    assert json.dumps(native_plan, sort_keys=True, ensure_ascii=False) == json.dumps(
        expected, sort_keys=True, ensure_ascii=False,
    )


def test_declared_dependency_preserves_excluded_paths_and_unknown_size():
    info = SimpleNamespace(
        sha="a" * 40,
        siblings=[
            SimpleNamespace(rfilename="extra.bin", size=6),
            SimpleNamespace(rfilename="config.json"),
            SimpleNamespace(rfilename="README.md", size=4),
        ],
    )
    task = SimpleNamespace(
        task_family="nlp",
        runtime_backend="transformers_pipeline",
        model_download_policy="auto",
        model_adapter="family-default",
    )
    with (
        patch("huggingface_hub.HfApi.model_info", return_value=info),
        patch(
            "huggingface_hub.hf_hub_download",
            side_effect=AssertionError("metadata-only dependency must not download payloads"),
        ),
        patch("acprof.container.download_model.configure_hf_transport"),
        patch("acprof.container.download_model.hf_token_for_url", return_value=None),
    ):
        plan = prepare_repository_plan(
            "https://fixture.invalid",
            "example/dependency",
            "a" * 40,
            dependency={"allow_patterns": ["config.json"]},
            task=task,
            native_types=set(),
            library_versions={},
        )

    assert plan["reason"] == "declared_dependency"
    assert plan["excluded_files"] == ["README.md", "extra.bin"]
    assert plan["files"] == [
        {"path": "config.json", "size": None, "blob_id": None, "lfs_sha256": None},
    ]
    assert plan["selected_bytes"] is None
    assert validate_plan(plan) is None


def test_sealing_mutates_in_place_and_validation_is_read_only(native_plan):
    original = copy.deepcopy(native_plan)
    native_plan["plan_sha256"] = "stale"

    assert seal_plan(native_plan) is native_plan
    assert native_plan == original
    assert seal_plan(native_plan) is native_plan
    assert validate_plan(native_plan) is None
    assert validate_plan_request(native_plan, {
        "model_id": "example/model",
        "model_revision": "a" * 40,
        "source": "huggingface",
        "dependencies": [],
    }) is None
    assert native_plan == original


def test_payload_verification_updates_the_same_plan_object(tmp_path):
    (tmp_path / "payload.bin").write_bytes(b"abc")
    plan = plan_download(
        model_id="example/model",
        revision="a" * 40,
        family="nlp",
        backend="custom",
        policy="full",
        files={"payload.bin": {}},
        read_json=lambda _: {},
    )

    assert verify_download(tmp_path, plan) is plan
    assert plan["files"] == [{
        "path": "payload.bin",
        "size": 3,
        "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    }]
    assert plan["selected_bytes"] == 3
    assert plan["verification"] == "sha256"
    assert validate_plan(plan) is None


def test_domain_import_is_isolated_and_retired_container_entry_is_absent():
    code = """
import importlib
import sys

domain = importlib.import_module("acprof.models.download_plan")
assert callable(domain.plan_download)
blocked = (
    "acprof.host", "acprof.container", "torch", "tensorflow", "transformers",
    "diffusers", "sentence_transformers", "chronos", "huggingface_hub",
)
loaded = [
    name for name in sys.modules
    if any(name == root or name.startswith(root + ".") for root in blocked)
]
assert not loaded, loaded

try:
    importlib.import_module("acprof.container.model_files")
except ModuleNotFoundError as exc:
    assert exc.name == "acprof.container.model_files", exc
else:
    raise AssertionError("retired container model_files entry point is still importable")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
