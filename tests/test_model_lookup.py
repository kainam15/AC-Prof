"""Model lookup failures retain actionable causes without terminating callers."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from acprof.cli.main import main
from acprof.host.detect import detect_task
from acprof.host.model_errors import ModelLookupError


def hub_error(kind, status):
    response = httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/api/models/asdf"))
    return kind("fixture Hub failure", response=response)


class ModelLookupTests(unittest.TestCase):
    def test_invalid_id_is_rejected_before_network_even_with_manual_task(self):
        with patch("huggingface_hub.HfApi.model_info", return_value=SimpleNamespace(
            sha="a" * 40, pipeline_tag="text-generation", library_name="transformers",
        )) as hub, patch(
            "acprof.host.detect._download_metadata",
        ) as download, self.assertRaises(ModelLookupError) as caught:
            detect_task("invalid model/id", override_tag="text-generation")
        self.assertEqual(caught.exception.reason_code, "invalid_model_id")
        self.assertIn("invalid model/id", str(caught.exception))
        hub.assert_not_called()
        download.assert_not_called()

    def test_permanent_lookup_errors_cannot_be_bypassed_by_manual_task(self):
        for failure, reason in (
            (hub_error(RepositoryNotFoundError, 401), "repository_unavailable"),
            (hub_error(RepositoryNotFoundError, 404), "repository_unavailable"),
            (hub_error(GatedRepoError, 403), "access_denied"),
            (hub_error(RevisionNotFoundError, 404), "revision_not_found"),
        ):
            for override in (None, "text-generation"):
                with self.subTest(reason=reason, override=override):
                    with patch("huggingface_hub.HfApi.model_info", side_effect=failure), patch(
                        "acprof.host.detect._download_metadata",
                    ) as download, self.assertRaises(ModelLookupError) as caught:
                        detect_task("asdf", revision="missing-revision", override_tag=override)
                    error = caught.exception
                    self.assertEqual(error.reason_code, reason)
                    self.assertFalse(error.retryable)
                    self.assertIs(error.__cause__, failure)
                    self.assertIn("asdf", str(error))
                    self.assertNotIn("--task-family", str(error))
                    download.assert_not_called()

    def test_network_and_service_errors_remain_distinct_from_missing_models(self):
        for failure, reason, retryable in (
            (httpx.ReadTimeout("fixture timeout"), "network_error", True),
            (httpx.ConnectError("fixture connection failure"), "network_error", True),
            (OfflineModeIsEnabled("fixture offline"), "offline", False),
            (hub_error(HfHubHTTPError, 429), "hub_unavailable", True),
            (hub_error(HfHubHTTPError, 503), "hub_unavailable", True),
            (RuntimeError("model not found timeout"), "lookup_failed", False),
        ):
            with self.subTest(reason=reason, failure=type(failure).__name__):
                with patch("huggingface_hub.HfApi.model_info", side_effect=failure), patch(
                    "acprof.host.detect._download_metadata", side_effect=failure,
                ), self.assertRaises(ModelLookupError) as caught:
                    detect_task("asdf")
                error = caught.exception
                self.assertEqual(error.reason_code, reason)
                self.assertEqual(error.retryable, retryable)
                self.assertIs(error.__cause__, failure)
                self.assertIn(type(failure).__name__, error.detail)

    def test_network_failure_can_still_use_pinned_cached_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "snapshots", "a" * 40, "config.json")
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"architectures": ["BertForMaskedLM"]}))
            with patch("huggingface_hub.HfApi.model_info", side_effect=httpx.ReadTimeout("fixture timeout")), patch(
                "acprof.host.detect._download_metadata", return_value=str(path),
            ):
                task = detect_task("asdf")
        self.assertEqual(task.model_revision, "a" * 40)
        self.assertEqual(task.pipeline_tag, "fill-mask")

    def test_accessible_model_without_task_is_not_reported_as_missing(self):
        hub = SimpleNamespace(sha="a" * 40, pipeline_tag=None, library_name="transformers")
        with patch("huggingface_hub.HfApi.model_info", return_value=hub):
            task = detect_task("asdf")
        self.assertEqual(task.model_id, "asdf")
        self.assertEqual(task.pipeline_tag, "unknown")

    def test_inspect_cli_reports_lookup_failure_without_traceback_or_probe(self):
        stderr = io.StringIO()
        with patch("acprof.host.env_utils.bootstrap_project_env"), patch(
            "huggingface_hub.HfApi.model_info", side_effect=hub_error(RepositoryNotFoundError, 404),
        ), patch("acprof.host.detect._download_metadata", side_effect=OSError("missing")), patch(
            "acprof.host.model_inspection.probe_model_contract",
        ) as probe, contextlib.redirect_stderr(stderr):
            result = main(["inspect", "asdf", "--probe", "full"])
        self.assertEqual(result, 1)
        self.assertIn("asdf", stderr.getvalue())
        self.assertIn("未找到", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertNotIn("--task-family", stderr.getvalue())
        probe.assert_not_called()
