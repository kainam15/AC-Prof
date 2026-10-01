import unittest
from unittest.mock import patch

from acprof.host.network_preflight import runtime_sources
from acprof.host.runtime_images import PROJECT_ROOT
from acprof.network_policy import (
    DownloadPolicyError,
    DownloadSource,
    enforce_download_budget,
    require_source_transition,
    summarize_downloads,
)
from acprof.runtime_profiles import PROFILES


class NetworkPreflightTests(unittest.TestCase):
    def test_route_totals_do_not_treat_unknown_as_zero(self):
        report = summarize_downloads([DownloadSource("model", "https://hf-mirror.com", 100),
                                     DownloadSource("oci", "https://ghcr.io/repo", 50),
                                     DownloadSource("python", "https://files.pythonhosted.org/a.whl", None)])
        self.assertEqual(report["direct_download_bytes"], 100)
        self.assertIsNone(report["proxy_download_bytes"])
        self.assertIsNone(report["expected_download_bytes"])
        with self.assertRaises(DownloadPolicyError):
            enforce_download_budget(report, "5GB")

    def test_domestic_to_proxy_fallback_requires_explicit_consent(self):
        with self.assertRaises(DownloadPolicyError):
            require_source_transition("https://docker.m.daocloud.io/repo", "https://ghcr.io/repo", allow_proxy=False)

    def test_local_runtime_hit_does_not_probe_registry_or_packages(self):
        profile = PROFILES["nlp-cpu"]
        with patch.dict("os.environ", {"ACPROF_RUNTIME_IMAGE_SOURCE": "auto"}, clear=True):
            sources, state = runtime_sources(profile, PROJECT_ROOT,
                inspect=lambda _: {"image_id": "sha256:" + "a" * 64},
                size_probe=lambda _: self.fail("local hit fetched artifact metadata"),
                manifest_probe=lambda _: self.fail("local hit queried registry"))
        self.assertEqual(sources, [])
        self.assertTrue(state["platform_local"])
        self.assertTrue(state["environment_local"])
        self.assertFalse(state["will_attempt_ghcr_pull"])

    def test_local_miss_reports_ghcr_and_compressed_upper_bound(self):
        with patch.dict("os.environ", {"ACPROF_RUNTIME_IMAGE_SOURCE": "auto"}, clear=True):
            sources, state = runtime_sources(PROFILES["nlp-cpu"], PROJECT_ROOT, inspect=lambda _: None,
                manifest_probe=lambda _: {"bytes": 100, "image_id": "sha256:" + "a" * 64})
        self.assertTrue(state["will_attempt_ghcr_pull"])
        self.assertEqual(summarize_downloads(sources)["expected_download_bytes"], 200)
