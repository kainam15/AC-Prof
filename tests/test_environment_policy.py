"""Regression contracts for platform identity, policy and dataset separation."""
import csv
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from acprof.capabilities import CapabilityReport, apply_collection_result, measurement_report
from acprof.platform import (
    Environment,
    capability_matrix,
    collection_policy_error,
    detect_environment,
    recorded_identity,
)

NATIVE = Environment("native_linux", "Linux", "6.8.0-generic")
WSL = Environment("wsl2", "Linux", "6.6-microsoft-standard-WSL2")


class TestEnvironmentPolicy:
    def detect(self, *, system="Linux", kernel="6.8.0-generic", proc="", paths=(), env=None):
        return detect_environment(system=system, release=kernel, version="test", machine="x86_64",
                                  environ=env or {}, read_text=lambda _: proc, exists=lambda p: p in paths)

    def test_native_wsl2_custom_kernel_container_and_unknown(self):
        assert (self.detect().environment) == ("native_linux")
        for kwargs in ({"kernel": "6.6.87.2-microsoft-standard-WSL2"},
                       {"proc": "Linux microsoft-standard-WSL2"},
                       {"kernel": "custom", "paths": ("/run/WSL",)}):
            result = self.detect(**kwargs)
            assert (result.environment) == ("wsl2")
            assert not (result.native)
            assert (result.collection_tier) == ("partial")
            assert (result.wsl["generation"]) == (2)
        assert (self.detect(paths=("/.dockerenv",)).environment) == ("container_host")
        assert (self.detect(kernel="4.4-Microsoft").environment) == ("unknown")
        assert (self.detect(env={"WSL_DISTRO_NAME": "Ubuntu"}).environment) == ("unknown")
        assert (self.detect(system="Windows", env={"WSL_DISTRO_NAME": "Ubuntu"}).environment) == ("unknown")

    def test_missing_and_conflicting_history_never_becomes_native(self):
        for payload in ({}, {"environment": "ubuntu24.04"}, {"platform": {"environment": "native_linux"}},
                        {**WSL.metadata(), "comparability_class": "native_linux"},
                        {**NATIVE.metadata(), "collection_tier": "partial"}):
            assert (recorded_identity(payload)["environment_class"]) == ("unknown")
        assert (recorded_identity(WSL.metadata())["comparability_class"]) == ("wsl2")

    def test_identity_is_json_stable_for_resume(self):
        for environment in (NATIVE, WSL, self.detect(kernel="6.6-microsoft-standard-WSL2")):
            assert (json.loads(json.dumps(environment.metadata()))) == (environment.metadata())

    def test_matrix_keeps_support_separate_from_actual_evidence(self):
        matrix = capability_matrix(WSL)
        assert (matrix["latency"]) == ("supported")
        assert (matrix["memory"]) == ("supported")
        for name in ("nvml", "cgroup", "gpu_memory", "cpu_topology", "cold_start", "affinity"):
            assert (matrix[name]) == ("partial")
        for name in ("rapl", "dram_energy", "pmu", "cycles", "ipc", "host_energy"):
            assert (matrix[name]) == ("unsupported")
        report = measurement_report("basic", gpu_modes=["on"], environment=WSL)
        assert (report.measurement["cpu_energy"].status.value) == ("unsupported")
        # Even malicious/fabricated numeric rows cannot promote unsupported capability.
        apply_collection_result(report, [{"status": "ok", "gpu_mode": "on", "cpu_energy_total_j": 0,
                                          "vcpu_energy_total_j": 0}])
        assert (report.measurement["cpu_energy"].status.value) == ("unsupported")
        assert not (report.to_dict()["full_profile_complete"])
        assert (CapabilityReport.from_dict(report.to_dict()).to_dict()) == (report.to_dict())

    def test_legacy_capability_reports_keep_unknown_identity(self):
        for version in (1, 2):
            report = CapabilityReport.from_dict({"schema_version": version, "profiling_mode": "full"})
            assert (report.to_dict()["comparability_class"]) == ("unknown")

    def test_wsl_basic_and_functional_probes_allowed_but_full_not_downgraded(self):
        assert (collection_policy_error(WSL)) == ("")
        assert (collection_policy_error(NATIVE, profiling_mode="full")) == ("")
        for options in ({"profiling_mode": "full"}, {"compute_tool": "torch"},
                        {"execution_tool": "nsys"}, {"dram_energy": "required"}):
            assert ("WSL2 / PARTIAL") in (collection_policy_error(WSL, **options))

    def test_native_probes_are_never_executed_under_wsl(self):
        from acprof.host import preflight
        with patch.object(preflight, "detect_environment", return_value=WSL), patch(
            "acprof.monitors.energy_cpu.detect_cpu_power_source", side_effect=AssertionError,
        ), patch("acprof.monitors.perf_mips.resolve_perf_command_prefix", side_effect=AssertionError):
            assert (preflight.probe_cpu_energy().status.value) == ("unsupported")
            assert (preflight.probe_perf_instructions().status.value) == ("unsupported")

    def test_resume_refuses_unknown_and_foreign_provenance(self):
        from acprof.host.preflight import require_result_environment
        with tempfile.TemporaryDirectory() as directory, patch(
            "acprof.host.preflight.detect_environment", return_value=WSL,
        ):
            path = Path(directory) / "static_meta.json"
            for payload in ({}, NATIVE.metadata()):
                path.write_text(json.dumps(payload))
                with pytest.raises(ValueError, match="Refusing to mix"):
                    require_result_environment(directory)
            path.write_text(json.dumps(WSL.metadata()))
            require_result_environment(directory)

    def test_resume_rejects_malformed_or_oversized_provenance(self):
        from acprof.host.preflight import require_result_environment
        with tempfile.TemporaryDirectory() as directory, patch(
            "acprof.host.preflight.detect_environment", return_value=WSL,
        ):
            path = Path(directory) / "static_meta.json"
            path.write_text(
                json.dumps(WSL.metadata())[:-1] + ', "corrupt_metric": NaN}',
                encoding="utf-8",
            )
            with pytest.raises(ValueError) as malformed:
                require_result_environment(directory)
            assert "Refusing to mix" in str(malformed.value)
            assert "invalid static metadata JSON" in str(malformed.value)

            path.write_text(
                json.dumps({**WSL.metadata(), "padding": "x" * (4 * 1024 * 1024)}),
                encoding="utf-8",
            )
            with pytest.raises(ValueError) as oversized:
                require_result_environment(directory)
            assert "Refusing to mix" in str(oversized.value)
            assert "static metadata exceeds the 4 MiB read limit" in str(oversized.value)

    def test_csv_merge_refuses_cross_environment_even_with_disjoint_keys(self):
        from acprof.result_csv import ResultValidationError, merge_result_csvs
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i, environment in enumerate(("native_linux", "wsl2")):
                path = Path(directory) / f"case{i}.csv"
                row = dict(cpu_cores=i + 1, mem_cap_gb=4, gpu_mode="off", input_scale=1,
                           warmup=0, repeat_idx=0, status="ok", error="", environment_class=environment)
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, row.keys())
                    writer.writeheader()
                    writer.writerow(row)
                paths.append(str(path))
            with pytest.raises(ResultValidationError, match="cannot merge result environments"):
                merge_result_csvs(paths, str(Path(directory) / "merged.csv"))
            assert not ((Path(directory) / "merged.csv").exists())

    def test_optional_nvml_query_failure_preserves_memory_and_real_zero(self):
        import pynvml

        from acprof.monitors import resource_readers, resource_usage
        with patch("acprof.platform.detect_environment", return_value=WSL), patch.object(
            resource_readers, "_resolve_container_metric_readers", return_value=resource_readers._ContainerReaders(),
        ):
            monitor = resource_usage.ResourceUsageMonitor()
        monitor._gpu_handle = "gpu"
        nvml = Mock(NVMLError_NotSupported=pynvml.NVMLError_NotSupported)
        nvml.nvmlDeviceGetUtilizationRates.side_effect = pynvml.NVMLError_NotSupported()
        nvml.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(used=0, total=8192)
        nvml.nvmlDeviceGetClockInfo.return_value = 1000
        nvml.nvmlDeviceGetPerformanceState.return_value = 0
        nvml.nvmlDeviceGetTemperature.return_value = 40
        with patch.object(resource_usage, "pynvml", nvml):
            sample = monitor._read_sample(1.0)
            assert (sample.gpu_util_pct) is None
            assert (sample.gpu_mem_used_bytes) == (0)
            assert (sample.gpu_mem_total_bytes) == (8192)
            assert (sample.cpu_freq_avg_hz) is None
            assert (monitor._runtime_error) == ("")
            nvml.nvmlDeviceGetUtilizationRates.side_effect = ValueError("ordinary bug")
            monitor._read_sample(2.0)
            assert (monitor._runtime_error) == ("ordinary bug")

    def test_environment_grouping_never_averages_native_and_wsl(self):
        import pandas as pd

        from acprof.plotting.data import aggregate_metric, build_plot_groups
        frame = pd.DataFrame([dict(cpu_cores=1, mem_cap_gb=4, gpu_mode="off", input_scale=1,
                                   environment_class=environment, latency_app_s=latency)
                              for environment, latency in (("native_linux", 1), ("wsl2", 9))])
        result = aggregate_metric(frame, "latency_app_s")
        assert (list(result["latency_app_s"])) == ([1, 9])
        assert (len(result)) == (2)
        assert (all(name.startswith(("native_linux/", "wsl2/")) for name, _ in build_plot_groups(frame)))
    def test_installed_version_is_recorded_without_git_or_hardware(self):
        from acprof.host import platform_metadata
        with patch('acprof.host.platform_metadata.__version__', '0.9.7'), \
                patch('acprof.host.platform_metadata.run_command', side_effect=FileNotFoundError('tool missing')), \
                patch.dict('sys.modules', {'pynvml': None}):
            metadata = platform_metadata.collect_platform_metadata(NATIVE, '/installed/acprof/_bundle')
        assert (metadata.get('acprof_version')) == ('0.9.7')
        assert (metadata['git_commit']) is None
        assert ('git') in (metadata['errors'])

    def test_historical_metadata_does_not_invent_package_version(self):
        from acprof.artifacts import read_static_metadata
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'static_meta.json'
            for runtime in ({}, {'acprof_version': '0.1.4', 'git_commit': None}):
                path.write_text(json.dumps({'schema_version': 7, 'platform_runtime': runtime}))
                metadata = read_static_metadata(temporary)
                assert (metadata['platform_runtime']) == (runtime)
