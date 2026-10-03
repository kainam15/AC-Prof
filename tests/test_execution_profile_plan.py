import json
import math
import os
import tempfile

import pytest

from acprof.host.execution_profile_plan import (
    EXECUTION_PROFILE_FIELDS,
    MASSIF_ERROR_FIELD,
    MASSIF_HEAP_EXTRA_PEAK_FIELD,
    MASSIF_HEAP_PEAK_FIELD,
    MASSIF_HEAP_PEAK_TOTAL_FIELD,
    MASSIF_METRIC_FIELDS,
    MASSIF_PEAK_AT_MS_FIELD,
    MASSIF_STACK_PEAK_FIELD,
    NSYS_CUDA_API_CALL_COUNT_FIELD,
    NSYS_CUDA_API_TIME_FIELD,
    NSYS_ERROR_FIELD,
    NSYS_GPU_KERNEL_LAUNCH_COUNT_FIELD,
    NSYS_GPU_KERNEL_TIME_FIELD,
    NSYS_GPU_MEMCPY_BYTES_FIELD,
    NSYS_GPU_MEMCPY_COUNT_FIELD,
    NSYS_GPU_MEMCPY_TIME_FIELD,
    NSYS_HOST_WALL_TIME_FIELD,
    NSYS_METRIC_FIELDS,
    find_execution_profile_entry,
    load_execution_profile_plan,
)


def _massif_entry(input_scale=8.0, **overrides):
    entry = {
        "input_scale": input_scale,
        MASSIF_HEAP_PEAK_FIELD: 1000,
        MASSIF_HEAP_EXTRA_PEAK_FIELD: 200,
        MASSIF_STACK_PEAK_FIELD: 50,
        MASSIF_HEAP_PEAK_TOTAL_FIELD: 1250,
        MASSIF_PEAK_AT_MS_FIELD: 12.5,
    }
    entry.update(overrides)
    return entry


def _nsys_entry(input_scale=8.0, **overrides):
    entry = {
        "input_scale": input_scale,
        NSYS_HOST_WALL_TIME_FIELD: 20.5,
        NSYS_CUDA_API_TIME_FIELD: 3.5,
        NSYS_CUDA_API_CALL_COUNT_FIELD: 11,
        NSYS_GPU_KERNEL_TIME_FIELD: 14.25,
        NSYS_GPU_KERNEL_LAUNCH_COUNT_FIELD: 7,
        NSYS_GPU_MEMCPY_TIME_FIELD: 1.25,
        NSYS_GPU_MEMCPY_COUNT_FIELD: 2,
        NSYS_GPU_MEMCPY_BYTES_FIELD: 4096,
    }
    entry.update(overrides)
    return entry


def _profile(
    *,
    cpu_cores=2,
    mem_cap_gb=4,
    gpu_mode="off",
    tools=None,
):
    return {
        "cpu_cores": cpu_cores,
        "mem_cap_gb": mem_cap_gb,
        "gpu_mode": gpu_mode,
        "tools": tools or {},
    }


class TestExecutionProfilePlan:
    def assert_empty_metrics(self, result, fields):
        for field in fields:
            assert (math.isnan(result[field])), field

    def test_load_valid_plan_preserves_metadata_and_profiles(self):
        expected = {
            "execution_profile_schema_version": 1,
            "model_id": "example/model",
            "profiles": [_profile()],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "execution_profile_plan.json")
            with open(path, "w", encoding="utf-8") as plan_file:
                json.dump(expected, plan_file)

            actual = load_execution_profile_plan(path)

        assert (actual) == (expected)

    @pytest.mark.parametrize('name_case', range(3))
    def test_load_invalid_shapes_return_soft_diagnostics(self, name_case):
        invalid_documents = (
            ("not_dict", []),
            ("missing_profiles", {}),
            ("profiles_not_list", {"profiles": {}}),
        )
        (name, document) = tuple(invalid_documents)[name_case]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "plan.json")
            with open(path, "w", encoding="utf-8") as plan_file:
                json.dump(document, plan_file)
            plan = load_execution_profile_plan(path)

            assert (plan["profiles"]) == ([])
            assert ("execution_profile_plan_invalid") in (plan["_load_error"])

    def test_empty_path_disables_profiles_without_error_pollution(self):
        plan = load_execution_profile_plan("")

        cpu_result = find_execution_profile_entry(plan, 2, 4, "off", 8)
        gpu_result = find_execution_profile_entry(plan, 2, 4, "on", 8)

        assert (set(cpu_result)) == (set(EXECUTION_PROFILE_FIELDS))
        assert (cpu_result[MASSIF_ERROR_FIELD]) == ("")
        assert (cpu_result[NSYS_ERROR_FIELD]) == ("")
        assert (gpu_result[MASSIF_ERROR_FIELD]) == ("")
        assert (gpu_result[NSYS_ERROR_FIELD]) == ("")
        self.assert_empty_metrics(cpu_result, MASSIF_METRIC_FIELDS + NSYS_METRIC_FIELDS)
        self.assert_empty_metrics(gpu_result, MASSIF_METRIC_FIELDS + NSYS_METRIC_FIELDS)

    def test_missing_file_reports_only_the_applicable_tool(self):
        plan = load_execution_profile_plan("/definitely/missing/execution-plan.json")

        cpu_result = find_execution_profile_entry(plan, 2, 4, "off", 8)
        gpu_result = find_execution_profile_entry(plan, 2, 4, "on", 8)

        assert ("execution_profile_plan_not_found") in (cpu_result[MASSIF_ERROR_FIELD])
        assert (cpu_result[NSYS_ERROR_FIELD]) == ("")
        assert (gpu_result[MASSIF_ERROR_FIELD]) == ("")
        assert ("execution_profile_plan_not_found") in (gpu_result[NSYS_ERROR_FIELD])

    def test_cpu_row_applies_only_matching_massif_entry(self):
        plan = {
            "profiles": [
                _profile(
                    tools={
                        "massif": {"entries": [_massif_entry()]},
                        "nsys": {"entries": [_nsys_entry()]},
                    },
                )
            ]
        }

        result = find_execution_profile_entry(plan, 2, 4, "off", 8)

        assert (result[MASSIF_HEAP_PEAK_FIELD]) == (1000.0)
        assert (result[MASSIF_HEAP_EXTRA_PEAK_FIELD]) == (200.0)
        assert (result[MASSIF_STACK_PEAK_FIELD]) == (50.0)
        assert (result[MASSIF_HEAP_PEAK_TOTAL_FIELD]) == (1250.0)
        assert (result[MASSIF_PEAK_AT_MS_FIELD]) == (12.5)
        assert (result[MASSIF_ERROR_FIELD]) == ("")
        self.assert_empty_metrics(result, NSYS_METRIC_FIELDS)
        assert (result[NSYS_ERROR_FIELD]) == ("")

    def test_gpu_row_applies_only_matching_nsys_entry(self):
        plan = {
            "profiles": [
                _profile(
                    gpu_mode="on",
                    tools={
                        "massif": {"entries": [_massif_entry()]},
                        "nsys": {"entries": [_nsys_entry()]},
                    },
                )
            ]
        }

        result = find_execution_profile_entry(plan, 2, 4, "on", 8)

        assert (result[NSYS_HOST_WALL_TIME_FIELD]) == (20.5)
        assert (result[NSYS_CUDA_API_TIME_FIELD]) == (3.5)
        assert (result[NSYS_CUDA_API_CALL_COUNT_FIELD]) == (11.0)
        assert (result[NSYS_GPU_KERNEL_TIME_FIELD]) == (14.25)
        assert (result[NSYS_GPU_KERNEL_LAUNCH_COUNT_FIELD]) == (7.0)
        assert (result[NSYS_GPU_MEMCPY_TIME_FIELD]) == (1.25)
        assert (result[NSYS_GPU_MEMCPY_COUNT_FIELD]) == (2.0)
        assert (result[NSYS_GPU_MEMCPY_BYTES_FIELD]) == (4096.0)
        assert (result[NSYS_ERROR_FIELD]) == ("")
        self.assert_empty_metrics(result, MASSIF_METRIC_FIELDS)
        assert (result[MASSIF_ERROR_FIELD]) == ("")

    def test_resource_case_uses_cpu_memory_and_gpu_mode(self):
        plan = {
            "profiles": [
                _profile(
                    cpu_cores=1,
                    mem_cap_gb=4,
                    tools={"massif": {"entries": [_massif_entry(
                        **{MASSIF_HEAP_PEAK_FIELD: 100}
                    )]}},
                ),
                _profile(
                    cpu_cores=2,
                    mem_cap_gb=8,
                    tools={"massif": {"entries": [_massif_entry(
                        **{MASSIF_HEAP_PEAK_FIELD: 200}
                    )]}},
                ),
            ]
        }

        result = find_execution_profile_entry(plan, 2, 8, "off", 8)

        assert (result[MASSIF_HEAP_PEAK_FIELD]) == (200.0)

    def test_scale_matching_uses_one_micro_abs_tolerance(self):
        plan = {
            "profiles": [
                _profile(
                    tools={"massif": {"entries": [_massif_entry(8.0)]}},
                )
            ]
        }

        matched = find_execution_profile_entry(
            plan,
            2,
            4,
            "off",
            8.0000005,
        )
        missing = find_execution_profile_entry(
            plan,
            2,
            4,
            "off",
            8.0000011,
        )

        assert (matched[MASSIF_HEAP_PEAK_FIELD]) == (1000.0)
        assert (matched[MASSIF_ERROR_FIELD]) == ("")
        self.assert_empty_metrics(missing, MASSIF_METRIC_FIELDS)
        assert ("execution_profile_missing_scale:massif") in (missing[MASSIF_ERROR_FIELD])

    def test_unenabled_tool_stays_nan_with_an_empty_error(self):
        cpu_plan = {
            "profiles": [
                _profile(gpu_mode="on", tools={"nsys": {"entries": [_nsys_entry()]}})
            ]
        }
        gpu_plan = {
            "profiles": [
                _profile(tools={"massif": {"entries": [_massif_entry()]}})
            ]
        }

        cpu_result = find_execution_profile_entry(cpu_plan, 2, 4, "off", 8)
        gpu_result = find_execution_profile_entry(gpu_plan, 2, 4, "on", 8)

        self.assert_empty_metrics(cpu_result, MASSIF_METRIC_FIELDS)
        assert (cpu_result[MASSIF_ERROR_FIELD]) == ("")
        self.assert_empty_metrics(gpu_result, NSYS_METRIC_FIELDS)
        assert (gpu_result[NSYS_ERROR_FIELD]) == ("")

    def test_enabled_tool_with_missing_resource_case_reports_error(self):
        plan = {
            "profiles": [
                _profile(
                    cpu_cores=1,
                    mem_cap_gb=4,
                    tools={"massif": {"entries": [_massif_entry()]}},
                )
            ]
        }

        result = find_execution_profile_entry(plan, 2, 8, "off", 8)

        self.assert_empty_metrics(result, MASSIF_METRIC_FIELDS)
        assert ("execution_profile_missing_case:massif") in (result[MASSIF_ERROR_FIELD])
        assert ("cpu_cores=2") in (result[MASSIF_ERROR_FIELD])
        assert ("mem_cap_gb=8") in (result[MASSIF_ERROR_FIELD])

    def test_tool_and_entry_errors_are_isolated_and_entry_can_clear_error(self):
        plan = {
            "profiles": [
                _profile(
                    tools={
                        "massif": {
                            "error": "tool_failed",
                            "entries": [
                                _massif_entry(8, error=""),
                                _massif_entry(16),
                            ],
                        }
                    },
                )
            ]
        }

        cleared = find_execution_profile_entry(plan, 2, 4, "off", 8)
        inherited = find_execution_profile_entry(plan, 2, 4, "off", 16)

        assert (cleared[MASSIF_ERROR_FIELD]) == ("")
        assert (inherited[MASSIF_ERROR_FIELD]) == ("tool_failed")
        assert (cleared[NSYS_ERROR_FIELD]) == ("")
        assert (inherited[NSYS_ERROR_FIELD]) == ("")

    def test_non_finite_or_unparseable_metrics_become_nan(self):
        plan = {
            "profiles": [
                _profile(
                    tools={
                        "massif": {
                            "entries": [
                                _massif_entry(
                                    **{
                                        MASSIF_HEAP_PEAK_FIELD: None,
                                        MASSIF_STACK_PEAK_FIELD: "bad",
                                        MASSIF_PEAK_AT_MS_FIELD: "inf",
                                    }
                                )
                            ]
                        }
                    },
                )
            ]
        }

        result = find_execution_profile_entry(plan, 2, 4, "off", 8)

        assert (math.isnan(result[MASSIF_HEAP_PEAK_FIELD]))
        assert (math.isnan(result[MASSIF_STACK_PEAK_FIELD]))
        assert (math.isnan(result[MASSIF_PEAK_AT_MS_FIELD]))
