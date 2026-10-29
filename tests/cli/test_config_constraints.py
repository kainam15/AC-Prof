"""Pure constraints, field errors and environment checks have distinct boundaries."""

from unittest.mock import Mock

import pytest

from acprof.config_constraints import ConstraintViolation, NumericConstraint, integer
from acprof.experiment import RunConfig, RunConfigError
from acprof.host.measurement.config import ClientConfig
from acprof.messages import Message
from acprof.run_args import arguments_from_options


@pytest.mark.parametrize("value", [2, 2.0, "2", " 2 ", "+2"])
def test_form_integer_normalizes_without_loss(value):
    result = RunConfig(model="fixture/model", repeat=value).validate()
    assert result.repeat == 2
    assert type(result.repeat) is int


@pytest.mark.parametrize("value", [True, False, 2.5, "2.0", "2e0", "", None,
                                  float("nan"), float("inf"), float("-inf")])
def test_form_integer_rejects_truncation_and_non_numeric_values(value):
    with pytest.raises(RunConfigError) as failure:
        RunConfig(model="fixture/model", repeat=value).validate()
    assert [issue.field for issue in failure.value.issues] == ["repeat"]


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), float("-inf")])
def test_finite_numeric_rule_is_shared_by_run_and_measurement(value):
    with pytest.raises(RunConfigError) as failure:
        RunConfig(model="fixture/model", sample_hz=value).validate()
    assert [issue.field for issue in failure.value.issues] == ["sample_hz"]
    with pytest.raises(ValueError, match="sample_hz"):
        ClientConfig(sample_hz=value).validate()


def test_large_integer_does_not_round_trip_through_float():
    large = 10 ** 1000
    assert integer(large) == large
    assert NumericConstraint(whole=True, minimum=1).parse(large) == large
    with pytest.raises(ConstraintViolation):
        NumericConstraint(exclusive=True).parse(large)


def test_internal_zero_windows_remain_valid():
    ClientConfig(repeat=0, warmup=0, auto_warmup_requests=0).validate()
    with pytest.raises(RunConfigError) as failure:
        RunConfig(model="fixture/model", repeat=0).validate()
    assert [issue.field for issue in failure.value.issues] == ["repeat"]


@pytest.mark.parametrize("field", ["batch_size", "warmup", "repeat_in_window", "auto_warmup_requests"])
def test_direct_measurement_rejects_fractional_counts(field):
    with pytest.raises(ValueError, match=field):
        ClientConfig(**{field: 2.5}).validate()


def test_raw_sample_limit_retains_measurement_integer_invariant():
    with pytest.raises(ValueError, match="raw_sample_limit"):
        ClientConfig(raw_sample_limit=2.0).validate()
    ClientConfig(raw_sample_limit=2).validate()


def test_all_profiler_and_form_errors_retain_their_field():
    with pytest.raises(RunConfigError) as failure:
        RunConfig(model="fixture/model", repeat=2.5, extra_options={
            "ncu_repeat": 0, "nsys_repeat": True, "massif_reference_mem": -1,
            "compute_profile_cpus": 2.5, "matrix_seed": 3.5,
        }).validate()
    assert {issue.field for issue in failure.value.issues} == {
        "repeat", "ncu_repeat", "nsys_repeat", "massif_reference_mem",
        "compute_profile_cpus", "matrix_seed",
    }
    assert all(isinstance(issue.reason, Message) for issue in failure.value.issues)


def test_extra_options_normalize_once_and_survive_cli_transport():
    original = {"ncu_repeat": 2.0, "massif_reference_cpu": "3", "nsys_reference_mem": None}
    config = RunConfig(model="fixture/model", extra_options=original).validate()
    assert config.extra_options == {"ncu_repeat": 2, "massif_reference_cpu": 3, "nsys_reference_mem": None}
    assert original["massif_reference_cpu"] == "3"
    args = config.to_namespace()
    assert (args.ncu_repeat, args.massif_reference_cpu, args.nsys_reference_mem) == (2, 3, None)


@pytest.mark.parametrize("value", [None, [], "invalid"])
def test_invalid_extra_options_are_reported_without_attribute_errors(value):
    with pytest.raises(RunConfigError) as failure:
        RunConfig(model="fixture/model", runtime="process", profiling_mode="basic",
                  compute_profile_tool="none", execution_profile_tool="none", extra_options=value).validate()
    assert [issue.field for issue in failure.value.issues] == ["extra_options"]


@pytest.mark.parametrize("value", [True, 2.5, float("nan"), float("inf")])
def test_frozen_integer_transport_does_not_truncate(value):
    with pytest.raises(ValueError, match="ncu_repeat"):
        arguments_from_options({"ncu_repeat": value})


def test_static_latency_and_dram_constraints_are_field_errors():
    with pytest.raises(RunConfigError) as failure:
        RunConfig(model="fixture/model", profiling_mode="basic",
                  extra_options={"latency_slo": ["1.5"], "dram_energy": "required"}).validate()
    assert {issue.field for issue in failure.value.issues} == {"latency_slo", "dram_energy"}


def test_validation_does_not_read_dynamic_affinity_or_slo_environment(monkeypatch):
    import os

    affinity = Mock(side_effect=AssertionError("queried live affinity during validation"))
    monkeypatch.setattr(os, "sched_getaffinity", affinity)
    monkeypatch.setenv("SLOW_LATENCY_THRESHOLD_S", "invalid")
    config = RunConfig(model="fixture/model", cpuset_cpus="3,1-2").validate()
    assert config.cpuset_cpus == "1-3"
    affinity.assert_not_called()


def test_preflight_uses_normalized_cpu_set_and_checks_live_affinity(monkeypatch):
    import acprof.cpu_affinity as affinity
    from acprof.application.run import RunApplication

    normalize = Mock(wraps=affinity.normalize_cpu_set)
    monkeypatch.setattr(affinity, "normalize_cpu_set", normalize)
    monkeypatch.setattr("acprof.application.run.os.sched_getaffinity", lambda pid: {0})
    monkeypatch.delenv("SLOW_LATENCY_THRESHOLD_S", raising=False)
    application = RunApplication()
    application.environment_setup = Mock()
    with pytest.raises(ValueError, match="CPUs unavailable"):
        application._validate_execution_options(RunConfig(model="fixture/model", cpuset_cpus="1,0"))
    normalize.assert_called_once_with("1,0")


def test_new_field_messages_translate_at_presentation_boundary():
    from acprof.tui.i18n import translate

    with pytest.raises(RunConfigError) as failure:
        RunConfig(model="fixture/model", runtime="invalid", sampling_policy="invalid").validate()
    translated = {issue.field: translate(issue.reason, "en") for issue in failure.value.issues}
    assert translated == {"runtime": "Execution runtime must be auto, docker or process",
                          "sampling_policy": "Sampling policy must be burst or skip"}
