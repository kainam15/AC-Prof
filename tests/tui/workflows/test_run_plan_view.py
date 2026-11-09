"""Collection plans project facts without Pilot, probes, or configuration mutation."""
from dataclasses import replace
from pathlib import Path

import pytest

from acprof.experiments.configuration import ConfigIssue, RunConfig
from acprof.messages import join_messages
from acprof.tui.i18n import message_id, translate
from acprof.tui.run.planning import PlanView, invalid_plan_view, run_plan_view
from acprof.tui.ui.view_models import FactRow


def fact(view: PlanView, label_id: str, *, detail: bool = False) -> FactRow:
    rows = view.section.detail_rows if detail else view.section.rows
    return next(row for row in rows if row.label == message_id(label_id))


@pytest.mark.parametrize("language", ["zh", "en"])
@pytest.mark.parametrize("repeat", [1, 2, 3])
def test_plan_distinguishes_total_and_per_condition_windows(repeat, language):
    config = replace(RunConfig.smoke("demo/model"), cpus="1,2", task="text-generation",
                     input_scales="64,128", warmup=1, repeat=repeat)
    view = run_plan_view(config)

    assert fact(view, "run.plan.resources_label").value == "2"
    assert fact(view, "run.plan.scales_label").value == message_id("run.plan.scale_count_unit", 2, "tokens")
    assert fact(view, "run.plan.windows_label").value == message_id("run.plan.window_totals", 4, 4 * repeat)
    per_condition = fact(view, "run.plan.per_condition_label")
    assert per_condition.value == message_id("run.plan.per_condition_value", repeat)
    assert per_condition.status == ("limited" if repeat < 3 else None)
    assert view.summary.level == ("warning" if repeat < 3 else "neutral")
    assert view.callout is None

    evidence = translate(fact(view, "run.plan.evidence_label", detail=True).value, language)
    headline = translate(view.summary.headline, language)
    if repeat < 3:
        assert ("无置信区间" if language == "zh" else "no CI") in evidence
        assert ("受限" if language == "zh" else "Limited") in headline
    else:
        assert evidence == ("有效数以结果为准" if language == "zh" else "Valid counts come from results")
        assert headline == ("采集计划已计算" if language == "zh" else "Collection plan calculated")


def test_unresolved_scales_keep_unknown_totals_and_the_per_condition_plan():
    view = run_plan_view(RunConfig())
    pending = message_id("common.state.pending")

    assert fact(view, "run.plan.resources_label").value == "32"
    assert fact(view, "run.plan.scales_label").value == message_id(
        "run.plan.scale_count_unit", pending, message_id("run.inputs.unit_pending"),
    )
    assert fact(view, "run.plan.windows_label").value == message_id("run.plan.window_totals", pending, pending)
    assert fact(view, "run.plan.per_condition_label").value == message_id("run.plan.per_condition_value", 5)
    assert view.summary.level == "neutral"
    assert view.summary.headline == message_id("run.plan.pending_inputs")
    assert fact(view, "run.plan.scale_values_label", detail=True).value == pending


def test_resolved_plan_supplies_the_actual_axis_values_and_window_counts():
    config = replace(RunConfig.smoke("demo/model"), repeat=3)
    planned = {"scales": [8, 16], "scale_type": "seq_length"}
    view = run_plan_view(config, planned)

    assert fact(view, "run.plan.scales_label").value == message_id("run.plan.scale_count_unit", 2, "tokens")
    assert fact(view, "run.plan.windows_label").value == message_id("run.plan.window_totals", 0, 6)
    values = fact(view, "run.plan.scale_values_label", detail=True)
    assert values.value == values.detail == "8, 16"
    assert values.copyable


def test_manual_scales_are_deduplicated_for_counts_but_original_text_is_preserved():
    config = replace(RunConfig.smoke("demo/model"), input_scales="64,64,128", repeat=3)
    view = run_plan_view(config)

    assert fact(view, "run.plan.scales_label").value == message_id(
        "run.plan.scale_count_unit", 2, message_id("run.inputs.unit_pending"),
    )
    assert fact(view, "run.plan.windows_label").value == message_id("run.plan.window_totals", 0, 6)
    raw_scales = fact(view, "run.plan.scale_values_label", detail=True)
    assert raw_scales.value == raw_scales.detail == config.input_scales


@pytest.mark.parametrize(("runtime", "cases"), [("process", "1"), ("docker", "16")])
def test_resource_cases_follow_runtime_without_treating_inheritance_as_a_failure(runtime, cases):
    config = replace(RunConfig.smoke("demo/model"), runtime=runtime, cpus="auto", mems="auto", repeat=3)
    view = run_plan_view(config)

    assert fact(view, "run.runtime.label").value == ("Process" if runtime == "process" else "Docker")
    assert fact(view, "run.plan.resources_label").value == cases
    assert view.summary.level == "neutral"
    label = "run.resources.inherited_limits" if runtime == "process" else "run.plan.automatic_resources"
    assert fact(view, "run.resources.cpu_list", detail=True).value == message_id(label)
    assert fact(view, "run.resources.memory_gb", detail=True).value == message_id(label)
    if runtime == "process":
        assert fact(view, "run.plan.resource_rules_label", detail=True).value == message_id(
            "run.runtime.process_limits_note",
        )


@pytest.mark.parametrize(("gpus", "expected"), [
    ("off", "CPU only"), ("on", "GPU only"), ("off,on", "CPU + GPU"),
    ("on,off", "CPU + GPU"), ("cpu", "CPU only"), ("yes", "GPU only"),
])
def test_gpu_details_explain_supported_modes_and_preserve_the_original_setting(gpus, expected):
    config = replace(RunConfig.smoke("demo/model"), gpus=gpus)
    row = fact(run_plan_view(config), "run.resources.gpu_mode", detail=True)

    assert translate(row.value, "en") == expected
    assert row.detail == gpus


def test_resource_case_count_does_not_multiply_by_selected_tasks():
    config = replace(RunConfig.smoke("demo/model"), cpus="1,2", repeat=3,
                     tasks="text-generation,text-classification")
    view = run_plan_view(config)

    assert fact(view, "run.plan.resources_label").value == "2"
    assert fact(view, "run.plan.windows_label").value == message_id("run.plan.window_totals", 0, 6)


def test_detail_rows_preserve_the_removed_summaries_and_raw_paths():
    output = "/tmp/原始 results/[raw]/a-very-long-experiment-output-path"
    cache = "/tmp/原始 cache/[raw]/a-very-long-model-store-path"
    config = replace(RunConfig.smoke("demo/model"), output_dir=output, resume=True,
                     request_timeout_seconds=17.5, compute_profile_tool="ncu",
                     execution_profile_tool="massif", model_source="modelscope")
    view = run_plan_view(config, download_budget="5GB", model_store=cache)

    for label, path in (("settings.experiment.output_directory", output),
                        ("downloads.model_store.path_label", cache)):
        row = fact(view, label, detail=True)
        assert row.value == row.detail == path
        assert row.copyable
    assert fact(view, "run.windows.request_timeout_seconds", detail=True).value == "17.5"
    assert fact(view, "run.plan.profilers_label", detail=True).value == message_id(
        "profiling.options.selected_summary", "ncu", "massif",
    )
    assert fact(view, "downloads.budget.label", detail=True).value == "5GB"
    assert fact(view, "models.source.label", detail=True).value == "ModelScope"
    assert fact(view, "models.source.label", detail=True).detail == "modelscope"
    assert fact(view, "downloads.size.expected", detail=True).value == message_id("downloads.plan.pending_estimate")
    assert fact(view, "run.plan.strategy_label", detail=True).value == message_id("downloads.strategy.automatic_summary")
    assert fact(view, "run.plan.resume_label", detail=True).value == message_id("run.resume.original_parameters")
    duration = fact(view, "run.plan.duration_label", detail=True)
    assert duration.value == join_messages("\n", (
        message_id("run.plan.duration_estimate", "1 s", message_id("run.plan.assumed_request_duration")),
        message_id("run.plan.duration_exclusions"),
    ))
    assert duration.detail == ""
    english = translate(duration.value, "en")
    assert "assuming 1s/request" in english
    assert "excludes downloads, builds, service startup and additional validation" in english
    assert "时间" not in english


@pytest.mark.parametrize(("seconds", "expected"), [(5, "5 s"), (60, "1.0 min"), (3600, "1.0 h")])
def test_automatic_window_estimates_retain_the_duration_assumption(seconds, expected):
    config = replace(RunConfig.smoke("demo/model"), repeat_in_window=0, repeat_window_seconds=seconds)
    duration = fact(run_plan_view(config), "run.plan.duration_label", detail=True)

    assert duration.value == join_messages("\n", (
        message_id("run.plan.duration_estimate", expected,
                   message_id("run.plan.assumed_window_duration", seconds)),
        message_id("run.plan.duration_exclusions"),
    ))


def test_environment_defaults_are_supplied_by_the_caller(monkeypatch):
    def unexpected_home():
        raise AssertionError("The view must not resolve a home directory")

    monkeypatch.setenv("ACPROF_MAX_DOWNLOAD", "999GB")
    monkeypatch.setenv("ACPROF_MODEL_STORE", "/unrelated/environment/cache")
    monkeypatch.setattr(Path, "home", unexpected_home)
    view = run_plan_view(RunConfig.smoke("demo/model"), download_budget="2GB", model_store="/provided/cache")

    assert fact(view, "downloads.budget.label", detail=True).value == "2GB"
    assert fact(view, "downloads.model_store.path_label", detail=True).value == "/provided/cache"


def test_invalid_plan_replaces_stale_facts_and_preserves_every_raw_issue():
    reason = message_id("validation.windows.valid_request_timeout")
    raw = "External failure [raw] at /tmp/原始 input"
    issues = (ConfigIssue("request_timeout_seconds", reason), ConfigIssue(None, raw))
    view = invalid_plan_view(issues)

    assert view.summary.level == "error"
    assert view.section.rows == view.section.detail_rows == ()
    assert view.callout is not None
    assert view.callout.problem is reason
    assert view.callout.detail == f"[request_timeout_seconds] {reason}\n{raw}"
    assert view.callout.actions == ()
    assert translate(view.callout.problem, "en") == "Request timeout must be finite and greater than 0"
    assert translate(view.callout.next_step, "en") == "Review and correct the relevant fields."
