"""Pre-run scale, window and cost estimates have explicit assumptions."""
from dataclasses import replace

import pytest

from acprof.experiment import RunConfig


@pytest.mark.parametrize('scales_case', range(5), ids=['[]', '[0]', '[True]', "[float('inf')]", '[1] * 10001'])
def test_preparation_plan_event_rejects_invalid_or_unbounded_scales(scales_case):
    from acprof.preparation_events import encode_event, parse_event
    plan = {"scales": [1.0], "scale_type": "duration_s"}
    assert (parse_event(encode_event("input", "passed", input_plan=plan))["input_plan"]) == (plan)
    scales = tuple(([], [0], [True], [float('inf')], [1] * 10001))[scales_case]
    with pytest.raises(ValueError):
        encode_event("input", "passed", input_plan={"scales": scales, "scale_type": "duration_s"})

@pytest.mark.parametrize('number', ['NaN', '1e999'])
def test_preparation_event_rejects_nonfinite_json(number):
    from acprof.preparation_events import PREFIX, parse_event
    line = PREFIX + '{"version":1,"stage":"input","status":"passed","corrupt_metric":' + number + '}'

    with pytest.raises(ValueError, match='non-finite'):
        parse_event(line)

@pytest.mark.parametrize('number', ['NaN', '1e999'])
def test_progress_event_rejects_nonfinite_json(number):
    from acprof.progress_events import PREFIX, parse_event
    line = PREFIX + '{"version":1,"event":"case_started","case_id":"case-1","corrupt_metric":' + number + '}'

    with pytest.raises(ValueError, match='non-finite'):
        parse_event(line)

def test_smoke_estimate_is_one_configuration_and_one_formal_window():
    from acprof.tui.run_planning import estimate_run
    estimate = estimate_run(RunConfig.smoke())
    assert ((estimate.cases, estimate.scales, estimate.warmup_windows, estimate.formal_windows)) == ((1, 1, 0, 1))
    assert (estimate.estimated_seconds) == (1)

def test_auto_matrix_count_stays_unknown_until_input_plan_exists():
    from acprof.tui.run_planning import estimate_run
    config = RunConfig()
    estimate = estimate_run(config)
    assert (estimate.cases) == (32)
    assert (estimate.scales) is None
    assert (estimate.formal_windows) is None
    assert (estimate.estimated_seconds) is None
    estimate = estimate_run(config, {"scales": [1, 2, 3], "scale_type": "duration_s"})
    assert ((estimate.warmup_windows, estimate.formal_windows)) == ((192, 480))
    assert (estimate.estimated_seconds) == (672 * 35)

def test_manual_duplicate_scales_follow_planner_deduplication():
    from acprof.tui.run_planning import estimate_run
    estimate = estimate_run(replace(RunConfig.smoke(), input_scales="1,1,2", repeat=3, warmup=1))
    assert ((estimate.scales, estimate.warmup_windows, estimate.formal_windows)) == ((2, 2, 6))

def test_units_follow_resolved_input_axis_without_guessing_model_names():
    from acprof.tui.run_planning import input_unit
    config = RunConfig.smoke("openai/whisper-tiny")
    assert ("待解析") in (str(input_unit(config)))
    for axis, expected in (("duration_s", "s"), ("seq_length", "tokens"), ("resolution_scale", "×224px")):
        assert (input_unit(config, {"scale_type": axis})) == (expected)


def test_input_plan_identity_changes_with_model_revision_but_not_repeat_count():
    from acprof.tui.run_planning import input_identity

    base = RunConfig.smoke("demo/model")
    revision = replace(base, revision="0123456789abcdef0123456789abcdef01234567")
    repeated = replace(base, repeat=7)

    assert input_identity(revision) != input_identity(base)
    assert input_identity(repeated) == input_identity(base)


@pytest.mark.parametrize("repeat,formal_windows,flow_check", ((1, 4, True), (2, 8, True), (3, 12, False)))
@pytest.mark.parametrize("language", ("zh", "en"))
def test_summary_distinguishes_planned_windows_per_condition_from_total(repeat, formal_windows, flow_check, language):
    from acprof.tui.i18n import translate
    from acprof.tui.run_planning import plan_summary

    config = replace(RunConfig.smoke("demo/model"), cpus="1,2",
                     input_scales="64,128", repeat=repeat)
    summary = translate(plan_summary(config), language)
    if language == "zh":
        assert f"正式 {formal_windows} 窗口" in summary
        assert f"每条件计划 {repeat} 个正式窗口" in summary
        assert ("不足 3 个：仅验证流程，无置信区间" in summary) is flow_check
        assert ("有效数以结果为准" in summary) is not flow_check
    else:
        assert f"{formal_windows} measured windows" in summary
        assert f"Plan: {repeat} windows/condition" in summary
        assert ("Under 3: flow check only, no CI" in summary) is flow_check
        assert ("Valid counts come from results" in summary) is not flow_check


def test_summary_shows_per_condition_plan_even_before_scales_are_resolved():
    from acprof.tui.run_planning import plan_summary

    summary = str(plan_summary(RunConfig()))
    assert "输入 待解析 档" in summary
    assert "每条件计划 5 个正式窗口" in summary
    assert "有效数以结果为准" in summary
