"""Cached machine evidence stays truthful without Pilot, probes, or mutable state."""
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import pytest

from acprof.tui.i18n import CatalogMessage, translate
from acprof.tui.run.diagnostics import PreflightCheck
from acprof.tui.run.environment_view import (
    check_group,
    check_label,
    check_presentation,
    check_priority,
    check_view,
    group_summary,
    item_status,
)


@pytest.mark.parametrize(("status", "group", "level", "symbol", "english"), [
    ("ok", "success", "success", "✓", "Available (environment checked)"),
    ("warn", "warning", "warning", "!", "Limited"),
    ("fail", "error", "error", "×", "Unavailable"),
    ("error", "error", "error", "×", "Check error"),
    ("pending", "pending", "neutral", "-", "Execution unverified"),
    ("declared", "pending", "neutral", "-", "Platform declaration"),
    ("not_requested", "pending", "neutral", "-", "Not enabled"),
    ("future_state", "pending", "neutral", "-", "Unknown"),
])
def test_statuses_describe_machine_evidence_without_exposing_enums(status, group, level, symbol, english):
    row = PreflightCheck("Docker", status, "", check_id="docker")
    presentation = check_presentation(row)

    assert item_status(row) == status
    assert check_group(row) == group
    assert (presentation.level, presentation.symbol) == (level, symbol)
    assert translate(presentation.label, "en") == english
    assert "Blocked" not in translate(presentation.label, "en")


@pytest.mark.parametrize(("status", "expected"), [
    ("ok", "error"), ("warn", "error"), ("future_state", "error"), ("not_requested", "not_requested"),
])
def test_capability_check_error_keeps_the_existing_not_requested_exception(status, expected):
    row = PreflightCheck("GPU", status, "", capability_status="error", check_id="gpu_device")

    assert item_status(row) == expected
    assert check_group(row) == ("pending" if status == "not_requested" else "error")


def test_unknown_status_sorting_preserves_stable_order_and_does_not_mutate_rows():
    rows = [
        PreflightCheck("future", "new_status", ""),
        PreflightCheck("unused", "not_requested", ""),
        PreflightCheck("limited", "warn", ""),
        PreflightCheck("failure", "fail", ""),
        PreflightCheck("declared", "declared", ""),
        PreflightCheck("passed", "ok", ""),
    ]
    original = tuple(rows)

    assert [row.label for row in sorted(rows, key=check_priority)] == [
        "failure", "limited", "passed", "future", "declared", "unused",
    ]
    assert tuple(rows) == original


def test_known_labels_are_typed_and_future_labels_keep_the_original_value():
    known = PreflightCheck("raw identifier", "ok", "", check_id="cpu_energy")
    label = check_label(known)

    assert isinstance(label, CatalogMessage)
    assert translate(label, "en") == "RAPL CPU energy"
    future = PreflightCheck("New device / 原始名称", "ok", "", check_id="future_device")
    assert check_label(future) is future.label


@pytest.mark.parametrize(("language", "expected"), [
    ("zh", "× RAPL CPU 能耗 · 不可用 · 权限不足"),
    ("en", "× RAPL CPU energy · Unavailable · Permission denied"),
])
def test_failure_is_actionable_and_preserves_the_caller_owned_details(language, expected):
    raw = " \nPermissionError: /sys/class/powercap/[raw]\nsource=probe\tstatus=permission_denied\n "
    row = PreflightCheck("energy", "fail", "untouched original",
                         capability_status="permission_denied", check_id="cpu_energy")
    before = asdict(row)
    view = check_view(row, technical_detail=raw, check_allowed=True)

    assert view.facts is None
    assert view.callout is not None
    assert view.callout.level == "error"
    assert translate(view.callout.problem, language) == expected
    assert view.callout.detail is raw
    assert asdict(row) == before
    action, = view.callout.actions
    assert action.id == "environment-retry"
    assert action.primary
    assert not action.disabled
    assert translate(action.label, "en") == "Check again"


def test_recheck_permission_is_supplied_by_the_caller_and_does_not_change_evidence():
    row = PreflightCheck("Docker", "fail", "original", check_id="docker")
    allowed = check_view(row, technical_detail="raw", check_allowed=True).callout
    disabled = check_view(row, technical_detail="raw", check_allowed=False).callout

    assert allowed is not None
    assert disabled is not None
    assert disabled.actions[0].disabled
    assert replace(disabled, actions=allowed.actions) == allowed


def test_exception_words_do_not_classify_capability_reasons():
    row = PreflightCheck("Docker", "fail", "PermissionError: unsupported token", check_id="docker")
    first = check_view(row, technical_detail=row.detail, check_allowed=True).callout
    second = check_view(replace(row, detail="unrelated failure"),
                        technical_detail="unrelated failure", check_allowed=True).callout

    assert first is not None
    assert second is not None
    assert first.problem == second.problem
    assert "Permission denied" not in translate(first.problem, "en")
    assert "Unsupported" not in translate(first.problem, "en")
    typed = check_view(replace(row, capability_status="unsupported"),
                       technical_detail=row.detail, check_allowed=True).callout
    assert typed is not None
    assert "Unsupported in this environment" in translate(typed.problem, "en")


@pytest.mark.parametrize(("identifier", "fragment"), [
    ("docker", "runtime capability"),
    ("cpu_energy", "CPU or memory metrics"),
    ("gpu_power", "GPU capabilities or metrics"),
    ("capture_interface", "Packet capture"),
    ("torch", "profiler analysis"),
    ("future_device", "launch checks"),
])
def test_capability_impacts_describe_scope_without_claiming_a_launch_block(identifier, fragment):
    view = check_view(PreflightCheck("device", "fail", "", check_id=identifier),
                      technical_detail="", check_allowed=True)
    assert view.callout is not None

    impact = translate(view.callout.impact, "en")
    assert fragment in impact
    assert "cannot start" not in impact.lower()
    assert "blocked" not in impact.lower()


def test_check_execution_error_does_not_claim_unavailability():
    row = PreflightCheck("Docker", "fail", "", capability_status="error", check_id="docker")
    view = check_view(row, technical_detail="RuntimeError: probe failed", check_allowed=True)

    assert view.callout is not None
    assert translate(view.callout.problem, "en") == "× Docker · Check error"
    assert translate(view.callout.impact, "en") == (
        "This check did not complete; capability availability is unconfirmed."
    )


@pytest.mark.parametrize("identifier", ["ncu", "nsys", "intel_advisor", "torch", "massif", "profiler"])
def test_profiler_recovery_points_to_installation_and_runtime_evidence(identifier):
    view = check_view(PreflightCheck(identifier, "fail", "", check_id=identifier),
                      technical_detail="", check_allowed=True)
    assert view.callout is not None

    step = translate(view.callout.next_step, "en")
    assert "tool installation" in step
    assert "runtime self-check" in step


@pytest.mark.parametrize(("status", "note"), [
    ("ok", "Actual collection was not verified"),
    ("declared", "Only platform support is declared"),
    ("pending", "This capability remains unverified"),
    ("not_requested", "availability has not been established"),
    ("future_state", "This check state is not recognized"),
])
def test_non_errors_use_facts_with_copyable_unmodified_technical_details(status, note):
    raw = "\n  /long/raw/path [literal]\tvalue=0\n"
    row = PreflightCheck("Docker", status, "original", check_id="docker",
                         capability_status="error" if status == "not_requested" else "")
    view = check_view(row, technical_detail=raw, check_allowed=False)

    assert view.callout is None
    assert view.facts is not None
    assert note in translate(view.facts.rows[1].value, "en")
    detail, = view.facts.detail_rows
    assert detail.value is raw
    assert detail.detail is raw
    assert detail.copyable
    assert not view.facts.collapsed
    assert "future_state" not in translate(view.facts.rows[0].value, "en")
    with pytest.raises(FrozenInstanceError):
        view.facts = None


def test_group_summary_counts_unavailable_and_check_errors_separately():
    rows = [
        PreflightCheck("unavailable", "fail", ""),
        PreflightCheck("probe error", "fail", "", capability_status="error"),
        PreflightCheck("passed", "ok", ""),
    ]
    summary = group_summary(rows, group="error", state="limited")

    assert summary.level == "error"
    assert translate(summary.headline, "en") == "Unavailable (1) · Check error (1)"
    assert "Blocked" not in translate(summary.headline, "en")


@pytest.mark.parametrize(("state", "headline"), [
    ("ready", "Error (0)"),
    ("limited", "Error (0)"),
    ("error", "Check failed · Error (0)"),
    ("checking", "Checking · Error (0)"),
    ("stale", "Stale · Error (0)"),
    ("future_state", "Unknown · Error (0)"),
])
def test_empty_group_is_neutral_and_keeps_explicit_check_state(state, headline):
    summary = group_summary([PreflightCheck("passed", "ok", "")], group="error", state=state)

    assert summary.level == "neutral"
    assert translate(summary.headline, "en") == headline


@pytest.mark.parametrize(("state", "level", "headline"), [
    ("ready", "success", "Passed (1)"),
    ("limited", "success", "Passed (1)"),
    ("checking", "info", "Checking · Passed (1)"),
    ("error", "error", "Check failed · Passed (1)"),
    ("stale", "neutral", "Stale · Passed (1)"),
    ("future_state", "neutral", "Unknown · Passed (1)"),
])
def test_group_state_is_explicit_without_deriving_freshness_from_details(state, level, headline):
    row = PreflightCheck("cached", "ok", "2020-01-01 stale error unknown")
    summary = group_summary([row], group="success", state=state)

    assert summary.level == level
    assert translate(summary.headline, "en") == headline
    assert summary.actions == ()


@pytest.mark.parametrize(("status", "headline"), [
    ("not_requested", "Not enabled (1)"),
    ("declared", "Platform declaration (1)"),
    ("future_state", "Unverified (1)"),
])
def test_unverified_groups_remain_neutral(status, headline):
    summary = group_summary([PreflightCheck("capability", status, "")], group="pending", state="limited")

    assert summary.level == "neutral"
    assert translate(summary.headline, "en") == headline


def test_projection_does_not_read_files_or_change_the_cache(monkeypatch):
    def unexpected_io(*args, **kwargs):
        raise AssertionError("Cached environment projections must not read host files")

    monkeypatch.setattr(Path, "open", unexpected_io)
    monkeypatch.setattr(Path, "home", unexpected_io)
    rows = (PreflightCheck("GPU", "warn", "cached only", check_id="gpu_power"),)
    before = tuple(asdict(row) for row in rows)

    assert check_view(rows[0], technical_detail="raw", check_allowed=False).callout is not None
    assert group_summary(rows, group="warning", state="limited").level == "warning"
    assert tuple(asdict(row) for row in rows) == before
