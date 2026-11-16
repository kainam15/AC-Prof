"""Global counts and admission must describe the same current configuration."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from acprof.experiments.configuration import ConfigIssue, RunConfig
from acprof.tui.run.diagnostics import PreflightCheck
from acprof.tui.run.environment_view import check_group
from acprof.tui.run.preflight_controller import PreflightController
from acprof.tui.ui.environment import display_checks


def controller(*, checks=(), inventory=(), issues=(), current=None, checked=None):
    config = current or replace(RunConfig.smoke('fixture/model'), runtime='process')
    app = SimpleNamespace(
        is_running=True, form_ready=True, ui_closing=False, tr=str,
        configuration=SimpleNamespace(config_issues=issues, config_warnings=(), validation_pending=False),
    )
    result = PreflightController(app)
    result.preflight_config = lambda: config
    result.check_completed = True
    result.checked_config = checked or config
    result.checks = tuple(checks)
    result.environment_checks = tuple(inventory)
    return result


def check(identifier, status, detail='fixture'):
    return PreflightCheck(identifier, status, detail, check_id=identifier)


def errors(value):
    return {row.check_id for row in display_checks(value) if check_group(row) == 'error'}


def test_two_independent_blockers_are_visible_globally():
    value = controller(checks=[check('process_resources', 'fail')],
                       issues=(ConfigIssue('runtime', 'Process requires basic'),))
    assert errors(value) == {'config_runtime', 'process_resources'}
    assert value.preflight_run_reason()


def test_configuration_incomplete_is_a_global_error():
    value = controller(issues=(ConfigIssue('model', 'model is required'),))
    assert errors(value) == {'config_model'}


def test_same_root_in_inventory_and_preflight_counts_once():
    failure = check('process_resources', 'fail')
    value = controller(checks=[failure, failure], inventory=[failure])
    assert sum(row.check_id == 'process_resources' for row in display_checks(value)) == 1
    assert errors(value) == {'process_resources'}


def test_successful_recheck_replaces_old_inventory_failure():
    value = controller(checks=[check('process_resources', 'ok')],
                       inventory=[check('process_resources', 'fail')])
    assert not errors(value)
    assert not value.preflight_run_reason()


def test_quota_edit_discards_old_denial_until_rechecked():
    previous = replace(RunConfig.smoke('fixture/model'), runtime='process', cpus='1')
    value = controller(checks=[check('process_resources', 'fail')],
                       inventory=[check('process_resources', 'fail')],
                       checked=previous, current=replace(previous, cpus='2'))
    rows = {row.check_id: row for row in display_checks(value)}
    assert not errors(value)
    assert rows['process_resources'].status in {'pending', 'stale'}
    assert value.preflight_run_reason()


@pytest.mark.parametrize('identifier', ['ncu', 'nsys', 'torch', 'massif', 'cpu_energy', 'docker'])
def test_unrequested_capability_cannot_block_process_basic(identifier):
    value = controller(inventory=[check(identifier, 'fail')])
    assert not errors(value)
    row = next(row for row in display_checks(value) if row.check_id == identifier)
    assert row.status == 'not_requested'
    assert not value.preflight_run_reason()


def test_combined_profilers_do_not_request_posthoc_advisor():
    config = replace(RunConfig.smoke('fixture/model'), runtime='docker', execution_profile_tool='both')
    value = controller(current=config, inventory=[check('intel_advisor', 'fail')])
    assert next(row for row in display_checks(value) if row.check_id == 'intel_advisor').status == 'not_requested'
    assert value.status.counts['warning'] == 0


def test_running_recheck_does_not_count_old_success_or_failure():
    value = controller(checks=[check('process_resources', 'fail')],
                       inventory=[check('uv', 'ok')])
    value.check_running = True
    rows = display_checks(value)
    assert not errors(value)
    assert not any(row.status == 'ok' for row in rows if row.check_id != 'configuration')
    assert any(row.status == 'checking' for row in rows)
    assert 'fixture' not in value.preflight_run_reason()


def test_pending_form_validation_hides_old_configuration_error():
    value = controller(issues=(ConfigIssue('model', 'model is required'),))
    value.app.configuration.validation_pending = True
    assert not errors(value)
    assert any(row.status == 'pending' for row in display_checks(value))


@pytest.mark.parametrize('status', ['pending', 'declared', 'unknown', 'future_status'])
def test_unverified_required_check_never_allows_collection(status):
    value = controller(checks=[check('process_resources', status)])
    assert not value.status.can_start
    assert value.status.counts['success'] == 1  # Only the valid form.
    assert value.preflight_run_reason()


def test_unrelated_config_error_does_not_hide_resource_blocker():
    value = controller(checks=[check('process_resources', 'fail')],
                       issues=(ConfigIssue('model', 'missing'), ConfigIssue('model', 'missing')))
    assert value.status.counts['error'] == 2
    assert not value.status.can_start


def test_cleanup_failure_survives_config_edit_and_deduplicates_shared_exception():
    previous = replace(RunConfig.smoke('fixture/model'), runtime='process')
    value = controller(current=replace(previous, cpus='2'), checked=previous)
    value.error = value.environment_error = value.cleanup_error = 'resource still active'
    assert errors(value) == {'cleanup_error'}
    assert not value.status.can_start


def test_inventory_exception_is_visible_advisory_when_required_checks_pass():
    value = controller(checks=[check('process_resources', 'ok')])
    value.environment_error = 'self-check failed'
    assert value.status.counts['warning'] == 1
    assert value.status.can_start
    assert value.collection_state == 'limited'


def test_auto_runtime_change_discards_old_process_denial():
    previous = replace(RunConfig.smoke('fixture/model'), runtime='process')
    value = controller(current=replace(previous, runtime='auto'), checked=previous,
                       checks=[replace(check('runtime', 'ok'), runtime_kind='process'),
                               check('process_resources', 'fail')])
    assert not errors(value)
    assert value.status.counts['pending'] > 0
    assert not value.status.can_start
    value.checked_config = value.preflight_config()
    value.checks = (replace(check('runtime', 'ok'), runtime_kind='docker'), check('docker', 'ok'))
    assert value.status.can_start
    assert 'process_resources' not in {row.check_id for row in display_checks(value)}


def test_process_validation_failure_has_configuration_identity(monkeypatch):
    from acprof.tui.run.diagnostics import _process_checks
    monkeypatch.setattr('acprof.host.runtimes.capabilities.probe_collection_plan', lambda *a, **kw: {'probes': {}})
    config = replace(RunConfig.smoke('fixture/model'), runtime='process', profiling_mode='full')
    rows = _process_checks(config)
    assert any(row.check_id == 'config_runtime' and row.status == 'fail' for row in rows)
    assert not any(row.check_id == 'process_resources' for row in rows)


def test_current_required_failure_is_not_hidden_by_unresolved_auto_runtime():
    value = controller(current=RunConfig.smoke('fixture/model'), checks=[check('docker', 'fail')])
    assert errors(value) == {'docker'}
    assert not value.status.can_start


def test_process_supervisor_failure_has_its_own_root(monkeypatch):
    from acprof.tui.run.diagnostics import _process_checks
    def denied():
        raise RuntimeError('supervision unavailable')
    monkeypatch.setattr('acprof.host.runtimes.process_control.require_process_supervision', denied)
    monkeypatch.setattr('acprof.host.runtimes.capabilities.probe_collection_plan', lambda *a, **kw: {'probes': {}})
    rows = _process_checks(replace(RunConfig.smoke('fixture/model'), runtime='process'))
    assert [row.check_id for row in rows if row.status == 'fail'] == ['process_runtime']


def test_missing_tcpdump_and_its_uncheckable_permissions_are_one_root():
    config = replace(RunConfig.smoke('fixture/model'), runtime='docker', profiling_mode='full')
    value = controller(current=config, checks=[check('tcpdump', 'fail')], inventory=[
        check('tcpdump', 'fail'), replace(check('capture_permissions', 'fail'), source='tcpdump'),
    ])
    assert value.status.counts['error'] == 1
    assert value.status.counts['warning'] == 0
