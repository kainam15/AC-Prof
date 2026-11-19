"""Global counts and admission must describe the same current configuration."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from acprof.experiments.configuration import ConfigIssue, RunConfig, RunConfigError
from acprof.tui.run.diagnostics import PreflightCheck
from acprof.tui.run.environment_view import check_group
from acprof.tui.run.preflight_controller import PreflightController
from acprof.tui.ui.environment import display_checks


def controller(*, checks=None, inventory=(), issues=(), current=None, checked=None):
    config = current or replace(RunConfig.smoke('fixture/model'), runtime='process')
    app = SimpleNamespace(
        is_running=True, form_ready=True, ui_closing=False, tr=str,
        configuration=SimpleNamespace(config_issues=issues, config_warnings=(), validation_pending=False),
    )
    result = PreflightController(app)
    result.preflight_config = lambda: config
    result.check_completed = True
    result.checked_config = checked or config
    result.checks = (check('environment', 'ok'),) if checks is None else tuple(checks)
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
    value = controller(current=replace(RunConfig.smoke('fixture/model'), runtime='process', profiling_mode='basic'),
                       inventory=[check(identifier, 'fail')])
    assert not errors(value)
    row = next(row for row in display_checks(value) if row.check_id == identifier)
    assert row.status == 'warn'
    assert row.detail == 'fixture'
    assert not value.preflight_run_reason()


def test_combined_profilers_do_not_request_posthoc_advisor():
    config = replace(RunConfig.smoke('fixture/model'), runtime='docker', execution_profile_tool='both')
    value = controller(current=config, inventory=[check('intel_advisor', 'fail')])
    assert next(row for row in display_checks(value) if row.check_id == 'intel_advisor').status == 'warn'
    assert value.status.counts['warning'] == 0
    assert value.status.can_start


@pytest.mark.parametrize('runtime', ['docker', 'process', 'auto'])
def test_profiler_warnings_follow_selection_without_losing_inventory(runtime):
    config = replace(RunConfig.smoke('fixture/model'), runtime=runtime, gpus='on')
    ncu = replace(check('ncu', 'fail'), reason_code='not_installed')
    for selection, group in [('none', 'not_requested'), ('ncu', 'warning'), ('both', 'warning')]:
        selected = replace(config, compute_profile_tool=selection)
        issues = ()
        if runtime == 'process' and selection != 'none':
            with pytest.raises(RunConfigError) as invalid:
                selected.validate()
            issues = invalid.value.issues
        value = controller(current=selected, inventory=[ncu], issues=issues)
        item = next(item for item in value.status.items if item.check.check_id == 'ncu')
        assert item.group == group
        assert value.status.counts['warning'] == int(group == 'warning')
        assert item.check.status == 'warn'
        assert item.check.reason_code == 'not_installed'
        assert value.status.can_start == (not issues)


@pytest.mark.parametrize('identifier', [
    'uv', 'process_runtime', 'process_cpu', 'process_memory',
    'ncu', 'nsys', 'intel_advisor', 'torch', 'massif',
])
@pytest.mark.parametrize('status', ['ok', 'pending'])
def test_unselected_inventory_retains_availability_and_unverified_results(identifier, status):
    config = replace(RunConfig.smoke('fixture/model'), runtime='docker')
    evidence = replace(check(identifier, status, 'host evidence'), source='inventory fixture')
    value = controller(current=config, inventory=[evidence])
    item = next(item for item in value.status.items if item.check.check_id == identifier)
    assert item.check == evidence
    assert not item.required
    assert value.status.counts['not_requested'] == 0
    assert value.status.counts['pending'] == (1 if status == 'pending' else 0)
    assert value.status.can_start


@pytest.mark.parametrize('status', ['ok', 'fail', 'pending'])
def test_unrequested_preflight_placeholder_does_not_replace_machine_evidence(status):
    evidence = replace(check('cpu_energy', status, 'host RAPL evidence'), source='rapl_probe')
    value = controller(checks=[check('environment', 'ok'),
                               check('cpu_energy', 'not_requested', 'Process basic')],
                       inventory=[evidence])
    item = next(item for item in value.status.items if item.check.check_id == 'cpu_energy')
    assert item.check == replace(evidence, status='warn' if status == 'fail' else status)
    assert not item.required
    assert value.status.can_start
    assert sum(row.check_id == 'cpu_energy' for row in display_checks(value)) == 1


def test_deselected_requirement_uses_inventory_during_config_recheck():
    previous = replace(RunConfig.smoke('fixture/model'), runtime='docker', compute_profile_tool='ncu')
    value = controller(current=replace(previous, compute_profile_tool='none'), checked=previous,
                       checks=[check('ncu', 'fail', 'previous request failed')],
                       inventory=[check('ncu', 'ok', 'host version verified')])
    value.check_running = True
    value.check_inventory = False
    item = next(item for item in value.status.items if item.check.check_id == 'ncu')
    assert item.check.status == 'ok'
    assert item.check.detail == 'host version verified'
    assert not item.required
    assert not value.status.can_start  # The changed experiment still needs preflight.


def test_inventory_refresh_keeps_unselected_checks_pending_until_complete():
    value = controller(inventory=[check('ncu', 'ok')])
    value.check_running = True
    value.check_inventory = True
    item = next(item for item in value.status.items if item.check.check_id == 'ncu')
    assert item.check.status == 'checking'
    assert item.group == 'pending'
    assert not item.required


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
    config = replace(RunConfig.smoke('fixture/model'), runtime='process', profiling_mode='full', compute_profile_tool='torch')
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


@pytest.mark.parametrize(('identifier', 'target'), [
    ('cpu_instructions', 'permissions:perf'), ('capture_permissions', 'permissions:tcpdump'),
    ('cpu_energy', 'diagnostic:rapl'), ('docker', 'diagnostic:docker'),
    ('process_resources', 'resources'), ('unknown_check', ''),
])
def test_issue_navigation_has_specific_targets_and_no_permission_fallback(identifier, target):
    value = controller(checks=[check(identifier, 'fail')])
    item = next(item for item in value.status.items if item.check.check_id == identifier)
    assert item.target == target


def test_empty_preflight_cannot_prove_required_capabilities():
    value = controller(checks=[])
    assert not value.status.can_start
    assert value.status.counts['pending'] > 0


def permission_controller():
    value = controller(current=replace(RunConfig.smoke('fixture/model'), runtime='docker', profiling_mode='full'),
                       checks=[check('docker', 'ok'), check('cpu_instructions', 'ok')])
    value.app.execution = SimpleNamespace(session=SimpleNamespace(snapshot=SimpleNamespace(measurement_active=False)))
    value.refresh_preflight_state = lambda *args: None
    return value


def test_permission_recheck_without_evidence_keeps_required_capability_pending():
    value = permission_controller()
    token = value.begin_permission_check()
    assert token is not None
    value.finish_permission_check([], 'probe failed', token)
    assert not value.status.can_start
    assert next(row for row in value.checks if row.check_id == 'cpu_instructions').status == 'pending'
    assert next(row for row in value.checks if row.check_id == 'docker').status == 'ok'
    assert any(row.check.detail == 'probe failed' for row in value.status.items)


def test_late_permission_result_cannot_replace_new_request_or_config():
    value = permission_controller()
    token = value.begin_permission_check()
    value.finish_permission_check([check('cpu_instructions', 'ok')], '', object())
    assert value.check_running
    assert next(row for row in value.checks if row.check_id == 'cpu_instructions').status == 'pending'
    current = replace(value.preflight_config(), sniff_iface='fixture-new-interface')
    value.preflight_config = lambda: current
    value.finish_permission_check([check('cpu_instructions', 'ok')], '', token)
    assert not value.check_running
    assert not value.status.can_start
    assert next(row for row in value.checks if row.check_id == 'cpu_instructions').status == 'pending'
    assert next(row for row in value.environment_checks if row.check_id == 'cpu_instructions').status == 'ok'


def test_missing_tool_replaces_dependent_permission_failure():
    value = permission_controller()
    value.checks = (check('capture_permissions', 'fail'),)
    token = value.begin_permission_check()
    value.finish_permission_check([check('tcpdump', 'fail')], '', token)
    assert errors(value) == {'tcpdump'}
    assert value.status.counts['error'] == 1


def test_permission_actions_do_not_invalidate_or_start_during_measurement():
    value = permission_controller()
    value.app.execution.session.snapshot.measurement_active = True
    before = value.checks
    assert value.begin_permission_check() is None
    value.invalidate_permission_checks()
    assert value.checks is before
    assert value.check_request is None
    assert not value.check_running
