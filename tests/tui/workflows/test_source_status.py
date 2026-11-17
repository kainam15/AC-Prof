"""Source summaries retain checks and never grant collection admission."""
from dataclasses import replace

import pytest

from acprof.experiments.configuration import RunConfig
from acprof.tui.run.global_status import GlobalStatus, StatusItem
from tests.tui.workflows.test_global_status import check, controller


def test_shared_capture_failure_retains_both_original_checks():
    dependency = replace(check('capture_permissions', 'fail'), source='tcpdump')
    value = controller(current=replace(RunConfig.smoke('fixture/model'), runtime='docker', profiling_mode='full'),
                       checks=[check('tcpdump', 'fail')], inventory=[dependency])
    assert {item.check.check_id for item in value.status.items} >= {'tcpdump', 'capture_permissions'}
    assert value.status.counts['error'] == 1
    assert value.status.counts['warning'] == 0
    assert not value.status.can_start


def test_every_inventory_check_has_a_source_independent_of_provenance():
    from acprof.tui.run.environment_checks import ENVIRONMENT_CHECK_IDS
    from acprof.tui.run.source_groups import source_group

    for identifier in ENVIRONMENT_CHECK_IDS:
        assert source_group(identifier) != 'other'
    assert source_group('process_cpu') == 'process'
    assert source_group('container_cpu') == 'docker'
    assert source_group('cpu_energy') == 'rapl'
    assert source_group('cpu_instructions') == 'perf'
    assert source_group('config_runtime') == 'configuration'
    assert source_group('new_collector') == 'other'


@pytest.mark.parametrize(('states', 'expected'), [
    (('ok', 'ok'), 'success'), (('fail', 'fail'), 'error'),
    (('ok', 'fail'), 'error'), (('ok', 'warn'), 'warning'),
    (('ok', 'checking'), 'checking'), (('ok', 'stale'), 'pending'),
    (('ok', 'declared'), 'pending'), (('ok', 'pending'), 'pending'),
    (('not_requested', 'not_requested'), 'not_requested'),
    (('ok', 'not_requested'), 'success'), (('ok', 'future'), 'pending'),
])
def test_source_summary_uses_all_children(states, expected):
    from acprof.tui.run.source_groups import source_summaries

    snapshot = GlobalStatus(tuple(StatusItem(check(identifier, state), required=True)
                                  for identifier, state in zip(('gpu_memory', 'gpu_power'), states, strict=True)))
    before = snapshot.can_start
    source, = source_summaries(snapshot)
    assert source.group_id == 'nvml'
    assert source.state == expected
    assert source.items == snapshot.items
    assert snapshot.can_start == before
    assert source.collapsed == (expected not in {'error', 'warning'})


def test_same_text_does_not_merge_independent_gpu_failures():
    snapshot = GlobalStatus(tuple(StatusItem(check(identifier, 'fail', 'unavailable'))
                                  for identifier in ('gpu_memory', 'gpu_power')))
    assert snapshot.counts['error'] == 2


def test_filter_keeps_successful_siblings_and_full_source_severity():
    from acprof.tui.run.source_groups import source_summaries

    snapshot = GlobalStatus((StatusItem(check('gpu_memory', 'ok')), StatusItem(check('gpu_power', 'warn'))))
    source, = source_summaries(snapshot, group='success')
    assert source.state == 'warning'
    assert [item.check.check_id for item in source.items] == ['gpu_memory', 'gpu_power']
    assert source.issue_count == 1


def test_declared_dependency_merges_issues_without_merging_admission_or_results():
    from acprof.tui.run.source_groups import source_summaries

    rows = tuple(replace(check(identifier, 'fail'), root_cause='nvml_initialization')
                 for identifier in ('gpu_device', 'gpu_memory', 'gpu_utilization', 'gpu_power'))
    snapshot = GlobalStatus(tuple(StatusItem(row, required=True) for row in rows))
    assert snapshot.counts == {'error': 1, 'warning': 0, 'success': 0, 'pending': 0, 'not_requested': 0}
    assert len(snapshot.blockers) == 4
    assert tuple(item.check for item in snapshot.items) == rows
    assert sum(len(issue.items) for issue in snapshot.issues) == len(rows)
    source, = source_summaries(snapshot)
    assert source.issue_count == 1
    assert source.state == 'error'


def test_permission_denial_is_not_the_missing_tool_dependency():
    rows = (check('tcpdump', 'fail'),
            replace(check('capture_permissions', 'fail'), source='tcpdump', capability_status='permission_denied'))
    assert GlobalStatus(tuple(StatusItem(row) for row in rows)).counts['error'] == 2


@pytest.mark.parametrize(('runtime', 'gpu', 'expected'), [('docker', 'on', True), ('process', 'on', False),
                                                        ('docker', 'off', False), ('auto', 'on', None)])
def test_requested_scope_is_separate_from_inventory_availability(runtime, gpu, expected):
    from acprof.tui.i18n import translate
    from acprof.tui.run.environment_view import source_overview
    from acprof.tui.run.source_groups import source_summaries

    config = replace(RunConfig.smoke('fixture/model'), runtime=runtime, gpus=gpu, profiling_mode='full')
    value = controller(current=config, inventory=[check('gpu_power', 'fail')])
    item = next(item for item in value.status.items if item.check.check_id == 'gpu_power')
    assert item.requested is expected
    assert item.check.status == 'warn'
    assert not item.blocks
    source = next(source for source in source_summaries(value.status) if source.group_id == 'nvml')
    explanation = translate(source_overview(source), 'en')
    assert ('gpu_energy_total_j' in explanation) == (expected is True)
    assert 'does not prove collection success' in explanation


@pytest.mark.parametrize(('environment', 'fragment'), [('native_linux', 'Linux host'), ('wsl2', 'WSL Linux')])
def test_missing_tool_repair_uses_platform_and_keeps_failure_reason(environment, fragment):
    from acprof.platform import Environment
    from acprof.tui.i18n import translate
    from acprof.tui.run.environment_view import global_check_view

    row = replace(check('tcpdump', 'fail'), reason_code='not_installed')
    view = global_check_view(StatusItem(row, required=True), technical_detail='raw', cause='missing',
                             check_allowed=True, navigate_allowed=True, environment=Environment(environment, system='Linux'))
    assert 'Tool not installed' in translate(view.callout.problem, 'en')
    assert fragment in translate(view.callout.next_step, 'en')
    assert 'sudo apt install tcpdump tshark' in translate(view.callout.next_step, 'en')


def test_affected_fields_are_registered_protocol_fields():
    from acprof.results.metrics import METRICS
    from acprof.tui.run.environment_view import CHECK_FIELDS

    assert {field for fields in CHECK_FIELDS.values() for field in fields} <= METRICS.keys()
