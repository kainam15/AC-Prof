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
    assert source_group('docker') == 'docker'
    for identifier in ('cgroup', 'container_cpu', 'container_memory', 'cgroup_metrics'):
        assert source_group(identifier) == 'cgroup'
    assert source_group('cpu_energy') == 'rapl'
    assert source_group('cpu_instructions') == 'perf'
    assert source_group('config_runtime') == 'configuration'
    assert source_group('new_collector') == 'other'


def test_docker_and_cgroup_have_independent_statuses_and_impacts():
    from acprof.tui.i18n import translate
    from acprof.tui.run.environment_view import check_view, source_title
    from acprof.tui.run.source_groups import source_summaries

    rows = (('docker', 'ok'), ('cgroup', 'ok'), ('container_cpu', 'ok'),
            ('container_memory', 'ok'), ('cgroup_metrics', 'warn'))
    snapshot = GlobalStatus(tuple(StatusItem(check(identifier, status)) for identifier, status in rows))
    docker, cgroup = source_summaries(snapshot)
    assert [source.group_id for source in (docker, cgroup)] == ['docker', 'cgroup']
    assert [item.check.check_id for item in docker.items] == ['docker']
    assert [item.check.check_id for item in cgroup.items] == [identifier for identifier, _ in rows[1:]]
    assert (docker.state, docker.issue_count, docker.collapsed) == ('success', 0, True)
    assert (cgroup.state, cgroup.issue_count, cgroup.collapsed) == ('warning', 1, False)
    assert source_summaries(snapshot, group='warning') == (cgroup,)
    assert snapshot.counts['warning'] == 1
    assert translate(source_title(docker), 'zh') == 'Docker · ✓ 1'
    assert translate(source_title(cgroup), 'zh') == 'cgroup v2 · ! 1 · ✓ 3'
    impact = check_view(check('cgroup_metrics', 'warn'), technical_detail='fixture',
                        check_allowed=True).callout
    assert impact is not None
    assert 'CPU or memory metrics' in translate(impact.impact, 'en')


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


@pytest.mark.parametrize(('states', 'suffix'), [
    (('warn', 'ok', 'ok', 'ok', 'ok'), '! 1 · ✓ 4'),
    (('fail', 'warn', 'ok', 'pending', 'not_requested'), '× 1 · ! 1 · ✓ 1 · … 1 · - 1'),
    (('ok', 'ok'), '✓ 2'),
])
def test_source_title_counts_visible_checks(states, suffix):
    from acprof.tui.i18n import translate
    from acprof.tui.run.environment_view import source_status_segments, source_title
    from acprof.tui.run.source_groups import SourceSummary

    identifiers = ('gpu_device', 'gpu_memory', 'gpu_utilization', 'gpu_power', 'gpu_temperature')
    snapshot = GlobalStatus(tuple(StatusItem(check(identifier, state))
                                  for identifier, state in zip(identifiers, states, strict=False)))
    source = SourceSummary('nvml', snapshot.items, snapshot.issues)
    assert sum(int(label.split()[-1]) for _, label in source_status_segments(source)) == len(states)
    for language in ('zh', 'en'):
        assert translate(source_title(source), language) == f'NVIDIA NVML · {suffix}'


def test_same_text_does_not_merge_independent_gpu_failures():
    snapshot = GlobalStatus(tuple(StatusItem(check(identifier, 'fail', 'unavailable'))
                                  for identifier in ('gpu_memory', 'gpu_power')))
    assert snapshot.counts['error'] == 2


def test_profiler_self_checks_share_a_confirmed_docker_failure():
    from acprof.tui.run.source_groups import source_summaries

    config = replace(RunConfig.smoke('fixture/model'), runtime='docker', compute_profile_tool='both',
                     execution_profile_tool='massif')
    dependent = [replace(check(tool, 'fail'), root_cause='profiler_docker_unavailable',
                         reason_code='dependency_unavailable') for tool in ('torch', 'massif')]
    ncu = replace(check('ncu', 'fail'), reason_code='not_installed')
    value = controller(current=config, checks=[check('docker', 'fail')], inventory=[*dependent, ncu])
    assert value.status.counts['error'] == value.status.counts['warning'] == 1
    assert not value.status.can_start
    blocker, = (issue for issue in value.status.issues if issue.group == 'error')
    assert blocker.primary.check.check_id == 'docker'
    assert {item.check.check_id for item in blocker.items} == {'docker', 'torch', 'massif'}
    assert {source.group_id for source in source_summaries(value.status, group='error')} == {'docker', 'profilers'}


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


@pytest.mark.parametrize('missing', [('tcpdump',), ('tshark',), ('tcpdump', 'tshark')])
def test_packet_collector_and_missing_executables_share_the_confirmed_cause(tmp_path, monkeypatch, missing):
    from acprof.tui.run.diagnostics import _automatic_checks

    plan = {'probes': {'packet_latency': {'status': 'tool_missing', 'detail': ', '.join(missing), 'source': 'pcap'}}}
    monkeypatch.setattr('acprof.host.runtimes.capabilities.probe_collection_plan', lambda *a, **kw: plan)
    config = replace(RunConfig.smoke('fixture/model'), runtime='process')
    checks = _automatic_checks(config, 'process', None, project_dir=tmp_path)
    inventory = [replace(check(tool, 'fail'), reason_code='not_installed') for tool in missing]
    if 'tcpdump' in missing:
        inventory.append(replace(check('capture_permissions', 'fail'), source='tcpdump', reason_code='dependency_unavailable'))
    result = controller(current=config, checks=checks, inventory=inventory).status
    assert result.counts['warning'] == len(missing)
    assert result.can_start


@pytest.mark.parametrize(('runtime', 'gpu', 'expected'), [('docker', 'on', True), ('process', 'on', False),
                                                        ('docker', 'off', False), ('auto', 'on', None)])
def test_requested_scope_is_separate_from_inventory_availability(runtime, gpu, expected):
    from acprof.tui.i18n import translate
    from acprof.tui.run.environment_view import global_check_view

    config = replace(RunConfig.smoke('fixture/model'), runtime=runtime, gpus=gpu, profiling_mode='full')
    value = controller(current=config, inventory=[check('gpu_power', 'fail')])
    item = next(item for item in value.status.items if item.check.check_id == 'gpu_power')
    assert item.requested is expected
    assert item.check.status == 'warn'
    assert not item.blocks
    view = global_check_view(item, technical_detail='raw', tr=lambda text: translate(text, 'en'),
                             check_allowed=True, navigate_allowed=True)
    assert ('not requested' in translate(view.callout.impact, 'en').lower()) == (expected is False)
    assert 'gpu_energy_total_j' not in translate(view.callout.impact, 'en')
    assert 'gpu_energy_total_j' in view.callout.detail


def test_affected_fields_are_not_duplicated_in_technical_details():
    from acprof.tui.i18n import translate
    from acprof.tui.run.environment_view import global_check_view

    row = replace(check('gpu_power', 'warn'), affected_fields=('gpu_energy_total_j',))
    item = StatusItem(row, requested=True)
    for technical_detail in ('raw diagnostic', 'raw diagnostic: gpu_energy_total_j'):
        view = global_check_view(item, technical_detail=technical_detail,
                                 tr=lambda text: translate(text, 'zh'),
                                 check_allowed=True, navigate_allowed=True)
        assert 'gpu_energy_total_j' not in translate(view.callout.impact, 'zh')
        assert view.callout.detail.count('gpu_energy_total_j') == 1



@pytest.mark.parametrize(('environment', 'fragment'), [('native_linux', 'Linux host'), ('wsl2', 'WSL Linux')])
def test_missing_tool_repair_uses_platform_and_keeps_failure_reason(environment, fragment):
    from acprof.platform import Environment
    from acprof.tui.i18n import translate
    from acprof.tui.run.environment_view import global_check_view

    row = replace(check('tcpdump', 'fail'), reason_code='not_installed')
    view = global_check_view(StatusItem(row, required=True), technical_detail='raw', tr=lambda value: translate(value, 'en'),
                             check_allowed=True, navigate_allowed=True, environment=Environment(environment, system='Linux'))
    assert translate(view.callout.problem, 'en') == '× tcpdump · Not installed\nfixture'
    assert fragment in translate(view.callout.next_step, 'en')
    assert 'sudo apt install tcpdump tshark' in translate(view.callout.next_step, 'en')


def test_affected_fields_are_registered_protocol_fields():
    from acprof.results.metrics import METRICS
    from acprof.tui.run.environment_view import CHECK_FIELDS

    assert {field for fields in CHECK_FIELDS.values() for field in fields} <= METRICS.keys()


@pytest.mark.parametrize(('identifier', 'source', 'root', 'dependency'), [
    ('torch', 'runtime self-check', 'profiler_docker_unavailable', 'Docker Engine'),
    ('massif', 'runtime self-check', 'profiler_docker_unavailable', 'Docker Engine'),
    ('capture_permissions', 'tcpdump', '', 'tcpdump'),
    ('capture_permissions', 'getcap', '', 'getcap'),
    ('capture_interface', 'ip', '', 'iproute2'),
    ('gpu_device', 'nvidia-smi', '', 'nvidia-smi'),
])
def test_missing_dependency_identifies_what_to_restore_on_the_host(identifier, source, root, dependency):
    from acprof.tui.i18n import translate
    from acprof.tui.run.environment_view import check_view

    row = replace(check(identifier, 'warn'), reason_code='dependency_unavailable', root_cause=root, source=source)
    view = check_view(row, technical_detail=row.detail, check_allowed=True)
    for language, host in [('en', 'host'), ('zh', '主机')]:
        remedy = translate(view.callout.next_step, language)
        assert dependency in remedy
        assert host in remedy
        if root:
            assert ('image' if language == 'en' else '镜像') in remedy
