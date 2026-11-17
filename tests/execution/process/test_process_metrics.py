import math

import pytest

from acprof.monitors.process_usage import reduce_process_samples


def test_missing_cpu_counter_preserves_rss_and_gpu():
    from acprof.monitors.process_usage import reduce_process_samples
    rows = [dict(timestamp=t, members={'root': {'cpu_s': None, 'rss': rss}},
                 gpu_util_pct=20, gpu_mem_bytes=1000) for t, rss in [(1, 100), (2, 300)]]
    values = reduce_process_samples(rows)
    assert 'process_cpu_time_s' not in values
    assert values['process_rss_avg_bytes'] == 200
    assert values['gpu_mem_used_avg_bytes'] == 1000


def test_empty_process_snapshot_never_becomes_zero_resource_usage():
    from acprof.monitors.process_usage import reduce_process_samples
    values = reduce_process_samples([dict(timestamp=t, members={}, gpu_util_pct=20, gpu_mem_bytes=1000) for t in (1, 2)])
    assert not any(name.startswith('process_') for name in values)
    assert values['gpu_util_avg_pct'] == 20


def sample(timestamp, *members):
    return {"timestamp": timestamp,
            "members": {key: {"cpu_s": cpu, "rss": rss} for key, cpu, rss in members}}


def test_process_cpu_percent_is_one_core_and_rss_is_time_weighted():
    values = reduce_process_samples([
        sample(10, ("1:42", 3, 100)), sample(12, ("1:42", 6, 300)),
        sample(13, ("1:42", 7.5, 500)),
    ])
    assert values["process_cpu_time_s"] == 4.5
    assert values["process_cpu_util_avg_pct"] == 150
    assert math.isclose(values["process_rss_avg_bytes"], 800 / 3)
    assert values["process_rss_peak_bytes"] == 500


def test_new_pid_identity_does_not_import_old_cpu_time():
    values = reduce_process_samples([sample(1, ("1:42", 10, 100)), sample(2, ("1:43", 99, 100))])
    assert 'process_cpu_time_s' not in values


def test_counter_decrease_is_not_reported_as_zero():
    with pytest.raises(ValueError, match="decreased"):
        reduce_process_samples([sample(1, ("1:42", 10, 100)), sample(2, ("1:42", 9, 100))], legacy=True)


def test_new_cpu_reset_keeps_rss_with_metric_error():
    from acprof.monitors.process_usage import ProcessUsageMonitor
    monitor = ProcessUsageMonitor(pid=42, start_ticks=1)
    monitor.samples = [sample(1, ("1:42", 10, 100)), sample(2, ("1:42", 9, 100))]
    result, error, _ = monitor.finish_samples()
    assert math.isnan(result.process_cpu_time_s)
    assert result.process_rss_avg_bytes == 100
    assert result.metric_evidence['process_cpu_time_s']['quality'] == 'error'
    assert 'counter_decreased' in error


def test_boundary_counters_do_not_need_periodic_io_or_fd_reads():
    rows = [sample(1, ("1:42", 10, 100)), sample(2, ("1:42", 11, 200)),
            sample(3, ("1:42", 12, 300))]
    rows[0]['boundary'] = rows[-1]['boundary'] = True
    rows[0]['members']['1:42'].update(cpu_user_s=6, cpu_system_s=4,
        io_read_bytes=1000, io_write_bytes=100, io_read_ops=10, io_write_ops=2,
        voluntary_ctx=10, involuntary_ctx=5, minor_faults=3, major_faults=1,
        swap_bytes=100, threads=2, fds=3)
    rows[1]['members']['1:42'].update(cpu_user_s=6.75, cpu_system_s=4.25)
    rows[2]['members']['1:42'].update(cpu_user_s=7.5, cpu_system_s=4.5,
        io_read_bytes=1512, io_write_bytes=100, io_read_ops=12, io_write_ops=2,
        voluntary_ctx=14, involuntary_ctx=6, minor_faults=10, major_faults=1,
        swap_bytes=200, threads=3, fds=4)
    values = reduce_process_samples(rows)
    assert values['process_cpu_user_time_s'] == 1.5
    assert values['process_cpu_system_time_s'] == .5
    assert values['process_io_read_bytes'] == 512
    assert values['process_io_write_bytes'] == 0
    assert values['process_io_read_ops'] == 2
    assert values['process_io_write_ops'] == 0
    assert values['process_leader_voluntary_ctx_switches'] == 4
    assert values['process_leader_involuntary_ctx_switches'] == 1
    assert values['process_minor_faults'] == 7
    assert values['process_major_faults'] == 0
    assert values['process_swap_bytes_end'] == 200
    assert values['process_count_end'] == 1
    assert values['process_threads_end'] == 3
    assert values['process_fds_end'] == 4
    assert values['process_rss_growth_bytes'] == 200


def test_boundary_counter_identity_change_is_unknown_not_zero():
    rows = [sample(1, ('1:42', 10, 100)), sample(2, ('1:42', 11, 200), ('2:99', 50, 100))]
    for row in rows:
        row['boundary'] = True
        for member in row['members'].values():
            member.update(io_read_bytes=100, io_write_bytes=200)
    values = reduce_process_samples(rows)
    assert 'process_io_read_bytes' not in values
    assert 'process_io_write_bytes' not in values
    assert values['process_count_end'] == 2
    assert values['process_rss_peak_bytes'] == 300


def test_optional_counter_reset_preserves_other_metrics():
    rows = [sample(1, ('1:42', 10, 100)), sample(2, ('1:42', 11, 200))]
    for row, reads, writes in zip(rows, (100, 99), (200, 250), strict=True):
        row['boundary'] = True
        row['members']['1:42'].update(io_read_bytes=reads, io_write_bytes=writes)
    values = reduce_process_samples(rows)
    assert 'process_io_read_bytes' not in values
    assert values['process_io_write_bytes'] == 50
    assert values['process_cpu_time_s'] == 1
