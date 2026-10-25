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
    assert values["process_cpu_time_s"] == 0


def test_counter_decrease_is_not_reported_as_zero():
    with pytest.raises(ValueError, match="decreased"):
        reduce_process_samples([sample(1, ("1:42", 10, 100)), sample(2, ("1:42", 9, 100))])
