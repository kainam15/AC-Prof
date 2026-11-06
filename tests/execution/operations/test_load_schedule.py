"""Reject unrepresentable arrival plans before any load request is sent."""
import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.cli import load as load_cli
from acprof.host.measurement import load
from acprof.host.measurement.streaming import OpenAIStreamingAdapter


@pytest.mark.parametrize(('arrival', 'rate', 'requests', 'seed'), [
    ('fixed', 1e-320, 2, 0),
    ('poisson', 1e-320, 2, 0),
    ('fixed', 1e-308, 3, 0),
    ('poisson', 1e-308, 5, 3),
])
def test_nonfinite_generated_schedule_is_rejected(arrival, rate, requests, seed):
    config = load.LoadConfig(scenario='arrival-rate', arrival=arrival,
                             rate=rate, requests=requests, seed=seed)
    with pytest.raises(ValueError, match='arrival-rate schedule.*finite'):
        config.schedule()


@pytest.mark.parametrize('arrival', ['fixed', 'poisson'])
@pytest.mark.parametrize('streaming', [False, True])
def test_invalid_schedule_never_constructs_transport_or_workers(monkeypatch, arrival, streaming):
    transport = Mock(side_effect=AssertionError('invalid schedule reached transport'))
    workers = Mock(side_effect=AssertionError('invalid schedule started workers'))
    monkeypatch.setattr('acprof.host.measurement.load.LoadTransport', transport)
    monkeypatch.setattr('acprof.host.measurement.load.ThreadPoolExecutor', workers)
    config = load.LoadConfig(scenario='arrival-rate', rate=1e-320, requests=2, arrival=arrival)
    adapter = OpenAIStreamingAdapter() if streaming else None
    payload = {'model': 'fixture', 'messages': [{'role': 'user', 'content': 'hello'}]}

    with pytest.raises(ValueError, match='arrival-rate schedule.*finite'):
        load.run_load('http://127.0.0.1:8999', payload, config, token='invalid', adapter=adapter)

    transport.assert_not_called()
    workers.assert_not_called()


@pytest.mark.parametrize('arrival', ['fixed', 'poisson'])
def test_replay_rejects_invalid_schedule_before_input_or_runtime_preparation(
    tmp_path, monkeypatch, capsys, arrival,
):
    source, output = tmp_path / 'source', tmp_path / 'output'
    state = {'status': 'complete', 'options': {'cpus': '1', 'mems': '2'}, 'runtime': {'kind': 'docker'}}
    monkeypatch.setattr('acprof.cli.load.bootstrap_project_env', lambda _root: None)
    monkeypatch.setattr('acprof.cli.load.resolve_single_result_source', lambda _source: source)
    monkeypatch.setattr('acprof.cli.load.load_run_state', lambda _source: state)
    monkeypatch.setattr('acprof.cli.load.ExecutionConditions', SimpleNamespace(
        from_options=lambda *_args, **_kwargs: SimpleNamespace(request_timeout_seconds=1),
    ))
    preparation = Mock(side_effect=AssertionError('invalid schedule reached input preparation'))
    runtime = Mock(side_effect=AssertionError('invalid schedule reached runtime'))
    monkeypatch.setattr('acprof.cli.load.ArtifactLayout.discover', preparation)
    monkeypatch.setattr('acprof.cli.load.execution_runtime', runtime)

    with pytest.raises(SystemExit) as caught:
        load_cli.main([str(source), '--gpu', 'off', '--output-dir', str(output),
                       '--scenario', 'arrival-rate', '--rate', '1e-320', '--requests', '2',
                       '--arrival', arrival])

    assert caught.value.code == 2
    assert 'arrival-rate schedule' in capsys.readouterr().err
    assert not output.exists()
    preparation.assert_not_called()
    runtime.assert_not_called()


def test_finite_fixed_schedule_preserves_exact_offsets_and_protocol():
    config = load.LoadConfig(scenario='arrival-rate', rate=2, requests=4)
    identity = config.identity()
    assert config.schedule() == [0.0, 0.5, 1.0, 1.5]
    assert config.identity() == identity


def test_finite_poisson_schedule_preserves_seeded_arrivals_without_rescaling():
    config = load.LoadConfig(scenario='arrival-rate', rate=2, requests=4, arrival='poisson', seed=7)
    # Independent inverse-CDF values for seed 7's first three uniform draws.
    expected = [0.0, 0.19565742211740214, 0.2774166506201412, 0.8036644596556046]
    assert config.schedule() == pytest.approx(expected)
    assert config.schedule() == pytest.approx(expected)


@pytest.mark.parametrize('arrival', ['fixed', 'poisson'])
def test_one_request_needs_no_interarrival_even_for_a_very_small_rate(arrival):
    config = load.LoadConfig(scenario='arrival-rate', rate=1e-320, requests=1, arrival=arrival)
    assert config.schedule() == [0.0]


@pytest.mark.parametrize('scenario', ['serial', 'concurrent'])
def test_closed_loop_still_has_no_arrival_plan(scenario):
    assert load.LoadConfig(scenario=scenario).schedule() is None


def test_overflow_stops_generation_at_the_first_nonfinite_offset(monkeypatch):
    generator = random.Random(0)
    draw = Mock(wraps=generator.expovariate)
    monkeypatch.setattr('acprof.experiments.load.random.Random',
                        lambda _seed: SimpleNamespace(expovariate=draw))
    config = load.LoadConfig(scenario='arrival-rate', rate=1e-320, requests=100, arrival='poisson')
    with pytest.raises(ValueError, match='arrival-rate schedule.*finite'):
        config.schedule()
    draw.assert_called_once_with(1e-320)
