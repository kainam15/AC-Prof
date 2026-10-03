"""Optional probe limit evidence remains separate from actual truncation."""
import pytest

from acprof.host.input_plan import _parse_probe_response, _probe_exceeds_limit


def test_legacy_probe_response_remains_usable():
    response = _parse_probe_response({'effective_input_scale': 8, 'truncated_by_limit': False,
                                      'reason': 'within model limit'}, 'legacy')
    assert not (response['limit_exceeded'])
    assert not (_probe_exceeds_limit(response))

@pytest.mark.parametrize('truncated,exceeded', ((True, False), (False, True)))
def test_rejected_input_and_actual_truncation_are_both_unusable_but_distinct(truncated, exceeded):
    response = _parse_probe_response({'effective_input_scale': 20,
        'truncated_by_limit': truncated, 'limit_exceeded': exceeded,
        'reason': 'model input constraint'}, 'candidate')
    assert (response['truncated_by_limit']) == (truncated)
    assert (response['limit_exceeded']) == (exceeded)
    assert (_probe_exceeds_limit(response))

@pytest.mark.parametrize('value', ('false', 0, None))
def test_malformed_optional_limit_flag_is_not_coerced_to_success(value):
    with pytest.raises(RuntimeError, match='non-boolean limit_exceeded'):
        _parse_probe_response({'effective_input_scale': 20, 'truncated_by_limit': False,
                               'limit_exceeded': value, 'reason': 'bad flag'}, 'candidate')
