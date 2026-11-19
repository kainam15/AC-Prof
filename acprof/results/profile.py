"""Read-only metric provenance shared by audit, statistics and presentation."""
from __future__ import annotations

import json
from functools import lru_cache

from acprof.results.coverage import collector_contract
from acprof.results.metrics import METRICS

QUALITIES = {'complete', 'partial', 'unsupported', 'permission_denied', 'not_requested', 'error', 'unavailable', 'unknown'}
USABLE_QUALITIES = frozenset({'complete', 'unknown'})
RESOURCE_SOURCES = {'psutil_observed_process_tree', 'procfs_observed_process_tree', 'service_cgroup', 'nvml_selected_device'}


def observation_signature(observation: dict) -> str:
    return json.dumps({key: observation[key] for key in ('source', 'scope', 'window', 'derivation', 'sampling')
                       if key in observation}, sort_keys=True, allow_nan=False)


def observations_comparable(left: list[dict], right: list[dict]) -> bool:
    return (all(item.get('quality') in USABLE_QUALITIES for item in [*left, *right])
            and len({observation_signature(item) for item in [*left, *right]}) <= 1
            and bool(left) == bool(right))


@lru_cache(maxsize=128)
def _decode(text: str) -> dict:
    if text.strip().lower() in {'', 'nan', 'null', 'none'}:
        return {}
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('resource_metric_quality must be an object')
    if 'schema_version' in value:
        if type(value['schema_version']) is not int or value['schema_version'] != 1:
            raise ValueError('unsupported resource metric quality version')
        value = value.get('metrics')
    if not isinstance(value, dict) or any(not isinstance(item, dict) for item in value.values()):
        raise ValueError('invalid resource metric evidence')
    return value


def metric_observation(row: dict, name: str) -> dict:
    metric = METRICS[name]
    contract = collector_contract(metric)
    result = {'source': metric.source, 'scope': contract.scope, 'window': metric.window,
              'derivation': contract.derivation, 'quality': 'unknown', 'reason': 'no_metric_quality_evidence'}
    if metric.source not in RESOURCE_SOURCES and name not in {"request_payload_bytes", "response_payload_bytes"}:
        return result
    try:
        item = _decode(str(row.get('resource_metric_quality', ''))).get(name)
        if item is None:
            return result
        if item.get('quality') not in QUALITIES:
            raise ValueError('invalid metric quality')
        if any(key in item and item[key] != result[key] for key in ('source', 'scope', 'window', 'derivation')):
            raise ValueError('metric source contract conflicts with registry')
        observation_signature(item)  # Reject non-finite sampling metadata before downstream hashing.
        if not isinstance(item.get('reason', ''), str):
            raise ValueError('metric reason must be text')
        return {**result, 'reason': '', **item}
    except (TypeError, ValueError) as exc:
        return {**result, 'quality': 'error', 'reason': str(exc)}


def metric_usable(row: dict, name: str) -> bool:
    """Preserve legacy analysis; explicit incomplete evidence never becomes a valid value."""
    return name not in METRICS or metric_observation(row, name)['quality'] in USABLE_QUALITIES
