"""Read-only schema and artifact routing for recorded hardware conditions.

Host observation lives in ``acprof.host.hardware_conditions``; offline analysis
must not import the collectors just to locate historic evidence.
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from acprof.artifacts import finite_json_decimal
from acprof.results.layout import ArtifactLayout

HARDWARE_FIELDS = ("host_id", "cpu_model", "cpu_affinity", "cpu_policy", "gpu", "runtime_threads")


def conditions_path(layout: ArtifactLayout) -> Path:
    """Locate optional case conditions without changing legacy layout semantics."""
    name = "metadata/hardware_conditions.json" if layout.layout_version == 2 else "hardware_conditions.json"
    return layout.contained(name)


def decode_hardware_conditions(text: str) -> object:
    """Keep ordinary JSON numbers and preserve quota tokens that would lose precision."""
    inexact = False

    def number(token):
        nonlocal inexact
        exact = finite_json_decimal(token)
        display = float(exact)
        # A float representing an integer must match the original integer exactly.
        if exact == (Decimal.from_float(display) if display.is_integer() else Decimal(str(display))):
            return display
        inexact = True
        return exact

    def restore_numbers(container, path):
        quota_group = (len(path) == 4 and path[0] == "cases" and path[2] == "resource_plan"
                       and path[3] in ("requested", "observed"))
        items = container.items() if isinstance(container, dict) else enumerate(container)
        for key, item in items:
            if isinstance(item, Decimal):
                if quota_group and key in ("cpu_quota", "memory_gib"):
                    container[key] = int(item) if item == item.to_integral_value() else str(item)
                else:
                    container[key] = float(item)
            elif isinstance(item, (dict, list)):
                restore_numbers(item, (*path, key))

    payload = json.loads(text, parse_float=number, parse_constant=number)
    if inexact and isinstance(payload, dict):
        restore_numbers(payload, ())
    return payload
