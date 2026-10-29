"""Finite abstract values, merge and analysis budgets for static dependency proofs."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Unknown(Enum):
    VALUE = "unknown"


UNKNOWN = Unknown.VALUE
MAX_CALL_DEPTH = 12
MAX_CALL_CONTEXTS = 256
MAX_CANDIDATES = 512
MAX_ANALYSIS_STEPS = 100000


@dataclass(frozen=True)
class Symbol:
    name: str


@dataclass
class Instance:
    name: str
    attributes: dict = field(default_factory=dict)


def activation(parent: str, condition) -> str:
    if parent == "inactive" or condition is False:
        return "inactive"
    return parent if condition is True else "unknown"


def known_truth(value):
    return UNKNOWN if value is UNKNOWN or isinstance(value, Symbol) else bool(value)


def merge(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return {key: merge(left.get(key, UNKNOWN), right.get(key, UNKNOWN)) for key in left.keys() | right.keys()}
    if isinstance(left, Instance) and isinstance(right, Instance) and left.name == right.name:
        return Instance(left.name, merge(left.attributes, right.attributes))
    return left if type(left) is type(right) and left == right else UNKNOWN
