"""Dependency analysis layers enforce budgets and unknown-branch evidence."""
import ast

import pytest

from acprof.models.inspection.dependency_domain import (
    MAX_ANALYSIS_STEPS,
    MAX_CALL_CONTEXTS,
    MAX_CALL_DEPTH,
    MAX_CANDIDATES,
    UNKNOWN,
    Instance,
    Symbol,
    activation,
    known_truth,
    merge,
)
from acprof.models.inspection.dependency_eval import DependencyEvaluationMixin
from acprof.models.inspection.dependency_flow import DependencyAnalyzer, analyze_dependencies


def test_abstract_value_merge_is_conservative():
    assert activation("active", UNKNOWN) == "unknown"
    assert activation("active", False) == "inactive"
    assert activation("inactive", True) == "inactive"
    assert known_truth(Symbol("unknown")) is UNKNOWN
    assert merge({"repo": "a"}, {"repo": "b"})["repo"] is UNKNOWN
    obj = merge(Instance("loader", {"x": 1}), Instance("loader", {"x": 2}))
    assert isinstance(obj, Instance)
    assert obj.attributes["x"] is UNKNOWN


def test_interpreter_requires_a_concrete_evidence_owner():
    with pytest.raises(TypeError):
        DependencyEvaluationMixin()
    assert not DependencyAnalyzer.__abstractmethods__


def test_static_analysis_never_executes_repository_raises():
    source = """raise AssertionError("source must not be executed")
AutoTokenizer.from_pretrained("example/tokenizer")
"""
    report = analyze_dependencies({"loader.py": source}, {}, "example/main")
    assert any(item["repo_id"] == "example/tokenizer" for item in report)


def test_analysis_budgets_and_unknown_representation_stay_stable():
    assert (MAX_CALL_DEPTH, MAX_CALL_CONTEXTS, MAX_CANDIDATES, MAX_ANALYSIS_STEPS) == (
        12, 256, 512, 100000
    )
    assert isinstance(ast.parse("x = 1"), ast.Module)
