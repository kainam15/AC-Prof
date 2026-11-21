"""Case paths use exact integer resources or inherited Process limits."""
from decimal import Decimal

import pytest

from acprof.results.layout import ArtifactLayout, resource_case_id


def test_adjacent_large_cases_have_separate_work_paths(tmp_path):
    layout = ArtifactLayout.for_new_run(tmp_path)
    first = layout.case("example/model", 2**53, 4, "off")
    second = layout.case("example/model", 2**53 + 1, 4, "off")
    assert first.case_id == "9007199254740992c_4g_off"
    assert second.case_id == "9007199254740993c_4g_off"
    assert first.csv != second.csv


@pytest.mark.parametrize("alias", [None, "", "NaN", "NONE", "inherited", float("nan")])
def test_case_label_preserves_inherited_and_exact_integer_aliases(alias):
    assert resource_case_id(alias, " 04.00e0 ", "on") == "inheritedc_4g_on"


def test_large_numeric_resources_keep_their_exact_value():
    assert resource_case_id("1e50", Decimal("4.0"), "off") == f"{10**50}c_4g_off"
    assert resource_case_id(1e23, 4, "off") == f"{int(1e23)}c_4g_off"


@pytest.mark.parametrize("value", [True, 0, -1, "1.0000000000000001", "3.5", "1e1000000000",
                                    "1e-1000000", "Infinity", [], "bad"])
def test_invalid_resource_cannot_be_used_as_a_case(value):
    with pytest.raises(ValueError, match="resource"):
        resource_case_id(1, value, "off")


@pytest.mark.parametrize("mode", ["yes", "../off", True])
def test_case_requires_recorded_gpu_mode(mode):
    with pytest.raises(ValueError, match="gpu"):
        resource_case_id(1, 4, mode)
