"""Configuration failures carry stable fields across the CLI and run form."""
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from acprof.experiment import RunConfig, RunConfigError


def test_list_number_and_path_errors_identify_their_fields():
    with tempfile.TemporaryDirectory() as directory:
        config = replace(RunConfig.smoke("fixture/model"), cpus="one", repeat="invalid",
                         workload_spec="missing-input.json")
        with pytest.raises(RunConfigError) as caught:
            config.validate(project_dir=Path(directory))
    issues = caught.value.issues
    assert ([issue.field for issue in issues]) == (["cpus", "repeat", "workload_spec"])
    assert (all(issue.reason for issue in issues))
    assert ("[cpus]") in (str(caught.value))
    assert ("[repeat]") in (str(caught.value))

@pytest.mark.parametrize('name_case', range(6))
def test_validity_bounds_have_field_ids_without_duplicate_parse_errors(name_case):
    cases = (("cpus", "0"), ("repeat", 0), ("warmup", -1), ("sample_hz", "nan"),
             ("max_download", "bad"), ("model_store_max", "bad"))
    (name, value) = tuple(cases)[name_case]
    with pytest.raises(RunConfigError) as caught:
        replace(RunConfig.smoke("fixture/model"), **{name: value}).validate()
    assert ([item.field for item in caught.value.issues]) == ([name])

def test_model_spec_failure_retains_the_input_field():
    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(RunConfigError) as caught:
            replace(RunConfig.smoke("fixture/model"), model_spec="missing.json").validate(project_dir=Path(directory))
    assert (caught.value.issues[0].field) == ("model_spec")
