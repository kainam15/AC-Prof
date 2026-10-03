"""Configuration failures carry stable fields across the CLI and run form."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from acprof.experiment import RunConfig, RunConfigError


class ConfigFieldErrorTests(unittest.TestCase):
    def test_list_number_and_path_errors_identify_their_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(RunConfig.smoke("fixture/model"), cpus="one", repeat="invalid",
                             workload_spec="missing-input.json")
            with self.assertRaises(RunConfigError) as caught:
                config.validate(project_dir=Path(directory))
        issues = caught.exception.issues
        self.assertEqual([issue.field for issue in issues], ["cpus", "repeat", "workload_spec"])
        self.assertTrue(all(issue.reason for issue in issues))
        self.assertIn("[cpus]", str(caught.exception))
        self.assertIn("[repeat]", str(caught.exception))

    def test_validity_bounds_have_field_ids_without_duplicate_parse_errors(self):
        cases = (("cpus", "0"), ("repeat", 0), ("warmup", -1), ("sample_hz", "nan"),
                 ("max_download", "bad"), ("model_store_max", "bad"))
        for name, value in cases:
            with self.subTest(field=name), self.assertRaises(RunConfigError) as caught:
                replace(RunConfig.smoke("fixture/model"), **{name: value}).validate()
            self.assertEqual([item.field for item in caught.exception.issues], [name])

    def test_model_spec_failure_retains_the_input_field(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RunConfigError) as caught:
                replace(RunConfig.smoke("fixture/model"), model_spec="missing.json").validate(project_dir=Path(directory))
        self.assertEqual(caught.exception.issues[0].field, "model_spec")
