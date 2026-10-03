import json
import os
import tempfile
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.experiment import RunConfig
from acprof.tui.settings import (
    SETTINGS_VERSION,
    TuiSettings,
    UiPreferences,
    default_settings_path,
    load_settings,
    save_settings,
)


class TestTuiSettings:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        scratch = Path(__file__).resolve().parents[1] / "internal-testing"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="tui-settings-", dir=scratch)
        self._request.addfinalizer(partial(self.temporary.cleanup))
        self.project = Path(self.temporary.name)
        self.path = self.project / "preferences" / "tui.json"

    def write_payload(self, payload):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    def test_settings_path_uses_xdg_and_resolved_project_identity(self):
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.project / "xdg")}):
            first = default_settings_path(self.project)
            equivalent = default_settings_path(self.project / "child" / "..")
            other = default_settings_path(self.project / "another-project")
        assert (first) == (equivalent)
        assert (first) != (other)
        assert (first.parent.parent) == (self.project / "xdg" / "acprof")
        assert (first.name) == ("tui.json")
        assert not (first.exists())

    @pytest.mark.parametrize('configured', ('', 'relative/path'))
    def test_empty_or_relative_xdg_uses_home_config(self, configured):
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": configured}):
            with patch("pathlib.Path.home", return_value=self.project):
                path = default_settings_path(self.project)
        assert (path.parent.parent) == (self.project / ".config" / "acprof")

    def test_first_launch_is_read_only_and_has_no_experiment_defaults(self):
        settings, warning = load_settings(self.path, self.project)
        assert (settings) == (TuiSettings())
        assert (settings.run_defaults) is None
        assert (warning) == ("")
        assert not (self.path.parent.exists())

    def test_roundtrip_ui_and_explicit_experiment_defaults(self):
        settings = TuiSettings(
            ui=UiPreferences(
                theme="acprof-light", log_wrap=False,
                log_max_lines=1000, show_command_bar=False, language="en",
            ),
            run_defaults=RunConfig.smoke("  demo/model  "),
            last_model="  demo/latest  ",
            last_result_dir="  results/模型 结果  ",
            last_result_csv="  results/另一个目录/custom.csv  ",
        )
        save_settings(self.path, settings, self.project)
        restored, warning = load_settings(self.path, self.project)
        assert (warning) == ("")
        assert (restored.ui) == (settings.ui)
        assert (restored.run_defaults) == (replace(settings.run_defaults, model="demo/model"))
        assert (restored.last_model) == ("demo/latest")
        assert (restored.last_result_dir) == ("results/模型 结果")
        assert (restored.last_result_csv) == ("results/另一个目录/custom.csv")
        assert (json.loads(self.path.read_text())["version"]) == (SETTINGS_VERSION)
        assert (self.path.stat().st_mode & 0o777) == (0o600)

    @pytest.mark.parametrize('version_fields_case', range(6), ids=['{}', "{'version': 1}", "{'version': 2}", "{'version': 3}", "{'version': True}", "{'version': SETTINGS_VERSION + 1}"])
    def test_old_settings_versions_fail_without_rewriting(self, version_fields_case):
        version_fields = tuple(({}, {'version': 1}, {'version': 2}, {'version': 3}, {'version': True}, {'version': SETTINGS_VERSION + 1}))[version_fields_case]
        self.write_payload({**version_fields, "run_defaults": {"model": "demo/saved"}})
        previous = self.path.read_bytes()
        with pytest.raises(ValueError, match="版本"):
            load_settings(self.path, self.project)
        assert (self.path.read_bytes()) == (previous)

    def test_empty_model_is_valid_for_defaults_but_other_validation_remains(self):
        settings = TuiSettings(run_defaults=RunConfig(model="  ", cpus="1, 2", gpus="OFF"))
        save_settings(self.path, settings, self.project)
        restored, warning = load_settings(self.path, self.project)
        assert (warning) == ("")
        assert (restored.run_defaults.model) == ("")
        assert (restored.run_defaults.cpus) == ("1,2")
        assert (restored.run_defaults.gpus) == ("off")
        previous = self.path.read_bytes()
        with pytest.raises(ValueError, match="CPU 列表"):
            save_settings(self.path, TuiSettings(run_defaults=RunConfig(cpus="0")), self.project)
        assert (self.path.read_bytes()) == (previous)

    @pytest.mark.parametrize('raw', (b'{broken', b'\xff\xfe'))
    def test_malformed_file_warns_without_overwriting_source(self, raw):
        self.path.parent.mkdir()
        self.path.write_bytes(raw)
        settings, warning = load_settings(self.path, self.project)
        assert (settings) == (TuiSettings())
        assert ("已使用默认值") in (warning)
        assert (self.path.read_bytes()) == (raw)

    @pytest.mark.parametrize("payload", [
        pytest.param([], id="not-object"),
        *(pytest.param({name: value}, id=f"{name}-{type(value).__name__}")
          for name, values in (
              ("last_model", (None, True, 5, [])),
              ("last_result_dir", (None, True, 5, [], {})),
              ("last_result_csv", (None, True, 5, [], {})),
          ) for value in values),
        pytest.param({"ui": []}, id="ui-not-object"),
        pytest.param({"ui": {"theme": "unknown"}}, id="unknown-theme"),
        pytest.param({"ui": {"language": "fr"}}, id="unsupported-language"),
        pytest.param({"ui": {"language": "EN"}}, id="uppercase-language"),
        pytest.param({"ui": {"language": True}}, id="bool-language"),
        pytest.param({"ui": {"language": None}}, id="null-language"),
        pytest.param({"ui": {"log_wrap": "false"}}, id="string-log-wrap"),
        pytest.param({"ui": {"show_command_bar": 0}}, id="integer-command-bar"),
        pytest.param({"ui": {"log_max_lines": True}}, id="bool-log-lines"),
        pytest.param({"ui": {"log_max_lines": 3000.0}}, id="float-log-lines"),
        pytest.param({"ui": {"log_max_lines": 5}}, id="small-log-lines"),
        pytest.param({"run_defaults": []}, id="defaults-not-object"),
        pytest.param({"run_defaults": {"model": 5}}, id="integer-model"),
        pytest.param({"run_defaults": {"skip_build": "false"}}, id="string-skip-build"),
        pytest.param({"run_defaults": {"repeat": True}}, id="bool-repeat"),
        pytest.param({"run_defaults": {"repeat": 1.5}}, id="float-repeat"),
        pytest.param({"run_defaults": {"sample_hz": True}}, id="bool-sample-rate"),
        pytest.param({"run_defaults": {"sample_hz": float("inf")}}, id="infinite-sample-rate"),
        pytest.param({"run_defaults": {"sample_hz": 10 ** 400}}, id="overflow-sample-rate"),
        pytest.param({"run_defaults": {"request_timeout_seconds": 0}}, id="zero-timeout"),
        pytest.param({"run_defaults": {"workload_spec": "missing-manifest.json"}}, id="missing-workload"),
        pytest.param({"run_defaults": {"token": "do-not-store"}}, id="unknown-nested-field"),
    ])
    def test_bad_types_and_values_fall_back_without_coercion(self, payload):
        self.write_payload({"version": SETTINGS_VERSION, **payload} if isinstance(payload, dict) else payload)
        restored, warning = load_settings(self.path, self.project)
        assert (restored) == (TuiSettings())
        assert (warning)

    def test_unknown_top_level_fields_are_discarded_and_not_saved(self):
        self.write_payload({"version": SETTINGS_VERSION, "ui": {"theme": "acprof-light"}, "token": "do-not-store"})
        settings, warning = load_settings(self.path, self.project)
        assert (warning) == ("")
        assert (settings.ui.theme) == ("acprof-light")
        save_settings(self.path, settings, self.project)
        assert ("token") not in (self.path.read_text(encoding="utf-8"))
        assert ("do-not-store") not in (self.path.read_text(encoding="utf-8"))

    @pytest.mark.parametrize('settings_case', range(9))
    def test_save_rejects_invalid_dataclasses_before_creating_file(self, settings_case):
        invalid = (
            TuiSettings(ui=UiPreferences(log_wrap=1)),
            TuiSettings(ui=UiPreferences(language="fr")),
            TuiSettings(ui=UiPreferences(log_max_lines=3)),
            TuiSettings(last_model=False),
            TuiSettings(last_result_dir=False),
            TuiSettings(last_result_csv=Path("result_all.csv")),
            TuiSettings(run_defaults=RunConfig(skip_build="false")),
            TuiSettings(run_defaults=RunConfig(repeat=True)),
            TuiSettings(run_defaults=RunConfig(sample_hz=10 ** 400)),
        )
        settings = tuple(invalid)[settings_case]
        with pytest.raises(ValueError):
            save_settings(self.path, settings, self.project)
        assert not (self.path.parent.exists())

    def test_failed_atomic_replace_preserves_old_file_and_removes_temp(self):
        save_settings(self.path, TuiSettings(), self.project)
        previous = self.path.read_bytes()
        with patch("acprof.tui.settings.os.replace", side_effect=OSError("disk error")):
            with pytest.raises(OSError, match="disk error"):
                save_settings(
                    self.path, TuiSettings(ui=UiPreferences(theme="acprof-light")), self.project
                )
        assert (self.path.read_bytes()) == (previous)
        assert (list(self.path.parent.iterdir())) == ([self.path])

    def test_read_io_error_returns_default_and_warning(self):
        self.path.mkdir(parents=True)
        settings, warning = load_settings(self.path, self.project)
        assert (settings) == (TuiSettings())
        assert ("已使用默认值") in (warning)
