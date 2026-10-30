"""Local, project-scoped preferences for the terminal controller.

Reading preferences never creates or repairs a file. UI preferences and
experiment defaults are saved explicitly; the latest confirmed run/probe model
and used result paths are remembered automatically. The whitelisted dataclasses
contain no environment variables or credentials.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, TextIO, get_type_hints

from acprof.artifacts import atomic_write, read_json_object
from acprof.experiments.configuration import RunConfig
from acprof.tui.i18n import UI_LANGUAGES, error_message, message_id
from acprof.tui.ui.themes import UI_THEMES

LOG_MAX_LINES = (500, 1000, 3000, 10000)
SETTINGS_VERSION = 4


class UnsupportedSettingsError(ValueError):
    """设置版本不受支持，调用方不得静默回退或覆盖文件。"""


@dataclass(frozen=True)
class UiPreferences:
    theme: str = "acprof-graphite"
    log_wrap: bool = True
    log_max_lines: int = 3000
    show_command_bar: bool = True
    language: str = "zh"

    def validate(self) -> UiPreferences:
        _validate_field_types(asdict(self), UiPreferences, message_id('settings.preferences.section'))
        if self.language not in UI_LANGUAGES:
            raise ValueError(message_id('validation.settings.invalid_language'))
        if self.theme not in UI_THEMES:
            raise ValueError(message_id('validation.settings.invalid_theme'))
        if self.log_max_lines not in LOG_MAX_LINES:
            raise ValueError(message_id('validation.settings.invalid_log_limit'))
        return self


@dataclass(frozen=True)
class TuiSettings:
    ui: UiPreferences = field(default_factory=UiPreferences)
    run_defaults: RunConfig | None = None
    last_model: str = ""
    last_result_dir: str = ""
    last_result_csv: str = ""
    last_html_report: str = ""
    last_log_export_dir: str = ""

    def validate(self, *, project_dir: Path) -> TuiSettings:
        if not isinstance(self.ui, UiPreferences):
            raise ValueError(message_id('validation.settings.preferences_type'))
        ui = self.ui.validate()
        if type(self.last_model) is not str:
            raise ValueError(message_id('validation.settings.model_id_type'))
        if type(self.last_result_dir) is not str or type(self.last_result_csv) is not str:
            raise ValueError(message_id('validation.settings.result_path_type'))
        if type(self.last_html_report) is not str:
            raise ValueError(message_id('validation.settings.html_report_type'))
        if type(self.last_log_export_dir) is not str:
            raise ValueError(message_id('validation.settings.log_export_directory_type'))
        config = self.run_defaults
        if config is not None:
            if not isinstance(config, RunConfig):
                raise ValueError(message_id('validation.settings.run_defaults_type'))
            _validate_field_types(asdict(config), RunConfig, message_id('settings.defaults.section'))
            config = replace(config, rebuild_image=False, resume=False)
            # A reusable default need not select a model, but must otherwise
            # meet the same requirements as the live experiment form.
            model = config.model.strip()
            config = replace(config, model=model or "settings/default-model")
            try:
                config = config.validate(project_dir=project_dir)
            except OverflowError as exc:
                raise ValueError(message_id('validation.settings.numeric_range')) from exc
            config = replace(config, model=model)
        return replace(
            self, ui=ui, run_defaults=config, last_model=self.last_model.strip(),
            last_result_dir=self.last_result_dir.strip(),
            last_result_csv=self.last_result_csv.strip(),
            last_html_report=self.last_html_report.strip(),
            last_log_export_dir=self.last_log_export_dir.strip(),
        )


def default_settings_path(project_dir: Path) -> Path:
    """Return an XDG user-config path isolated by resolved project directory."""
    configured_root = os.environ.get("XDG_CONFIG_HOME", "")
    config_root = Path(configured_root).expanduser() if configured_root else None
    if config_root is None or not config_root.is_absolute():
        config_root = Path.home() / ".config"
    project = str(Path(project_dir).expanduser().resolve())
    project_key = hashlib.sha256(project.encode("utf-8")).hexdigest()[:16]
    return config_root / "acprof" / project_key / "tui.json"


def _validate_field_types(
    values: dict[str, Any], cls: type, label: str
) -> None:
    """Reject coercions, especially JSON booleans masquerading as integers."""
    allowed = {item.name for item in fields(cls)}
    if values.keys() - allowed:
        raise ValueError(message_id('validation.settings.unknown_fields', label))
    annotations = get_type_hints(cls)
    for name, value in values.items():
        expected = annotations[name]
        valid = (
            type(value) in (int, float)
            if expected is float
            else type(value) is expected
        )
        if not valid:
            raise ValueError(message_id('validation.settings.field_type', label, name))


def _decode_settings(payload: Any, project_dir: Path) -> TuiSettings:
    if not isinstance(payload, dict):
        raise ValueError(message_id('validation.settings.root_object'))
    version = payload.get("version")
    if type(version) is not int or version != SETTINGS_VERSION:
        raise UnsupportedSettingsError(
            f"不支持设置文件版本 {version!r}；当前要求 version={SETTINGS_VERSION}。"
            "请归档旧设置文件后重新配置。"
        )
    ui_values = payload.get("ui", {})
    if not isinstance(ui_values, dict):
        raise ValueError(message_id('validation.settings.preferences_object'))
    _validate_field_types(ui_values, UiPreferences, message_id('settings.preferences.section'))
    ui = UiPreferences(**ui_values)
    defaults_values = payload.get("run_defaults")
    defaults = None
    if defaults_values is not None:
        if not isinstance(defaults_values, dict):
            raise ValueError(message_id('validation.settings.defaults_object'))
        defaults_values = {key: value for key, value in defaults_values.items()
                           if key.lower() not in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}}
        if "allow_cgroup_v1" in defaults_values:
            raise UnsupportedSettingsError(
                "设置包含已删除的 allow_cgroup_v1；请归档旧设置文件后重新配置。"
            )
        if "skip_build" in defaults_values:
            if type(defaults_values.pop("skip_build")) is not bool:
                raise ValueError(message_id('validation.settings.retired_skip_build_type'))
        _validate_field_types(defaults_values, RunConfig, message_id('settings.defaults.section'))
        if defaults_values.get("download_mode") in {"mirror-only", "mirror-preferred", "official"}:
            defaults_values = {**defaults_values, "download_mode": "auto"}
        defaults = RunConfig(**defaults_values)
    # Ignore unknown top-level keys; only recognized fields can be saved again.
    return TuiSettings(
        ui=ui, run_defaults=defaults, last_model=payload.get("last_model", ""),
        last_result_dir=payload.get("last_result_dir", ""),
        last_result_csv=payload.get("last_result_csv", ""),
        last_html_report=payload.get("last_html_report", ""),
        last_log_export_dir=payload.get("last_log_export_dir", ""),
    ).validate(project_dir=project_dir)


def load_settings(path: Path, project_dir: Path) -> tuple[TuiSettings, str]:
    """Load valid preferences, or return defaults and a readable warning.

    Missing files are a normal first launch. Retired versions and fields raise
    without modifying the file; malformed current values retain a warning.
    """
    try:
        payload = read_json_object(path, label="TUI settings")
        settings = _decode_settings(payload, Path(project_dir))
    except UnsupportedSettingsError:
        raise
    except FileNotFoundError:
        return TuiSettings(), ""
    except (OSError, ValueError, UnicodeError) as exc:
        return TuiSettings(), message_id('settings.load.defaults_fallback', error_message(exc))
    return settings, ""


def save_settings(path: Path, settings: TuiSettings, project_dir: Path) -> None:
    """Validate and atomically replace a local preferences file.

    Raise ValueError for invalid settings and OSError for file-system failures.
    A failed validation or replacement leaves the previous file untouched.
    """
    if not isinstance(settings, TuiSettings):
        raise ValueError(message_id('validation.settings.settings_type'))
    normalized = settings.validate(project_dir=Path(project_dir))
    payload = {
        "version": SETTINGS_VERSION,
        "ui": asdict(normalized.ui),
        "last_model": normalized.last_model,
        "last_result_dir": normalized.last_result_dir,
        "last_result_csv": normalized.last_result_csv,
        "last_html_report": normalized.last_html_report,
        "last_log_export_dir": normalized.last_log_export_dir,
        "run_defaults": (
            {key: value for key, value in asdict(normalized.run_defaults).items()
             if key not in {"rebuild_image", "resume"}}
            if normalized.run_defaults is not None
            else None
        ),
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    destination = Path(path)

    def write(stream: TextIO) -> None:
        stream.write(serialized)

    atomic_write(destination, write, default_mode=0o600)
